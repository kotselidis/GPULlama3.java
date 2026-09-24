@file:JvmName("TransformerComputeKernelsLayered")

package org.beehive.jitllm.backend.tornado.kernels

import uk.ac.manchester.tornado.api.KernelContext
import uk.ac.manchester.tornado.kotlin.api.parallelFor
import uk.ac.manchester.tornado.api.math.TornadoMath
import uk.ac.manchester.tornado.api.types.HalfFloat
import uk.ac.manchester.tornado.api.types.arrays.ByteArray
import uk.ac.manchester.tornado.api.types.arrays.FloatArray
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray
import uk.ac.manchester.tornado.api.types.arrays.Int8Array
import uk.ac.manchester.tornado.api.types.arrays.IntArray
import uk.ac.manchester.tornado.api.types.vectors.Half2



fun fusedQKvBiasAddition(
        context: KernelContext,
        q_out: FloatArray,
        k_out: FloatArray,
        qBias: FloatArray,
        v_out: FloatArray,
        kBias: FloatArray,
        vBias: FloatArray,
        dimQ: Int,
        dimKV: Int,
) {

    var gid: Int = context.globalIdx

    if (gid < dimQ) {
        // 1. Add Q bias
        q_out.set(gid, q_out.get(gid) + qBias.get(gid))

        // 2. Conditionally Add K and V Bias
        if (gid < dimKV) {
            k_out.set(gid, k_out.get(gid) + kBias.get(gid))
            v_out.set(gid, v_out.get(gid) + vBias.get(gid))
        }
    }
}

fun fusedRmsNormFFNGateUp(
        context: KernelContext,
        x: FloatArray, // raw input (FP32)
        hb: FloatArray, // output
        rmsWeights: FloatArray, // RMS norm weights
        rmsScale: FloatArray, // temp[0] = scale factor
        w1: HalfFloatArray,
        w3: HalfFloatArray,
        dim: Int, // input dimension
        hiddenDim: Int, // output dimension
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= hiddenDim) {
        return
    }

    var scale: Float = rmsScale.get(0)

    // Allocate shared memory for normalized input (reused for both W1 and W3)
    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localWorkGroupSize)

    var rowOffsetW1: Int = rowId * dim
    var rowOffsetW3: Int = rowId * dim

    // === W1 matmul with inline normalization ===
    // Weights are read as packed FP16 pairs (single 32-bit loads); dim is even for all
    // supported models, so the pair indices stay even as getHalf2 requires.
    var sum1: Float = 0.0f
    run {
        var j: Int = localId * 2
        while (j < dim) {
            var normalized0: Float = rmsWeights.get(j) * scale * x.get(j)
            var normalized1: Float = rmsWeights.get(j + 1) * scale * x.get(j + 1)
            var pairW1: Half2 = w1.getHalf2(rowOffsetW1 + j)
            sum1 += Half2.lowFloat(pairW1) * normalized0
            sum1 += Half2.highFloat(pairW1) * normalized1
            j += localWorkGroupSize * 2
        }
    }

    localSum[localId] = sum1
    context.localBarrier()

    run {
        var stride: Int = localWorkGroupSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }
    var result1: Float = localSum[0]

    // === W3 matmul with inline normalization (same computation) ===
    var sum3: Float = 0.0f
    run {
        var j: Int = localId * 2
        while (j < dim) {
            var normalized0: Float = rmsWeights.get(j) * scale * x.get(j)
            var normalized1: Float = rmsWeights.get(j + 1) * scale * x.get(j + 1)
            var pairW3: Half2 = w3.getHalf2(rowOffsetW3 + j)
            sum3 += Half2.lowFloat(pairW3) * normalized0
            sum3 += Half2.highFloat(pairW3) * normalized1
            j += localWorkGroupSize * 2
        }
    }

    localSum[localId] = sum3
    context.localBarrier()

    run {
        var stride: Int = localWorkGroupSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }
    var result3: Float = localSum[0]

    // === SiLU + GLU ===
    if (localId == 0) {
        var silu: Float = result1 / (1.0f + TornadoMath.exp(-result1))
        hb.set(rowId, silu * result3)
    }
}

/**
 * Warp-shuffle variant of {@link #fusedRmsNormFFNGateUp}. Assumes a 32-lane workgroup per
 * output row (the decode worker shape). Reduces each row's dot product with {@code
 * simdShuffleDown} instead of a shared-memory tree, eliminating the per-row barriers and shared
 * round-trip — the same reduction strategy as llama.cpp's {@code mul_mat_vec}. Used on CUDA
 * only (see {@code SchedulerDetectionService.isWarpShuffleSupported}); OpenCL miscompiles the
 * shuffle.
 */
fun fusedRmsNormFFNGateUpWarp(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        rmsWeights: FloatArray,
        rmsScale: FloatArray,
        w1: HalfFloatArray,
        w3: HalfFloatArray,
        dim: Int,
        hiddenDim: Int,
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= hiddenDim) {
        return
    }

    var scale: Float = rmsScale.get(0)
    var rowOffset: Int = rowId * dim

    var sum1: Float = 0.0f
    var sum3: Float = 0.0f
    run {
        var j: Int = localId
        while (j < dim) {
            var normalized: Float = rmsWeights.get(j) * scale * x.get(j)
            sum1 += w1.get(rowOffset + j).getFloat32() * normalized
            sum3 += w3.get(rowOffset + j).getFloat32() * normalized
            j += 32
        }
    }

    sum1 += context.simdShuffleDown(sum1, 16)
    sum1 += context.simdShuffleDown(sum1, 8)
    sum1 += context.simdShuffleDown(sum1, 4)
    sum1 += context.simdShuffleDown(sum1, 2)
    sum1 += context.simdShuffleDown(sum1, 1)

    sum3 += context.simdShuffleDown(sum3, 16)
    sum3 += context.simdShuffleDown(sum3, 8)
    sum3 += context.simdShuffleDown(sum3, 4)
    sum3 += context.simdShuffleDown(sum3, 2)
    sum3 += context.simdShuffleDown(sum3, 1)

    if (localId == 0) {
        var silu: Float = sum1 / (1.0f + TornadoMath.exp(-sum1))
        hb.set(rowId, silu * sum3)
    }
}

/**
 * Warp-shuffle variant of {@link #fusedRmsNormFFNGateUpQ8_0}: one 32-lane warp per hidden row,
 * two independent {@code simdShuffleDown} reductions (gate W1 and up W3), no shared-memory
 * tree. Q8_0 byte layout (34-byte blocks: 2-byte half scale + 32 int8 quants).
 */
fun fusedRmsNormFFNGateUpQ8_0Warp(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        rmsWeights: FloatArray,
        rmsScale: FloatArray,
        w1: ByteArray,
        w3: ByteArray,
        inputDim: Int,
        hiddenDim: Int,
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= hiddenDim) {
        return
    }

    var scale: Float = rmsScale.get(0)
    var blockSize: Int = 32
    var Q8_0_BLOCK_BYTES: Int = 34 // 2-byte scale + 32 int8 quants
    var blocksPerRow: Int = (inputDim + blockSize - 1) / blockSize
    var rowBlockOffset: Int = rowId * blocksPerRow

    var sum1: Float = 0.0f
    var sum3: Float = 0.0f
    run {
        var j: Int = localId
        while (j < inputDim) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j - blockIdx * blockSize
            var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
            var w1s: Float = w1.getHalfFloat(blockByteOffset).getFloat32()
            var w3s: Float = w3.getHalfFloat(blockByteOffset).getFloat32()
            var w1q: Byte = w1.get(blockByteOffset + 2 + withinBlockIdx)
            var w3q: Byte = w3.get(blockByteOffset + 2 + withinBlockIdx)
            var normalized: Float = rmsWeights.get(j) * (scale * x.get(j))
            sum1 += (w1q.toFloat() * w1s) * normalized
            sum3 += (w3q.toFloat() * w3s) * normalized
            j += 32
        }
    }

    sum1 += context.simdShuffleDown(sum1, 16)
    sum1 += context.simdShuffleDown(sum1, 8)
    sum1 += context.simdShuffleDown(sum1, 4)
    sum1 += context.simdShuffleDown(sum1, 2)
    sum1 += context.simdShuffleDown(sum1, 1)

    sum3 += context.simdShuffleDown(sum3, 16)
    sum3 += context.simdShuffleDown(sum3, 8)
    sum3 += context.simdShuffleDown(sum3, 4)
    sum3 += context.simdShuffleDown(sum3, 2)
    sum3 += context.simdShuffleDown(sum3, 1)

    if (localId == 0) {
        var silu: Float = sum1 / (1.0f + TornadoMath.exp(-sum1))
        hb.set(rowId, silu * sum3)
    }
}

/**
 * Fused RMSNorm apply + Gate/Up projection + SiLU + GLU for Q8_0 weights. Combines:
 * reductionOneBlock2WithLayer + fusedFeedForwardWithSiLUAndGLUActivationQ8_0Byte
 */
fun fusedRmsNormFFNGateUpQ8_0(
        context: KernelContext,
        x: FloatArray, // raw input (FP32)
        hb: FloatArray, // output: SiLU(x·W1) ⊙ (x·W3)
        rmsWeights: FloatArray, // RMS norm weights
        rmsScale: FloatArray, // tempFFN[0] = scale factor
        w1: ByteArray, // W1 (gate) Q8_0 weights
        w3: ByteArray, // W3 (up) Q8_0 weights
        inputDim: Int, // input dimension
        hiddenDim: Int, // hidden dimension
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= hiddenDim) {
        return
    }

    var scale: Float = rmsScale.get(0)
    var blockSize: Int = 32
    var Q8_0_BLOCK_BYTES: Int = 34 // 2 bytes scale + 32 bytes quants

    // Allocate local memory for reduction
    var localSums: kotlin.FloatArray = context.allocateFloatLocalArray(localWorkGroupSize)

    // Calculate block offsets for W1 and W3 matrices
    var blocksPerRow: Int = (inputDim + blockSize - 1) / blockSize
    var w1RowBlockOffset: Int = rowId * blocksPerRow
    var w3RowBlockOffset: Int = rowId * blocksPerRow

    // ========== W1 computation with inline RMS normalization ==========
    var partialSum1_1: Float = 0.0f
    var partialSum1_2: Float = 0.0f
    var partialSum1_3: Float = 0.0f
    var partialSum1_4: Float = 0.0f

    // Main loop with 4-way unrolling for W1
    run {
        var j: Int = localId * 4
        while (j < inputDim - 3) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize

            // W1 block access
            var w1BlockByteOffset: Int = (w1RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
            var w1Scale: HalfFloat = w1.getHalfFloat(w1BlockByteOffset)
            var w1ScaleFloat: Float = w1Scale.getFloat32()

            var w1QuantsOffset: Int = w1BlockByteOffset + 2 + withinBlockIdx
            var w1Quant1: Byte = w1.get(w1QuantsOffset)
            var w1Quant2: Byte = w1.get(w1QuantsOffset + 1)
            var w1Quant3: Byte = w1.get(w1QuantsOffset + 2)
            var w1Quant4: Byte = w1.get(w1QuantsOffset + 3)

            // Apply RMS normalization inline (equivalent to reductionOneBlock2WithLayer)
            var norm1: Float = rmsWeights.get(j) * (scale * x.get(j))
            var norm2: Float = rmsWeights.get(j + 1) * (scale * x.get(j + 1))
            var norm3: Float = rmsWeights.get(j + 2) * (scale * x.get(j + 2))
            var norm4: Float = rmsWeights.get(j + 3) * (scale * x.get(j + 3))

            partialSum1_1 += (w1Quant1.toFloat() * w1ScaleFloat) * norm1
            partialSum1_2 += (w1Quant2.toFloat() * w1ScaleFloat) * norm2
            partialSum1_3 += (w1Quant3.toFloat() * w1ScaleFloat) * norm3
            partialSum1_4 += (w1Quant4.toFloat() * w1ScaleFloat) * norm4
            j += localWorkGroupSize * 4
        }
    }

    var partialSum1: Float = partialSum1_1 + partialSum1_2 + partialSum1_3 + partialSum1_4

    // Handle remaining elements for W1
    run {
        var j: Int = ((inputDim / 4) * 4) + localId
        while (j < inputDim) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize

            var w1BlockByteOffset: Int = (w1RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
            var w1Scale: HalfFloat = w1.getHalfFloat(w1BlockByteOffset)
            var w1ScaleFloat: Float = w1Scale.getFloat32()

            var w1Quant: Byte = w1.get(w1BlockByteOffset + 2 + withinBlockIdx)
            var normalized: Float = rmsWeights.get(j) * (scale * x.get(j))

            partialSum1 += (w1Quant.toFloat() * w1ScaleFloat) * normalized
            j += localWorkGroupSize
        }
    }

    localSums[localId] = partialSum1
    context.localBarrier()

    // Parallel reduction for W1
    run {
        var stride: Int = localWorkGroupSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    var sum1: Float = localSums[0]

    // ========== W3 computation with inline RMS normalization ==========
    var partialSum3_1: Float = 0.0f
    var partialSum3_2: Float = 0.0f
    var partialSum3_3: Float = 0.0f
    var partialSum3_4: Float = 0.0f

    // Main loop with 4-way unrolling for W3
    run {
        var j: Int = localId * 4
        while (j < inputDim - 3) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize

            // W3 block access
            var w3BlockByteOffset: Int = (w3RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
            var w3Scale: HalfFloat = w3.getHalfFloat(w3BlockByteOffset)
            var w3ScaleFloat: Float = w3Scale.getFloat32()

            var w3QuantsOffset: Int = w3BlockByteOffset + 2 + withinBlockIdx
            var w3Quant1: Byte = w3.get(w3QuantsOffset)
            var w3Quant2: Byte = w3.get(w3QuantsOffset + 1)
            var w3Quant3: Byte = w3.get(w3QuantsOffset + 2)
            var w3Quant4: Byte = w3.get(w3QuantsOffset + 3)

            // Apply RMS normalization inline (same computation as W1)
            var norm1: Float = rmsWeights.get(j) * (scale * x.get(j))
            var norm2: Float = rmsWeights.get(j + 1) * (scale * x.get(j + 1))
            var norm3: Float = rmsWeights.get(j + 2) * (scale * x.get(j + 2))
            var norm4: Float = rmsWeights.get(j + 3) * (scale * x.get(j + 3))

            partialSum3_1 += (w3Quant1.toFloat() * w3ScaleFloat) * norm1
            partialSum3_2 += (w3Quant2.toFloat() * w3ScaleFloat) * norm2
            partialSum3_3 += (w3Quant3.toFloat() * w3ScaleFloat) * norm3
            partialSum3_4 += (w3Quant4.toFloat() * w3ScaleFloat) * norm4
            j += localWorkGroupSize * 4
        }
    }

    var partialSum3: Float = partialSum3_1 + partialSum3_2 + partialSum3_3 + partialSum3_4

    // Handle remaining elements for W3
    run {
        var j: Int = ((inputDim / 4) * 4) + localId
        while (j < inputDim) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize

            var w3BlockByteOffset: Int = (w3RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
            var w3Scale: HalfFloat = w3.getHalfFloat(w3BlockByteOffset)
            var w3ScaleFloat: Float = w3Scale.getFloat32()

            var w3Quant: Byte = w3.get(w3BlockByteOffset + 2 + withinBlockIdx)
            var normalized: Float = rmsWeights.get(j) * (scale * x.get(j))

            partialSum3 += (w3Quant.toFloat() * w3ScaleFloat) * normalized
            j += localWorkGroupSize
        }
    }

    localSums[localId] = partialSum3
    context.localBarrier()

    // Parallel reduction for W3
    run {
        var stride: Int = localWorkGroupSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    var sum3: Float = localSums[0]

    // ========== SiLU + GLU (same as original) ==========
    if (localId == 0) {
        var silu: Float = siluActivation(sum1)
        var result: Float = silu * sum3
        hb.set(rowId, result)
    }
}

/**
 * Performs RMS (Root Mean Square) normalization using parallel reduction. This is the first
 * phase of RMS normalization that computes the variance and scaling factor across all work
 * groups.
 *
 * <p>Algorithm: 1. Each thread computes square of its input element 2. Work group performs
 * parallel reduction of squares 3. Partial sums stored per work group 4. First thread combines
 * all partial sums and computes normalization factor
 *
 * @param context Kernel execution context
 * @param output Array to store partial sums and final normalization factor
 * @param x Input array to normalize
 * @param size Number of elements to process
 * @param ermsNorm Epsilon value squared for numerical stability
 * @param localMemSize Size of local memory allocation (must match work group size)
 */
fun reductionOneBlockWithLayer(
        context: KernelContext,
        output: FloatArray,
        x: FloatArray,
        size: Int,
        ermsNorm: Float,
        localMemSize: Int,
) {
    var gid: Int = context.globalIdx
    var lid: Int = context.localIdx
    var groupId: Int = context.groupIdx
    var groupSize: Int = context.localGroupSizeX

    // Allocate local memory with the provided size
    var localX: kotlin.FloatArray = context.allocateFloatLocalArray(localMemSize)

    // Load input value and compute square
    if (gid < size) {
        localX[lid] = x.get(gid)
        localX[lid] = localX[lid] * localX[lid]
    } else {
        localX[lid] = 0.0f
    }

    // Perform parallel reduction within the work group
    run {
        var stride: Int = (groupSize / 2)
        while (stride > 0) {
            context.localBarrier()
            if (lid < stride) {
                localX[lid] += localX[lid + stride]
            }
            stride /= 2
        }
    }

    // Each workgroup stores its partial sum in a different location
    if (lid == 0) {
        // Store the partial sum from each workgroup
        output.set(groupId + 1, localX[0])
    }

    // Only the first thread in the first workgroup computes the final normalization factor
    if (gid == 0) {
        // Combine partial sums from all workgroups
        var ss: Float = 0.0f
        run {
            var i: Int = 1
            while (i <= (size / localMemSize)) {  // Assuming 8 workgroups
                ss += output.get(i)
                i++
            }
        }

        ss /= size
        ss += ermsNorm
        ss = 1.0f / TornadoMath.sqrt(ss)
        output.set(0, ss); // Store the final scale factor
    }
}

/**
 * Applies the computed normalization factor to input and weight elements. This is the second
 * phase of RMS normalization.
 *
 * <p>Formula: output[i] = weight[i] * (normalizationFactor * x[i])
 *
 * @param context Kernel execution context
 * @param output Array for normalized output
 * @param x Input values to normalize
 * @param weights Weight values for each element
 * @param temp Temporary array containing normalization factor at index 0
 */
fun reductionOneBlock2WithLayer(
        context: KernelContext,
        output: FloatArray,
        x: FloatArray,
        weights: FloatArray,
        temp: FloatArray,
) {
    var gid: Int = context.globalIdx

    var ss: Float = temp.get(0)
    output.set(gid, weights.get(gid) * (ss * x.get(gid)))
}

fun splitQKV(
        qkv: FloatArray,
        q: FloatArray,
        k: FloatArray,
        v: FloatArray,
        dimQ: Int,
        dimKV: Int,
) {
    var totalSize: Int = dimQ + 2 * dimKV

    parallelFor(0, totalSize) { i ->
        if (i < dimQ) {
            // Copy to Q
            q.set(i, qkv.get(i))
        } else if (i < dimQ + dimKV) {
            // Copy to K
            var kIndex: Int = i - dimQ
            k.set(kIndex, qkv.get(i))
        } else {
            // Copy to V
            var vIndex: Int = i - dimQ - dimKV
            v.set(vIndex, qkv.get(i))
        }
    }
}

fun ropeRotationPhi3(
        context: KernelContext,
        positionHolder: IntArray,
        sq: FloatArray,
        sk: FloatArray,
        kv_dim: Int,
        head_size: Int,
) {
    var idx: Int = context.globalIdx

    // For Phi3, we process pairs with offset of head_size/2
    var dimHalf: Int = head_size / 2

    // Each thread processes one dimension pair
    if (idx >= dimHalf) {
        return
    }

    var position: Int = positionHolder.get(0)

    // Calculate frequency for this dimension
    var freq: Float = 1.0f / TornadoMath.pow(10000.0f, (idx * 2).toFloat() / head_size.toFloat())
    var `val`: Float = position * freq
    var fcr: Float = TornadoMath.cos(`val`)
    var fci: Float = TornadoMath.sin(`val`)

    // Process all heads
    var totalDim: Int = sq.getSize()
    run {
        var base: Int = 0
        while (base < totalDim) {
            // Skip if we're beyond the bounds
            if (base + idx >= totalDim || base + idx + dimHalf >= totalDim) {
                break
            }

            // Rotate query
            var v0: Float = sq.get(base + idx)
            var v1: Float = sq.get(base + idx + dimHalf)
            sq.set(base + idx, v0 * fcr - v1 * fci)
            sq.set(base + idx + dimHalf, v0 * fci + v1 * fcr)

            // Rotate key if within kv_dim
            if (base < kv_dim && base + idx < sk.getSize() && base + idx + dimHalf < sk.getSize()) {
                var k0: Float = sk.get(base + idx)
                var k1: Float = sk.get(base + idx + dimHalf)
                sk.set(base + idx, k0 * fcr - k1 * fci)
                sk.set(base + idx + dimHalf, k0 * fci + k1 * fcr)
            }
            base += head_size
        }
    }
}

/**
 * Qwen3-family decode attention, split-KV (flash-decoding) phase 2: combine.
 *
 * <p>One workgroup per query head. Reads the {@code nSplits} partial states ({@code out_s, M_s,
 * L_s}) produced by {@link #processHeadsFlashAttentionSplitKV} from {@code att} ({@code
 * wrapAttSplit}) and merges them with the standard online-softmax rule: global max {@code Mg =
 * max_s M_s}, scale {@code f_s = exp(M_s - Mg)}, output {@code xb[d] = Σ_s f_s·out_s[d] / Σ_s
 * f_s·L_s}. Writes the final attention result to {@code xb}.
 */
fun combineSplitKVAttention(
        context: KernelContext,
        att: FloatArray,
        xb: FloatArray,
        nHeads: Int,
        headSize: Int,
        nSplits: Int,
) {

    var MAX_SPLITS: Int = 64

    var tid: Int = context.localIdx
    var h: Int = context.groupIdx
    var localSize: Int = context.localGroupSizeX

    if (h >= nHeads) {
        return
    }

    // Must match the COMPACT layout written by processHeadsFlashAttentionSplitKV.
    var headBase: Int = h * nSplits * (headSize + 2)
    var mBase: Int = headBase + nSplits * headSize
    var lBase: Int = mBase + nSplits

    var fShared: kotlin.FloatArray = context.allocateFloatLocalArray(MAX_SPLITS)
    var bcast: kotlin.FloatArray = context.allocateFloatLocalArray(1)

    // Global max over split maxima.
    if (tid == 0) {
        var gMax: Float = Float.NEGATIVE_INFINITY
        run {
            var s: Int = 0
            while (s < nSplits) {
                var ms: Float = att.get(mBase + s)
                if (ms > gMax) {
                    gMax = ms
                }
                s++
            }
        }
        bcast[0] = gMax
    }
    context.localBarrier()
    var gMax: Float = bcast[0]

    // Per-split scale factors f_s = exp(M_s - gMax).
    run {
        var s: Int = tid
        while (s < nSplits) {
            var ms: Float = att.get(mBase + s)
            fShared[s] = if (ms == Float.NEGATIVE_INFINITY) 0.0f else TornadoMath.exp(ms - gMax)
            s += localSize
        }
    }
    context.localBarrier()

    // Global denominator Σ_s f_s · L_s.
    if (tid == 0) {
        var denom: Float = 0.0f
        run {
            var s: Int = 0
            while (s < nSplits) {
                denom += fShared[s] * att.get(lBase + s)
                s++
            }
        }
        bcast[0] = denom
    }
    context.localBarrier()
    var denom: Float = bcast[0]
    var inv: Float = if (denom > 0.0f) (1.0f / denom) else 0.0f

    // Merge per-dimension numerators across splits and normalize.
    run {
        var d: Int = tid
        while (d < headSize) {
            var acc: Float = 0.0f
            run {
                var s: Int = 0
                while (s < nSplits) {
                    acc += fShared[s] * att.get(headBase + s * headSize + d)
                    s++
                }
            }
            xb.set(h * headSize + d, acc * inv)
            d += localSize
        }
    }
}

/**
 * Performs optimized matrix-vector multiplication where each work group processes one row of
 * the matrix.
 *
 * <p>Algorithm: 1. Each work group handles one output dimension 2. Threads in work group
 * compute partial dot products 3. Parallel reduction yields final row result
 *
 * @param context Kernel execution context
 * @param x Input vector
 * @param hb Output vector
 * @param w Weight matrix (row-major)
 * @param n Input dimension
 * @param d Output dimension
 * @param localWorkGroupSize Number of threads per work group
 */
fun matrixVectorGeneric(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        w: FloatArray,
        n: Int,
        d: Int,
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var localSize: Int = localWorkGroupSize

    // Early exit if this workgroup is beyond our output dimension
    if (rowId >= d) {
        return
    }
    var sum: Float = matrixVectorRowMajorOptimized(context, localSize, x, w, n)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        hb.set(rowId, sum)
    }
}

// @formatter:off
fun matrixVectorGeneric(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray, // output
        w: HalfFloatArray,
        dim1: Int, // inner loop
        dim0: Int, // outer loop
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var localSize: Int = localWorkGroupSize

    // Early exit if this workgroup is beyond our output dimension
    if (rowId >= dim0) {
        return
    }
    var sum: Float = matrixVectorRowMajorOptimized(context, localSize, x, w, dim1)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        hb.set(rowId, sum)
    }
}

/**
 * Fused Q/K/V matrix-vector multiplication. Reduces kernel launch overhead and improves input
 * vector cache utilization.
 *
 * <p>Workgroup assignment: - rowId [0, dim): Q projection - rowId [dim, dim+kvDim): K
 * projection - rowId [dim+kvDim, dim+2*kvDim): V projection
 */
fun fusedQKVMatmulX(
        context: KernelContext,
        x: HalfFloatArray, // input vector (FP16)
        q: FloatArray, // output Q (FP32)
        k: FloatArray, // output K (FP32)
        v: FloatArray, // output V (FP32)
        wq: HalfFloatArray, // Q weight matrix
        wk: HalfFloatArray, // K weight matrix
        wv: HalfFloatArray, // V weight matrix
        dim: Int, // model dimension (Q output size)
        kvDim: Int, // KV dimension (K/V output size)
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    // Allocate local memory for reduction
    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localWorkGroupSize)

    if (rowId < dim) {
        // ========== Q projection ==========
        var rowOffset: Int = rowId * dim

        var partialSum: Float = 0.0f
        // Packed FP16 pair loads for both the weight row and the FP16 input, multiplied
        // with the packed __hmul2 intrinsic; accumulation stays FP32. dim is even.
        run {
            var j: Int = localId * 2
            while (j < dim) {
                var product: Half2 = Half2.mult(wq.getHalf2(rowOffset + j), x.getHalf2(j))
                partialSum += Half2.lowFloat(product)
                partialSum += Half2.highFloat(product)
                j += localWorkGroupSize * 2
            }
        }

        localSum[localId] = partialSum
        context.localBarrier()

        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }

        if (localId == 0) {
            q.set(rowId, localSum[0])
        }

    } else if (rowId < dim + kvDim) {
        // ========== K projection ==========
        var kRow: Int = rowId - dim
        var rowOffset: Int = kRow * dim

        var partialSum: Float = 0.0f
        // Packed FP16 pair loads for both the weight row and the FP16 input, multiplied
        // with the packed __hmul2 intrinsic; accumulation stays FP32. dim is even.
        run {
            var j: Int = localId * 2
            while (j < dim) {
                var product: Half2 = Half2.mult(wk.getHalf2(rowOffset + j), x.getHalf2(j))
                partialSum += Half2.lowFloat(product)
                partialSum += Half2.highFloat(product)
                j += localWorkGroupSize * 2
            }
        }

        localSum[localId] = partialSum
        context.localBarrier()

        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }

        if (localId == 0) {
            k.set(kRow, localSum[0])
        }

    } else if (rowId < dim + 2 * kvDim) {
        // ========== V projection ==========
        var vRow: Int = rowId - dim - kvDim
        var rowOffset: Int = vRow * dim

        var partialSum: Float = 0.0f
        // Packed FP16 pair loads for both the weight row and the FP16 input, multiplied
        // with the packed __hmul2 intrinsic; accumulation stays FP32. dim is even.
        run {
            var j: Int = localId * 2
            while (j < dim) {
                var product: Half2 = Half2.mult(wv.getHalf2(rowOffset + j), x.getHalf2(j))
                partialSum += Half2.lowFloat(product)
                partialSum += Half2.highFloat(product)
                j += localWorkGroupSize * 2
            }
        }

        localSum[localId] = partialSum
        context.localBarrier()

        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }

        if (localId == 0) {
            v.set(vRow, localSum[0])
        }
    }
}

/**
 * {@link #fusedQKVMatmulX} with the products taken in FP32 instead of packed FP16.
 *
 * <p>Identical structure and identical work decomposition; the only difference is that each
 * FP16 pair is widened before it is multiplied, rather than multiplied by {@code __hmul2} and
 * widened afterwards. That rounds every product to FP16 before it reaches the FP32 accumulator,
 * once per term, and over a 2048-term row the loss is systematic rather than cancelling.
 *
 * <p>Selected where {@code DeviceCapability.PACKED_HALF2_MATH} is withheld. It is measurably
 * more accurate everywhere; it is not the default because the packed form is faster where it
 * has been shown to hold parity.
 */
fun fusedQKVMatmulXFp32Products(
        context: KernelContext,
        x: HalfFloatArray,
        q: FloatArray,
        k: FloatArray,
        v: FloatArray,
        wq: HalfFloatArray,
        wk: HalfFloatArray,
        wv: HalfFloatArray,
        dim: Int,
        kvDim: Int,
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localWorkGroupSize)

    if (rowId < dim) {
        var rowOffset: Int = rowId * dim
        var partialSum: Float = 0.0f
        run {
            var j: Int = localId
            while (j < dim) {
                partialSum += wq.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
                j += localWorkGroupSize
            }
        }
        localSum[localId] = partialSum
        context.localBarrier()
        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }
        if (localId == 0) {
            q.set(rowId, localSum[0])
        }

    } else if (rowId < dim + kvDim) {
        var kRow: Int = rowId - dim
        var rowOffset: Int = kRow * dim
        var partialSum: Float = 0.0f
        run {
            var j: Int = localId
            while (j < dim) {
                partialSum += wk.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
                j += localWorkGroupSize
            }
        }
        localSum[localId] = partialSum
        context.localBarrier()
        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }
        if (localId == 0) {
            k.set(kRow, localSum[0])
        }

    } else if (rowId < dim + 2 * kvDim) {
        var vRow: Int = rowId - dim - kvDim
        var rowOffset: Int = vRow * dim
        var partialSum: Float = 0.0f
        run {
            var j: Int = localId
            while (j < dim) {
                partialSum += wv.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
                j += localWorkGroupSize
            }
        }
        localSum[localId] = partialSum
        context.localBarrier()
        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }
        if (localId == 0) {
            v.set(vRow, localSum[0])
        }
    }
}

/**
 * CUDA/Metal SIMD variant of fusedQKVMatmulX for LLaMA FP16 decode. It assumes the existing
 * decode worker shape: one 32-lane workgroup per Q/K/V output row.
 */
fun fusedQKVMatmulXSimd32(
        context: KernelContext,
        x: HalfFloatArray,
        q: FloatArray,
        k: FloatArray,
        v: FloatArray,
        wq: HalfFloatArray,
        wk: HalfFloatArray,
        wv: HalfFloatArray,
        dim: Int,
        kvDim: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId < dim) {
        var rowOffset: Int = rowId * dim
        var partialSum: Float = 0.0f
        run {
            var j: Int = localId
            while (j < dim) {
                partialSum += wq.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
                j += 32
            }
        }
        partialSum += context.simdShuffleDown(partialSum, 16)
        partialSum += context.simdShuffleDown(partialSum, 8)
        partialSum += context.simdShuffleDown(partialSum, 4)
        partialSum += context.simdShuffleDown(partialSum, 2)
        partialSum += context.simdShuffleDown(partialSum, 1)
        if (localId == 0) {
            q.set(rowId, partialSum)
        }

    } else if (rowId < dim + kvDim) {
        var kRow: Int = rowId - dim
        var rowOffset: Int = kRow * dim
        var partialSum: Float = 0.0f
        run {
            var j: Int = localId
            while (j < dim) {
                partialSum += wk.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
                j += 32
            }
        }
        partialSum += context.simdShuffleDown(partialSum, 16)
        partialSum += context.simdShuffleDown(partialSum, 8)
        partialSum += context.simdShuffleDown(partialSum, 4)
        partialSum += context.simdShuffleDown(partialSum, 2)
        partialSum += context.simdShuffleDown(partialSum, 1)
        if (localId == 0) {
            k.set(kRow, partialSum)
        }

    } else if (rowId < dim + 2 * kvDim) {
        var vRow: Int = rowId - dim - kvDim
        var rowOffset: Int = vRow * dim
        var partialSum: Float = 0.0f
        run {
            var j: Int = localId
            while (j < dim) {
                partialSum += wv.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
                j += 32
            }
        }
        partialSum += context.simdShuffleDown(partialSum, 16)
        partialSum += context.simdShuffleDown(partialSum, 8)
        partialSum += context.simdShuffleDown(partialSum, 4)
        partialSum += context.simdShuffleDown(partialSum, 2)
        partialSum += context.simdShuffleDown(partialSum, 1)
        if (localId == 0) {
            v.set(vRow, partialSum)
        }
    }
}

/** Fused QKV matmul for FP16 models where Q output dim != input dim. */
fun fusedQKVMatmulXNonSquare(
        context: KernelContext,
        x: HalfFloatArray,
        q: FloatArray,
        k: FloatArray,
        v: FloatArray,
        wq: HalfFloatArray,
        wk: HalfFloatArray,
        wv: HalfFloatArray,
        inputDim: Int,
        qDim: Int,
        kvDim: Int,
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localWorkGroupSize)

    if (rowId < qDim) {
        var rowOffset: Int = rowId * inputDim
        var partialSum: Float = 0.0f
        run {
            var j: Int = localId
            while (j < inputDim) {
                partialSum += wq.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
                j += localWorkGroupSize
            }
        }
        localSum[localId] = partialSum
        context.localBarrier()
        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }
        if (localId == 0) {
            q.set(rowId, localSum[0])
        }

    } else if (rowId < qDim + kvDim) {
        var kRow: Int = rowId - qDim
        var rowOffset: Int = kRow * inputDim
        var partialSum: Float = 0.0f
        run {
            var j: Int = localId
            while (j < inputDim) {
                partialSum += wk.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
                j += localWorkGroupSize
            }
        }
        localSum[localId] = partialSum
        context.localBarrier()
        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }
        if (localId == 0) {
            k.set(kRow, localSum[0])
        }

    } else if (rowId < qDim + 2 * kvDim) {
        var vRow: Int = rowId - qDim - kvDim
        var rowOffset: Int = vRow * inputDim
        var partialSum: Float = 0.0f
        run {
            var j: Int = localId
            while (j < inputDim) {
                partialSum += wv.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
                j += localWorkGroupSize
            }
        }
        localSum[localId] = partialSum
        context.localBarrier()
        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSum[localId] += localSum[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }
        if (localId == 0) {
            v.set(vRow, localSum[0])
        }
    }
}

// @formatter:off
fun matrixVectorGeneric(
        context: KernelContext,
        x: HalfFloatArray,
        hb: FloatArray, // output
        w: HalfFloatArray,
        dim1: Int, // inner loop
        dim0: Int, // outer loop
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var localSize: Int = localWorkGroupSize

    // Early exit if this workgroup is beyond our output dimension
    if (rowId >= dim0) {
        return
    }
    var sum: Float = matrixVectorRowMajorOptimizedSingle(context, localSize, x, w, dim1)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        hb.set(rowId, sum)
    }
}

// @formatter:on

/**
 * CUDA/Metal SIMD variant of {@link #matrixVectorGeneric(KernelContext, HalfFloatArray,
 * FloatArray, HalfFloatArray, int, int, int)} — same kernel shape as {@link
 * #matrixVectorGenericWithResidualSimd32}, minus the residual add: this call site (the
 * vocabulary projection) has no residual to accumulate into. Assumes the existing decode worker
 * shape: one 32-lane workgroup per output row.
 */
fun matrixVectorGenericSimd32(
        context: KernelContext,
        x: HalfFloatArray,
        hb: FloatArray,
        w: HalfFloatArray,
        n: Int,
        d: Int,
) {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= d) {
        return
    }

    var rowOffset: Int = rowId * n
    var partialSum: Float = 0.0f
    run {
        var j: Int = localId
        while (j < n) {
            partialSum += w.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
            j += 32
        }
    }

    partialSum += context.simdShuffleDown(partialSum, 16)
    partialSum += context.simdShuffleDown(partialSum, 8)
    partialSum += context.simdShuffleDown(partialSum, 4)
    partialSum += context.simdShuffleDown(partialSum, 2)
    partialSum += context.simdShuffleDown(partialSum, 1)

    if (localId == 0) {
        hb.set(rowId, partialSum)
    }
}

/**
 * Matrix-vector multiplication with residual connection. Combines regular matrix multiplication
 * with addition of existing values.
 *
 * <p>Formula: hb[i] = hb[i] + w[i]·x
 *
 * @param context Kernel execution context
 * @param x Input vector
 * @param hb Input/output vector (contains residual, receives result)
 * @param w Weight matrix
 * @param n Input dimension
 * @param d Output dimension
 * @param localWorkGroupSize Work group size
 */
fun matrixVectorGenericWithResidual(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        w: HalfFloatArray,
        n: Int,
        d: Int,
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var localSize: Int = localWorkGroupSize

    // Early exit if this workgroup is beyond our output dimension
    if (rowId >= d) {
        return
    }

    var sum: Float = matrixVectorRowMajorOptimized(context, localSize, x, w, n)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        var result: Float = hb.get(rowId) + sum
        hb.set(rowId, result)
    }
}

/**
 * CUDA/Metal SIMD variant of matrixVectorGenericWithResidual for LLaMA FP16 decode. It assumes
 * the existing decode worker shape: one 32-lane workgroup per output row.
 */
fun matrixVectorGenericWithResidualSimd32(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        w: HalfFloatArray,
        n: Int,
        d: Int,
) {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= d) {
        return
    }

    var rowOffset: Int = rowId * n
    var partialSum: Float = 0.0f
    run {
        var j: Int = localId
        while (j < n) {
            partialSum += w.get(rowOffset + j).getFloat32() * x.get(j)
            j += 32
        }
    }

    partialSum += context.simdShuffleDown(partialSum, 16)
    partialSum += context.simdShuffleDown(partialSum, 8)
    partialSum += context.simdShuffleDown(partialSum, 4)
    partialSum += context.simdShuffleDown(partialSum, 2)
    partialSum += context.simdShuffleDown(partialSum, 1)

    if (localId == 0) {
        hb.set(rowId, hb.get(rowId) + partialSum)
    }
}

fun matrixVectorGenericWithResidual(
        context: KernelContext,
        x: HalfFloatArray,
        hb: FloatArray,
        w: HalfFloatArray,
        n: Int,
        d: Int,
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var localSize: Int = localWorkGroupSize

    // Early exit if this workgroup is beyond our output dimension
    if (rowId >= d) {
        return
    }

    var sum: Float = matrixVectorRowMajorOptimized(context, localSize, x, w, n)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        var result: Float = hb.get(rowId) + sum
        hb.set(rowId, result)
    }
}

/**
 * Fused feed-forward network with SiLU activation and GLU gating. Implements the SwiGLU variant
 * used in LLaMA-style models.
 *
 * <p>Formula: FFN(x) = SiLU(x·W1) ⊙ (x·W3) where ⊙ denotes element-wise multiplication
 *
 * @param context Kernel execution context
 * @param x Input vector
 * @param hb Output buffer
 * @param w1 First feed-forward weight matrix
 * @param w3 Third feed-forward weight matrix (gate)
 * @param n Input dimension
 * @param d Hidden dimension
 * @param localWorkGroupSize Work group size
 */
fun fusedFeedForwardWithSiLUAndGLUActivation(
        context: KernelContext,
        x: HalfFloatArray,
        hb: HalfFloatArray,
        w1: HalfFloatArray,
        w3: HalfFloatArray,
        n: Int,
        d: Int,
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= d) {
        return
    }

    var sum1: Float = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w1, n)
    var sum3: Float = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w3, n)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        var silu: Float = siluActivation(sum1) // Using the new SiLU method
        var result: Float = silu * sum3
        hb.set(rowId, HalfFloat(result))
    }
}

fun fusedFeedForwardWithSiLUAndGLUActivation(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        w1: HalfFloatArray,
        w3: HalfFloatArray,
        n: Int,
        d: Int,
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= d) {
        return
    }

    var sum1: Float = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w1, n)
    var sum3: Float = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, w3, n)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        var silu: Float = siluActivation(sum1) // Using the new SiLU method
        var result: Float = silu * sum3
        hb.set(rowId, result)
    }
}

fun fusedFeedForwardWithSiLUAndGLUActivation(
        context: KernelContext,
        x: HalfFloatArray,
        hb: FloatArray,
        w1: HalfFloatArray,
        w3: HalfFloatArray,
        n: Int,
        d: Int,
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= d) {
        return
    }

    var sum1: HalfFloat = matrixVectorRowMajorOptimizedFHF(context, localWorkGroupSize, x, w1, n)
    var sum3: HalfFloat = matrixVectorRowMajorOptimizedFHF(context, localWorkGroupSize, x, w3, n)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        var silu: Float = siluActivation(sum1.getFloat32()) // Using the new SiLU method
        var result: Float = silu * sum3.getFloat32()
        hb.set(rowId, result)
    }
}

/**
 * Gaussian Error Linear Unit (GELU) activation function. Approximation formula: GELU(x) ≈ 0.5 *
 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 *
 * @param x Input value
 * @return Activated value
 */
fun geluActivation(x: Float): Float {
    var x3: Float = x * x * x
    return 0.5f * x * (1.0f + TornadoMath.tanh((0.797885f * (x + 0.044715f * x3))))
}

/**
 * Sigmoid-weighted Linear Unit (SiLU) activation function. Also known as Swish activation.
 *
 * <p>Formula: SiLU(x) = x * σ(x) = x / (1 + e^(-x))
 *
 * @param x Input value
 * @return Activated value
 */
fun siluActivation(x: Float): Float {
    return x * (1.0f / (1.0f + TornadoMath.exp(-x)))
}

/**
 * Optimized row-major matrix-vector multiplication for a single row. Uses parallel reduction
 * within a work group to compute one dot product.
 *
 * <p>Algorithm: 1. Each thread computes partial dot product 2. Partial results stored in local
 * memory 3. Tree-based reduction combines partial results 4. Returns final dot product for the
 * row
 *
 * @param context Kernel execution context
 * @param localSize Work group size
 * @param x Input vector
 * @param w Weight matrix row
 * @param n Input dimension
 * @return Dot product result for this row
 */
fun matrixVectorRowMajorOptimized(
        context: KernelContext,
        localSize: Int,
        x: FloatArray,
        w: FloatArray,
        n: Int,
): Float {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    // Allocate local memory for reduction
    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localSize)

    var rowOffset: Int = rowId * n

    // Each thread calculates partial dot product
    var partialSum: Float = 0.0f
    run {
        var j: Int = localId
        while (j < n) {
            var matrixIdx: Int = rowOffset + j
            partialSum += w.get(matrixIdx) * x.get(j)
            j += localSize
        }
    }

    // Store partial sum in local memory
    localSum[localId] = partialSum
    context.localBarrier()

    // Parallel reduction within workgroup
    run {
        var stride: Int = localSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    return localSum[0]
}

fun matrixVectorRowMajorOptimized(
        context: KernelContext,
        localSize: Int,
        x: FloatArray,
        w: HalfFloatArray,
        n: Int,
): Float {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    // Allocate local memory for reduction
    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localSize)

    var rowOffset: Int = rowId * n

    // Each thread accumulates over consecutive FP16 pairs read with a single packed
    // 32-bit load; consecutive threads read consecutive pairs, so a warp still issues
    // fully coalesced 128-byte transactions with half the memory instructions.
    // Rows start at rowId * n with n even for all supported models, so pair indices
    // stay even (4-byte aligned) as required by getHalf2.
    var partialSum: Float = 0.0f
    var nEven: Int = n and 1.inv()
    run {
        var j: Int = localId * 2
        while (j < nEven) {
            var matrixIdx: Int = rowOffset + j
            var pair: Half2 = w.getHalf2(matrixIdx)
            partialSum += Half2.lowFloat(pair) * x.get(j)
            partialSum += Half2.highFloat(pair) * x.get(j + 1)
            j += localSize * 2
        }
    }
    if (nEven != n && localId == 0) {
        partialSum += w.get(rowOffset + nEven).getFloat32() * x.get(nEven)
    }

    // Store partial sum in local memory
    localSum[localId] = partialSum
    context.localBarrier()

    // Parallel reduction within workgroup
    run {
        var stride: Int = localSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    return localSum[0]
}

fun matrixVectorRowMajorOptimizedFHF(
        context: KernelContext,
        localSize: Int,
        x: HalfFloatArray,
        w: HalfFloatArray,
        n: Int,
): HalfFloat {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    // Allocate local memory for reduction
    var localSum: Array<HalfFloat> = context.allocateHalfFloatLocalArray(localSize)

    var rowOffset: Int = rowId * n

    // Each thread calculates partial dot product
    var partialSum: Float = 0.0f
    //        HalfFloat partialSum = new HalfFloat(0f);
    run {
        var j: Int = localId
        while (j < n) {
            var matrixIdx: Int = rowOffset + j
            //            HalfFloat mul = HalfFloat.mult(w.get(matrixIdx), x.get(j));
            partialSum += w.get(matrixIdx).getFloat32() * x.get(j).getFloat32()
            //            partialSum = HalfFloat.add(partialSum, mul);
            j += localSize
        }
    }

    // Store partial sum in local memory
    localSum[localId] = HalfFloat(partialSum)
    context.localBarrier()

    // Parallel reduction within workgroup
    run {
        var stride: Int = localSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] = HalfFloat.add(localSum[localId], localSum[localId + stride])
                //                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    return localSum[0]
}

fun matrixVectorRowMajorOptimizedF(
        context: KernelContext,
        localSize: Int,
        x: HalfFloatArray,
        w: HalfFloatArray,
        n: Int,
): Float {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    // Allocate local memory for reduction
    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localSize)

    var rowOffset: Int = rowId * n

    // Each thread calculates partial dot product
    var partialSum: Float = 0.0f
    //        HalfFloat partialSum = new HalfFloat(0f);
    run {
        var j: Int = localId
        while (j < n) {
            var matrixIdx: Int = rowOffset + j
            //            HalfFloat mul = HalfFloat.mult(w.get(matrixIdx), x.get(j));
            partialSum += w.get(matrixIdx).getFloat32() * x.get(j).getFloat32()
            //            partialSum = HalfFloat.add(partialSum, mul);
            j += localSize
        }
    }

    // Store partial sum in local memory
    localSum[localId] = partialSum
    context.localBarrier()

    // Parallel reduction within workgroup
    run {
        var stride: Int = localSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    return localSum[0]
}

fun matrixVectorRowMajorOptimizedXX(
        context: KernelContext,
        localSize: Int,
        x: HalfFloatArray,
        w: HalfFloatArray,
        n: Int,
): Float {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localSize)

    var rowOffset: Int = rowId * n
    var sum0: Float = 0.0f
    var sum1: Float = 0.0f
    var sum2: Float = 0.0f
    var sum3: Float = 0.0f

    var stride: Int = localSize
    var stride2: Int = localSize shl 1
    var stride3: Int = localSize * 3
    var stride4: Int = localSize shl 2

    // Already coalesced: thread 0 reads idx 0, thread 1 reads idx 1, etc.
    var j: Int = localId
    var limit: Int = n - stride3

    while (j < limit) {
        var base: Int = rowOffset + j
        // Hoist x.get() calls - they're reused across all rows
        var x0: Float = x.get(j).getFloat32()
        var x1: Float = x.get(j + stride).getFloat32()
        var x2: Float = x.get(j + stride2).getFloat32()
        var x3: Float = x.get(j + stride3).getFloat32()

        sum0 += w.get(base).getFloat32() * x0
        sum1 += w.get(base + stride).getFloat32() * x1
        sum2 += w.get(base + stride2).getFloat32() * x2
        sum3 += w.get(base + stride3).getFloat32() * x3
        j += stride4
    }

    while (j < n) {
        sum0 += w.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
        j += stride
    }

    localSum[localId] = (sum0 + sum1) + (sum2 + sum3)
    context.localBarrier()

    // Reduction with minimal barriers
    run {
        var s: Int = localSize shr 1
        while (s > 0) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s]
            }
            context.localBarrier()
            s = s shr (1)
        }
    }

    return localSum[0]
}

fun matrixVectorRowMajorOptimizedSingle(
        context: KernelContext,
        localSize: Int,
        x: HalfFloatArray,
        w: HalfFloatArray,
        n: Int,
): Float {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    // Allocate local memory for reduction
    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localSize)

    var rowOffset: Int = rowId * n

    // Both operands are FP16: read them as packed pairs (single 32-bit loads), multiply
    // with the packed __hmul2 intrinsic and accumulate in FP32. n is even.
    var partialSum: Float = 0.0f
    run {
        var j: Int = localId * 2
        while (j < n) {
            var matrixIdx: Int = rowOffset + j
            var product: Half2 = Half2.mult(w.getHalf2(matrixIdx), x.getHalf2(j))
            partialSum += Half2.lowFloat(product)
            partialSum += Half2.highFloat(product)
            j += localSize * 2
        }
    }

    // Store partial sum in local memory
    localSum[localId] = partialSum
    context.localBarrier()

    // Parallel reduction within workgroup
    run {
        var stride: Int = localSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    return localSum[0]
}

fun matrixVectorRowMajorOptimized(
        context: KernelContext,
        localSize: Int,
        x: HalfFloatArray,
        w: HalfFloatArray,
        n: Int,
): Float {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localSize)

    var rowOffset: Int = rowId * n

    // Accumulate in HalfFloat to avoid conversions in inner loop
    var sum0: HalfFloat = HalfFloat(0f)
    var sum1: HalfFloat = HalfFloat(0f)
    var sum2: HalfFloat = HalfFloat(0f)
    var sum3: HalfFloat = HalfFloat(0f)

    var stride: Int = localSize
    var stride2: Int = localSize shl 1
    var stride3: Int = localSize * 3
    var stride4: Int = localSize shl 2

    var j: Int = localId
    var limit: Int = n - stride3

    while (j < limit) {
        var base: Int = rowOffset + j

        // Stay in HalfFloat - no getFloat32() calls
        var x0: HalfFloat = x.get(j)
        var x1: HalfFloat = x.get(j + stride)
        var x2: HalfFloat = x.get(j + stride2)
        var x3: HalfFloat = x.get(j + stride3)

        sum0 = HalfFloat.add(sum0, HalfFloat.mult(w.get(base), x0))
        sum1 = HalfFloat.add(sum1, HalfFloat.mult(w.get(base + stride), x1))
        sum2 = HalfFloat.add(sum2, HalfFloat.mult(w.get(base + stride2), x2))
        sum3 = HalfFloat.add(sum3, HalfFloat.mult(w.get(base + stride3), x3))
        j += stride4
    }

    // Cleanup loop
    while (j < n) {
        sum0 = HalfFloat.add(sum0, HalfFloat.mult(w.get(rowOffset + j), x.get(j)))
        j += stride
    }

    // Convert to float32 only at the end for reduction
    localSum[localId] = sum0.getFloat32() + sum1.getFloat32() + sum2.getFloat32() + sum3.getFloat32()
    context.localBarrier()

    run {
        var s: Int = localSize shr 1
        while (s > 0) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s]
            }
            context.localBarrier()
            s = s shr (1)
        }
    }

    return localSum[0]
}

fun fusedQKVMatmul(
        context: KernelContext,
        x: HalfFloatArray, // input (read once!)
        q: FloatArray,
        k: FloatArray,
        v: FloatArray, // outputs
        wq: HalfFloatArray,
        wk: HalfFloatArray,
        wv: HalfFloatArray,
        dim: Int,
        kvDim: Int,
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    // Determine which output this workgroup computes
    var totalRows: Int = dim + 2 * kvDim // Q rows + K rows + V rows

    if (rowId < dim) {
        // Q projection
        var sum: Float = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, wq, dim)
        if (localId == 0) {
            q.set(rowId, sum)
        }
    } else if (rowId < dim + kvDim) {
        // K projection
        var kRow: Int = rowId - dim
        var sum: Float = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, wk, dim)
        if (localId == 0) {
            k.set(kRow, sum)
        }
    } else {
        // V projection
        var vRow: Int = rowId - dim - kvDim
        var sum: Float = matrixVectorRowMajorOptimized(context, localWorkGroupSize, x, wv, dim)
        if (localId == 0) {
            v.set(vRow, sum)
        }
    }
}

fun matrixVectorRowMajorOptimizedx(
        context: KernelContext,
        localSize: Int,
        x: HalfFloatArray,
        w: HalfFloatArray,
        n: Int,
): Float {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    // Allocate local memory for reduction
    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localSize)

    var rowOffset: Int = rowId * n

    // Each thread calculates partial dot product - UNROLLED BY 4
    var sum0: Float = 0.0f
    var sum1: Float = 0.0f
    var sum2: Float = 0.0f
    var sum3: Float = 0.0f

    var j: Int = localId
    var stride: Int = localSize
    var stride4: Int = localSize shl 2 // localSize * 4
    var limit: Int = n - (stride * 3) // Safe limit for 4 elements

    // Main loop unrolled by 4 with separate accumulators
    while (j < limit) {
        var base: Int = rowOffset + j
        var j1: Int = j + stride
        var j2: Int = j + (stride shl 1)
        var j3: Int = j + stride * 3

        sum0 += w.get(base).getFloat32() * x.get(j).getFloat32()
        sum1 += w.get(base + stride).getFloat32() * x.get(j1).getFloat32()
        sum2 += w.get(base + (stride shl 1)).getFloat32() * x.get(j2).getFloat32()
        sum3 += w.get(base + stride * 3).getFloat32() * x.get(j3).getFloat32()
        j += stride4
    }

    // Handle remainder
    while (j < n) {
        sum0 += w.get(rowOffset + j).getFloat32() * x.get(j).getFloat32()
        j += stride
    }

    // Combine accumulators (tree reduction for better precision)
    var partialSum: Float = (sum0 + sum1) + (sum2 + sum3)

    // Store partial sum in local memory
    localSum[localId] = partialSum
    context.localBarrier()

    // Parallel reduction within workgroup
    run {
        var s: Int = localSize shr 1
        while (s > 0) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s]
            }
            context.localBarrier()
            s = s shr (1)
        }
    }

    return localSum[0]
}

/**
 * Race-free RMS-norm reduction in ONE workgroup: threads stride over the input, reduce in local
 * memory, and lane 0 writes the final scale factor to {@code output[0]}.
 *
 * <p>The multi-workgroup variant ({@code reductionOneBlockWithLayer}) has workgroup 0 combining
 * the other workgroups' partial sums with <em>no inter-workgroup synchronization</em> — a data
 * race. On the CUDA backend the race outcome is schedule/compilation dependent: e.g. Qwen3-1.7B
 * reads stale partials on every layer (wrong normalization scale → garbage output) while
 * Llama-1B and Qwen3-4B happen to win the race. The NON_NVIDIA scheduler avoids the race with a
 * separate {@code reductionFinalNormalization} task; this kernel is the NVIDIA-path equivalent.
 * Launch with exactly one workgroup: {@code global == local == localMemSize}.
 */
fun reductionOneBlockWithLayerSingleGroup(
        context: KernelContext,
        output: FloatArray,
        x: FloatArray,
        size: Int,
        ermsNorm: Float,
        localMemSize: Int,
) {
    var lid: Int = context.localIdx
    var groupSize: Int = context.localGroupSizeX
    var localX: kotlin.FloatArray = context.allocateFloatLocalArray(localMemSize)

    var partial: Float = 0.0f
    run {
        var j: Int = lid
        while (j < size) {
            var v: Float = x.get(j)
            partial += v * v
            j += groupSize
        }
    }
    localX[lid] = partial

    run {
        var stride: Int = groupSize / 2
        while (stride > 0) {
            context.localBarrier()
            if (lid < stride) {
                localX[lid] += localX[lid + stride]
            }
            stride /= 2
        }
    }

    if (lid == 0) {
        var ss: Float = localX[0] / size + ermsNorm
        output.set(0, 1.0f / TornadoMath.sqrt(ss))
    }
}

// Second kernel - Combines partial sums and computes final normalization
fun reductionFinalNormalization(
        context: KernelContext,
        output: FloatArray,
        size: Int,
        ermsNorm: Float,
) {
    var gid: Int = context.globalIdx

    // Only one thread needs to perform this calculation
    if (gid == 0) {
        // Combine partial sums from all workgroups
        var ss: Float = 0.0f
        run {
            var i: Int = 1
            while (i < output.getSize()) {  // Fixed bounds to avoid out of bounds
                ss += output.get(i)
                i++
            }
        }

        ss /= size
        ss += ermsNorm
        ss = 1.0f / TornadoMath.sqrt(ss)
        output.set(0, ss); // Store the final scale factor
    }
}

fun splitGateUpAndSiLU(
        hb: FloatArray,
        hbG: FloatArray,
        hbU: FloatArray,
        hiddenDim: Int,
) {
    // Copy and apply SiLU to gate in one pass
    parallelFor(0, hiddenDim) { i ->
        var gateVal: Float = hb.get(i)
        var upVal: Float = hb.get(hiddenDim + i)

        // Apply SiLU to gate
        var siluGate: Float = gateVal / (1.0f + TornadoMath.exp(-gateVal))

        // Store activated gate and multiply with up
        hbG.set(i, siluGate)
        hbU.set(i, siluGate * upVal)
    }
}

fun addInPlace(arrayA: FloatArray, arrayB: FloatArray, size: Int) {
    // Element-wise addition: arrayA[i] = arrayA[i] + arrayB[i]
    parallelFor(0, size) { i ->
        var result: Float = arrayA.get(i) + arrayB.get(i)
        arrayA.set(i, result)
    }
}

/**
 * Matrix-vector multiplication for Q8_0 quantized weights.
 *
 * @param context Kernel context
 * @param x Input activations (FloatArray)
 * @param output Output array (FloatArray)
 * @param weightsQ Quantized weights (Int8Array) - from Q8_0QuantizedTensor.getQuants()
 * @param weightScales Scale factors (HalfFloatArray) - from Q8_0QuantizedTensor.getScales()
 * @param dim1 Input dimension (n - number of columns)
 * @param dim0 Output dimension (d - number of rows)
 * @param localWorkGroupSize Local workgroup size
 */
fun matrixVectorGeneric(
        context: KernelContext,
        x: FloatArray,
        output: FloatArray,
        weightsQ: Int8Array,
        weightScales: HalfFloatArray,
        dim1: Int,
        dim0: Int,
        localWorkGroupSize: Int,
) {

    // One row per workgroup
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    // Early exit if this workgroup is beyond output dimension
    if (rowId >= dim0) {
        return
    }

    var sum: Float = matrixVectorRowMajorOptimizedQ8_0( context, localWorkGroupSize, x, weightsQ, weightScales, dim1)

    // Thread 0 writes the result
    if (localId == 0) {
        output.set(rowId, sum)
    }
}

/**
 * Helper method to compute dot product for a single row with Q8_0 quantized weights. Uses 4-way
 * unrolling for better performance.
 */
fun matrixVectorRowMajorOptimizedQ8_0(
        context: KernelContext,
        localSize: Int,
        x: FloatArray,
        weightsQ: Int8Array,
        weightScales: HalfFloatArray,
        n: Int,
): Float {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var blockSize: Int = 32

    // Allocate local memory for reduction
    var localSums: kotlin.FloatArray = context.allocateFloatLocalArray(localSize)

    var rowOffset: Int = rowId * n
    var scalesRowOffset: Int = rowId * (n / blockSize)

    // 4-way unrolling
    var partialSum1: Float = 0.0f
    var partialSum2: Float = 0.0f
    var partialSum3: Float = 0.0f
    var partialSum4: Float = 0.0f

    // Main loop - process 4 elements at a time
    run {
        var j: Int = localId * 4
        while (j < n - 3) {
            var blockIdx: Int = j / blockSize
            var scale: Float = weightScales.get(scalesRowOffset + blockIdx).getFloat32()

            // Dequantize and multiply
            partialSum1 += ((weightsQ.get(rowOffset + j)).toFloat() * scale) * x.get(j)
            partialSum2 += ((weightsQ.get(rowOffset + j + 1)).toFloat() * scale) * x.get(j + 1)
            partialSum3 += ((weightsQ.get(rowOffset + j + 2)).toFloat() * scale) * x.get(j + 2)
            partialSum4 += ((weightsQ.get(rowOffset + j + 3)).toFloat() * scale) * x.get(j + 3)
            j += localSize * 4
        }
    }

    var partialSum: Float = partialSum1 + partialSum2 + partialSum3 + partialSum4

    // Handle remaining elements
    run {
        var j: Int = ((n / 4) * 4) + localId
        while (j < n) {
            var blockIdx: Int = j / blockSize
            var scale: Float = weightScales.get(scalesRowOffset + blockIdx).getFloat32()
            partialSum += ((weightsQ.get(rowOffset + j)).toFloat() * scale) * x.get(j)
            j += localSize
        }
    }

    // Store partial sum
    localSums[localId] = partialSum
    context.localBarrier()

    // Parallel reduction
    run {
        var stride: Int = localSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    return localSums[0]
}

fun fusedRmsNormQKVMatmulQ8(
        context: KernelContext,
        x: FloatArray,
        q: FloatArray,
        k: FloatArray,
        v: FloatArray,
        rmsWeights: FloatArray,
        rmsScale: FloatArray,
        wqkv: ByteArray,
        dim: Int,
        kvDim: Int,
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    var totalRows: Int = dim + 2 * kvDim
    if (rowId >= totalRows) {
        return
    }

    var rmsScaleFactor: Float = rmsScale.get(0)

    var blockSize: Int = 32
    var Q8_0_BLOCK_BYTES: Int = 34

    var localSums: kotlin.FloatArray = context.allocateFloatLocalArray(localWorkGroupSize)

    var blocksPerRow: Int = (dim + blockSize - 1) / blockSize
    var rowBlockOffset: Int = rowId * blocksPerRow

    var partialSum1: Float = 0.0f
    var partialSum2: Float = 0.0f
    var partialSum3: Float = 0.0f
    var partialSum4: Float = 0.0f

    // Main loop - 4-way unrolled, only when we have complete groups of 4
    var mainLoopEnd: Int = (dim / 4) * 4 // Largest multiple of 4 <= dim
    run {
        var j: Int = localId * 4
        while (j < mainLoopEnd) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize

            var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

            var qScale: HalfFloat = wqkv.getHalfFloat(blockByteOffset)
            var qScaleFloat: Float = qScale.getFloat32()

            var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
            var quant1: Byte = wqkv.get(quantsOffset)
            var quant2: Byte = wqkv.get(quantsOffset + 1)
            var quant3: Byte = wqkv.get(quantsOffset + 2)
            var quant4: Byte = wqkv.get(quantsOffset + 3)

            var norm1: Float = rmsWeights.get(j) * rmsScaleFactor * x.get(j)
            var norm2: Float = rmsWeights.get(j + 1) * rmsScaleFactor * x.get(j + 1)
            var norm3: Float = rmsWeights.get(j + 2) * rmsScaleFactor * x.get(j + 2)
            var norm4: Float = rmsWeights.get(j + 3) * rmsScaleFactor * x.get(j + 3)

            partialSum1 += (quant1.toFloat() * qScaleFloat) * norm1
            partialSum2 += (quant2.toFloat() * qScaleFloat) * norm2
            partialSum3 += (quant3.toFloat() * qScaleFloat) * norm3
            partialSum4 += (quant4.toFloat() * qScaleFloat) * norm4
            j += localWorkGroupSize * 4
        }
    }

    // Tail loop - handle remaining 0-3 elements (one element per thread)
    var tailIdx: Int = mainLoopEnd + localId
    if (tailIdx < dim) {
        var blockIdx: Int = tailIdx / blockSize
        var withinBlockIdx: Int = tailIdx % blockSize
        var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

        var qScale: HalfFloat = wqkv.getHalfFloat(blockByteOffset)
        var qScaleFloat: Float = qScale.getFloat32()

        var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
        var quant: Byte = wqkv.get(quantsOffset)

        var normalized: Float = rmsWeights.get(tailIdx) * rmsScaleFactor * x.get(tailIdx)
        partialSum1 += (quant.toFloat() * qScaleFloat) * normalized
    }

    localSums[localId] = partialSum1 + partialSum2 + partialSum3 + partialSum4
    context.localBarrier()

    // Parallel reduction
    run {
        var stride: Int = localWorkGroupSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    // Thread 0 writes to appropriate output
    if (localId == 0) {
        var result: Float = localSums[0]

        if (rowId < dim) {
            q.set(rowId, result)
        } else if (rowId < dim + kvDim) {
            k.set(rowId - dim, result)
        } else {
            v.set(rowId - dim - kvDim, result)
        }
    }
}

fun matrixVectorGenericQ8Byte(
        context: KernelContext,
        x: FloatArray,
        output: FloatArray,
        q: ByteArray,
        dim1: Int,
        dim0: Int,
        localWorkGroupSize: Int,
) {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= dim0) {
        return
    }

    var sum: Float = matrixVectorRowMajorOptimizedQ8_0Byte(context, localWorkGroupSize, x, q, dim1)

    // Thread 0 writes the result
    if (localId == 0) {
        output.set(rowId, sum)
    }
}

fun matrixVectorRowMajorOptimizedQ8_0Byte(
        context: KernelContext,
        localSize: Int,
        x: FloatArray,
        q: ByteArray,
        n: Int,
): Float {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var blockSize: Int = 32
    var Q8_0_BLOCK_BYTES: Int = 34 // 2 bytes scale + 32 bytes quants

    // Allocate local memory for reduction
    var localSums: kotlin.FloatArray = context.allocateFloatLocalArray(localSize)

    var blocksPerRow: Int = (n + blockSize - 1) / blockSize
    var rowBlockOffset: Int = rowId * blocksPerRow // Starting block index for this row

    // 4-way unrolling
    var partialSum1: Float = 0.0f
    var partialSum2: Float = 0.0f
    var partialSum3: Float = 0.0f
    var partialSum4: Float = 0.0f

    // Main loop - process 4 elements at a time
    run {
        var j: Int = localId * 4
        while (j < n - 3) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize

            // Calculate byte offset for this Q8_0 block
            var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

            // Load scale (first 2 bytes of block as HalfFloat)
            var scale: HalfFloat = q.getHalfFloat(blockByteOffset)
            var scaleFloat: Float = scale.getFloat32()

            // Load 4 consecutive quantized values
            var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx // Skip 2-byte scale
            var quant1: Byte = q.get(quantsOffset)
            var quant2: Byte = q.get(quantsOffset + 1)
            var quant3: Byte = q.get(quantsOffset + 2)
            var quant4: Byte = q.get(quantsOffset + 3)

            // Dequantize and multiply
            partialSum1 += (quant1.toFloat() * scaleFloat) * x.get(j)
            partialSum2 += (quant2.toFloat() * scaleFloat) * x.get(j + 1)
            partialSum3 += (quant3.toFloat() * scaleFloat) * x.get(j + 2)
            partialSum4 += (quant4.toFloat() * scaleFloat) * x.get(j + 3)
            j += localSize * 4
        }
    }

    var partialSum: Float = partialSum1 + partialSum2 + partialSum3 + partialSum4

    // Handle remaining elements
    run {
        var j: Int = ((n / 4) * 4) + localId
        while (j < n) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize

            // Calculate byte offset for this Q8_0 block
            var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

            // Load scale
            var scale: HalfFloat = q.getHalfFloat(blockByteOffset)
            var scaleFloat: Float = scale.getFloat32()

            // Load quantized value
            var quant: Byte = q.get(blockByteOffset + 2 + withinBlockIdx)

            partialSum += (quant.toFloat() * scaleFloat) * x.get(j)
            j += localSize
        }
    }

    localSums[localId] = partialSum
    context.localBarrier()

    // Parallel reduction
    run {
        var stride: Int = localSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    return localSums[0]
}

fun matrixVectorGenericWithResidual(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        w_quants: Int8Array,
        w_scales: HalfFloatArray,
        n: Int,
        d: Int,
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var localSize: Int = localWorkGroupSize

    // Early exit if this workgroup is beyond our output dimension
    if (rowId >= d) {
        return
    }

    var sum: Float = matrixVectorRowMajorOptimizedQ8_0(context, localSize, x, w_quants, w_scales, n)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        var result: Float = hb.get(rowId) + sum
        hb.set(rowId, result)
    }
}

fun matrixVectorGenericWithResidualQ8_0Byte(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        w: ByteArray,
        n: Int,
        d: Int,
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx
    var localSize: Int = localWorkGroupSize

    // Early exit if this workgroup is beyond our output dimension
    if (rowId >= d) {
        return
    }

    var sum: Float = matrixVectorRowMajorOptimizedQ8_0Byte(context, localSize, x, w, n)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        var result: Float = hb.get(rowId) + sum
        hb.set(rowId, result)
    }
}

/**
 * Warp-shuffle variant of {@link #matrixVectorGenericWithResidualQ8_0Byte}: one 32-lane warp
 * per output row, reduced via {@code simdShuffleDown} instead of a shared-memory tree. Q8_0
 * byte layout (34-byte blocks: 2-byte half scale + 32 int8 quants).
 */
fun matrixVectorGenericWithResidualQ8_0ByteSimd32(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        w: ByteArray,
        n: Int,
        d: Int,
) {
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= d) {
        return
    }

    var blockSize: Int = 32
    var Q8_0_BLOCK_BYTES: Int = 34 // 2-byte scale + 32 int8 quants
    var blocksPerRow: Int = (n + blockSize - 1) / blockSize
    var rowBlockOffset: Int = rowId * blocksPerRow

    var partialSum: Float = 0.0f
    run {
        var j: Int = localId
        while (j < n) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j - blockIdx * blockSize
            var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
            var scaleFloat: Float = w.getHalfFloat(blockByteOffset).getFloat32()
            var quant: Byte = w.get(blockByteOffset + 2 + withinBlockIdx)
            partialSum += (quant.toFloat() * scaleFloat) * x.get(j)
            j += 32
        }
    }

    partialSum += context.simdShuffleDown(partialSum, 16)
    partialSum += context.simdShuffleDown(partialSum, 8)
    partialSum += context.simdShuffleDown(partialSum, 4)
    partialSum += context.simdShuffleDown(partialSum, 2)
    partialSum += context.simdShuffleDown(partialSum, 1)

    if (localId == 0) {
        hb.set(rowId, hb.get(rowId) + partialSum)
    }
}

fun fusedFeedForwardWithSiLUAndGLUActivation(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        w1_quants: Int8Array,
        w1_scales: HalfFloatArray,
        w3_quants: Int8Array,
        w3_scales: HalfFloatArray,
        n: Int,
        d: Int,
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= d) {
        return
    }

    var sum1: Float = matrixVectorRowMajorOptimizedQ8_0( context, localWorkGroupSize, x, w1_quants, w1_scales, n)
    var sum3: Float = matrixVectorRowMajorOptimizedQ8_0( context, localWorkGroupSize, x, w3_quants, w3_scales, n)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        var silu: Float = siluActivation(sum1) // Using the new SiLU method
        var result: Float = silu * sum3
        hb.set(rowId, result)
    }
}

fun fusedFeedForwardWithSiLUAndGLUActivationQ8_0Byte(
        context: KernelContext,
        x: FloatArray,
        hb: FloatArray,
        w1: ByteArray,
        w3: ByteArray,
        n: Int,
        d: Int,
        localWorkGroupSize: Int,
) {
    // One row per workgroup (not per thread)
    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= d) {
        return
    }

    var sum1: Float = matrixVectorRowMajorOptimizedQ8_0Byte(context, localWorkGroupSize, x, w1, n)
    var sum3: Float = matrixVectorRowMajorOptimizedQ8_0Byte(context, localWorkGroupSize, x, w3, n)

    // Thread 0 in each workgroup writes the final result
    if (localId == 0) {
        var silu: Float = siluActivation(sum1) // Using the new SiLU method
        var result: Float = silu * sum3
        hb.set(rowId, result)
    }
}

/**
 * Fused Q/K/V matrix-vector multiplication for Q8_0 quantized weights. Reduces kernel launch
 * overhead and improves input vector cache utilization.
 *
 * <p>Workgroup assignment: - rowId [0, dim): Q projection - rowId [dim, dim+kvDim): K
 * projection - rowId [dim+kvDim, dim+2*kvDim): V projection
 */
fun fusedQKVMatmulQ8(
        context: KernelContext,
        x: FloatArray,
        q: FloatArray,
        k: FloatArray,
        v: FloatArray,
        wq: ByteArray,
        wk: ByteArray,
        wv: ByteArray,
        dim: Int,
        kvDim: Int,
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    var blockSize: Int = 32
    var Q8_0_BLOCK_BYTES: Int = 34
    var blocksPerRow: Int = (dim + blockSize - 1) / blockSize

    var localSums: kotlin.FloatArray = context.allocateFloatLocalArray(localWorkGroupSize)

    if (rowId < dim) {
        // ========== Q projection ==========
        var rowBlockOffset: Int = rowId * blocksPerRow

        var partialSum1: Float = 0.0f
        var partialSum2: Float = 0.0f
        var partialSum3: Float = 0.0f
        var partialSum4: Float = 0.0f

        run {
            var j: Int = localId * 4
            while (j < dim - 3) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

                var scale: HalfFloat = wq.getHalfFloat(blockByteOffset)
                var scaleFloat: Float = scale.getFloat32()

                var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
                var quant1: Byte = wq.get(quantsOffset)
                var quant2: Byte = wq.get(quantsOffset + 1)
                var quant3: Byte = wq.get(quantsOffset + 2)
                var quant4: Byte = wq.get(quantsOffset + 3)

                partialSum1 += (quant1.toFloat() * scaleFloat) * x.get(j)
                partialSum2 += (quant2.toFloat() * scaleFloat) * x.get(j + 1)
                partialSum3 += (quant3.toFloat() * scaleFloat) * x.get(j + 2)
                partialSum4 += (quant4.toFloat() * scaleFloat) * x.get(j + 3)
                j += localWorkGroupSize * 4
            }
        }

        var partialSum: Float = partialSum1 + partialSum2 + partialSum3 + partialSum4

        run {
            var j: Int = ((dim / 4) * 4) + localId
            while (j < dim) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

                var scale: HalfFloat = wq.getHalfFloat(blockByteOffset)
                var scaleFloat: Float = scale.getFloat32()

                var quant: Byte = wq.get(blockByteOffset + 2 + withinBlockIdx)
                partialSum += (quant.toFloat() * scaleFloat) * x.get(j)
                j += localWorkGroupSize
            }
        }

        localSums[localId] = partialSum
        context.localBarrier()

        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }

        if (localId == 0) {
            q.set(rowId, localSums[0])
        }

    } else if (rowId < dim + kvDim) {
        // ========== K projection ==========
        var kRow: Int = rowId - dim
        var rowBlockOffset: Int = kRow * blocksPerRow

        var partialSum1: Float = 0.0f
        var partialSum2: Float = 0.0f
        var partialSum3: Float = 0.0f
        var partialSum4: Float = 0.0f

        run {
            var j: Int = localId * 4
            while (j < dim - 3) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

                var scale: HalfFloat = wk.getHalfFloat(blockByteOffset)
                var scaleFloat: Float = scale.getFloat32()

                var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
                var quant1: Byte = wk.get(quantsOffset)
                var quant2: Byte = wk.get(quantsOffset + 1)
                var quant3: Byte = wk.get(quantsOffset + 2)
                var quant4: Byte = wk.get(quantsOffset + 3)

                partialSum1 += (quant1.toFloat() * scaleFloat) * x.get(j)
                partialSum2 += (quant2.toFloat() * scaleFloat) * x.get(j + 1)
                partialSum3 += (quant3.toFloat() * scaleFloat) * x.get(j + 2)
                partialSum4 += (quant4.toFloat() * scaleFloat) * x.get(j + 3)
                j += localWorkGroupSize * 4
            }
        }

        var partialSum: Float = partialSum1 + partialSum2 + partialSum3 + partialSum4

        run {
            var j: Int = ((dim / 4) * 4) + localId
            while (j < dim) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

                var scale: HalfFloat = wk.getHalfFloat(blockByteOffset)
                var scaleFloat: Float = scale.getFloat32()

                var quant: Byte = wk.get(blockByteOffset + 2 + withinBlockIdx)
                partialSum += (quant.toFloat() * scaleFloat) * x.get(j)
                j += localWorkGroupSize
            }
        }

        localSums[localId] = partialSum
        context.localBarrier()

        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }

        if (localId == 0) {
            k.set(kRow, localSums[0])
        }

    } else if (rowId < dim + 2 * kvDim) {
        // ========== V projection ==========
        var vRow: Int = rowId - dim - kvDim
        var rowBlockOffset: Int = vRow * blocksPerRow

        var partialSum1: Float = 0.0f
        var partialSum2: Float = 0.0f
        var partialSum3: Float = 0.0f
        var partialSum4: Float = 0.0f

        run {
            var j: Int = localId * 4
            while (j < dim - 3) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

                var scale: HalfFloat = wv.getHalfFloat(blockByteOffset)
                var scaleFloat: Float = scale.getFloat32()

                var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
                var quant1: Byte = wv.get(quantsOffset)
                var quant2: Byte = wv.get(quantsOffset + 1)
                var quant3: Byte = wv.get(quantsOffset + 2)
                var quant4: Byte = wv.get(quantsOffset + 3)

                partialSum1 += (quant1.toFloat() * scaleFloat) * x.get(j)
                partialSum2 += (quant2.toFloat() * scaleFloat) * x.get(j + 1)
                partialSum3 += (quant3.toFloat() * scaleFloat) * x.get(j + 2)
                partialSum4 += (quant4.toFloat() * scaleFloat) * x.get(j + 3)
                j += localWorkGroupSize * 4
            }
        }

        var partialSum: Float = partialSum1 + partialSum2 + partialSum3 + partialSum4

        run {
            var j: Int = ((dim / 4) * 4) + localId
            while (j < dim) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

                var scale: HalfFloat = wv.getHalfFloat(blockByteOffset)
                var scaleFloat: Float = scale.getFloat32()

                var quant: Byte = wv.get(blockByteOffset + 2 + withinBlockIdx)
                partialSum += (quant.toFloat() * scaleFloat) * x.get(j)
                j += localWorkGroupSize
            }
        }

        localSums[localId] = partialSum
        context.localBarrier()

        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }

        if (localId == 0) {
            v.set(vRow, localSums[0])
        }
    }
}

/**
 * Fused QKV matmul for models where Q output dim != input dim (e.g., Devstral 2). Separates
 * inputDim (embedding size) from qDim (num_heads * head_dim).
 */
fun fusedQKVMatmulQ8NonSquare(
        context: KernelContext,
        x: FloatArray,
        q: FloatArray,
        k: FloatArray,
        v: FloatArray,
        wq: ByteArray,
        wk: ByteArray,
        wv: ByteArray,
        inputDim: Int,
        qDim: Int,
        kvDim: Int,
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    var blockSize: Int = 32
    var Q8_0_BLOCK_BYTES: Int = 34
    var blocksPerRow: Int = (inputDim + blockSize - 1) / blockSize

    var localSums: kotlin.FloatArray = context.allocateFloatLocalArray(localWorkGroupSize)

    if (rowId < qDim) {
        // ========== Q projection ==========
        var rowBlockOffset: Int = rowId * blocksPerRow

        var partialSum1: Float = 0.0f
        var partialSum2: Float = 0.0f
        var partialSum3: Float = 0.0f
        var partialSum4: Float = 0.0f

        run {
            var j: Int = localId * 4
            while (j < inputDim - 3) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

                var scaleFloat: Float = wq.getHalfFloat(blockByteOffset).getFloat32()

                var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
                partialSum1 += ((wq.get(quantsOffset)).toFloat()) * scaleFloat * x.get(j)
                partialSum2 += ((wq.get(quantsOffset + 1)).toFloat()) * scaleFloat * x.get(j + 1)
                partialSum3 += ((wq.get(quantsOffset + 2)).toFloat()) * scaleFloat * x.get(j + 2)
                partialSum4 += ((wq.get(quantsOffset + 3)).toFloat()) * scaleFloat * x.get(j + 3)
                j += localWorkGroupSize * 4
            }
        }

        var partialSum: Float = partialSum1 + partialSum2 + partialSum3 + partialSum4

        run {
            var j: Int = ((inputDim / 4) * 4) + localId
            while (j < inputDim) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
                var scaleFloat: Float = wq.getHalfFloat(blockByteOffset).getFloat32()
                partialSum += ((wq.get(blockByteOffset + 2 + withinBlockIdx)).toFloat()) * scaleFloat * x.get(j)
                j += localWorkGroupSize
            }
        }

        localSums[localId] = partialSum
        context.localBarrier()

        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }

        if (localId == 0) {
            q.set(rowId, localSums[0])
        }

    } else if (rowId < qDim + kvDim) {
        // ========== K projection ==========
        var kRow: Int = rowId - qDim
        var rowBlockOffset: Int = kRow * blocksPerRow

        var partialSum1: Float = 0.0f
        var partialSum2: Float = 0.0f
        var partialSum3: Float = 0.0f
        var partialSum4: Float = 0.0f

        run {
            var j: Int = localId * 4
            while (j < inputDim - 3) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
                var scaleFloat: Float = wk.getHalfFloat(blockByteOffset).getFloat32()
                var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
                partialSum1 += ((wk.get(quantsOffset)).toFloat()) * scaleFloat * x.get(j)
                partialSum2 += ((wk.get(quantsOffset + 1)).toFloat()) * scaleFloat * x.get(j + 1)
                partialSum3 += ((wk.get(quantsOffset + 2)).toFloat()) * scaleFloat * x.get(j + 2)
                partialSum4 += ((wk.get(quantsOffset + 3)).toFloat()) * scaleFloat * x.get(j + 3)
                j += localWorkGroupSize * 4
            }
        }

        var partialSum: Float = partialSum1 + partialSum2 + partialSum3 + partialSum4

        run {
            var j: Int = ((inputDim / 4) * 4) + localId
            while (j < inputDim) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
                var scaleFloat: Float = wk.getHalfFloat(blockByteOffset).getFloat32()
                partialSum += ((wk.get(blockByteOffset + 2 + withinBlockIdx)).toFloat()) * scaleFloat * x.get(j)
                j += localWorkGroupSize
            }
        }

        localSums[localId] = partialSum
        context.localBarrier()

        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }

        if (localId == 0) {
            k.set(kRow, localSums[0])
        }

    } else if (rowId < qDim + 2 * kvDim) {
        // ========== V projection ==========
        var vRow: Int = rowId - qDim - kvDim
        var rowBlockOffset: Int = vRow * blocksPerRow

        var partialSum1: Float = 0.0f
        var partialSum2: Float = 0.0f
        var partialSum3: Float = 0.0f
        var partialSum4: Float = 0.0f

        run {
            var j: Int = localId * 4
            while (j < inputDim - 3) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
                var scaleFloat: Float = wv.getHalfFloat(blockByteOffset).getFloat32()
                var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
                partialSum1 += ((wv.get(quantsOffset)).toFloat()) * scaleFloat * x.get(j)
                partialSum2 += ((wv.get(quantsOffset + 1)).toFloat()) * scaleFloat * x.get(j + 1)
                partialSum3 += ((wv.get(quantsOffset + 2)).toFloat()) * scaleFloat * x.get(j + 2)
                partialSum4 += ((wv.get(quantsOffset + 3)).toFloat()) * scaleFloat * x.get(j + 3)
                j += localWorkGroupSize * 4
            }
        }

        var partialSum: Float = partialSum1 + partialSum2 + partialSum3 + partialSum4

        run {
            var j: Int = ((inputDim / 4) * 4) + localId
            while (j < inputDim) {
                var blockIdx: Int = j / blockSize
                var withinBlockIdx: Int = j % blockSize
                var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES
                var scaleFloat: Float = wv.getHalfFloat(blockByteOffset).getFloat32()
                partialSum += ((wv.get(blockByteOffset + 2 + withinBlockIdx)).toFloat()) * scaleFloat * x.get(j)
                j += localWorkGroupSize
            }
        }

        localSums[localId] = partialSum
        context.localBarrier()

        run {
            var stride: Int = localWorkGroupSize / 2
            while (stride > 0) {
                if (localId < stride) {
                    localSums[localId] += localSums[localId + stride]
                }
                context.localBarrier()
                stride = stride shr (1)
            }
        }

        if (localId == 0) {
            v.set(vRow, localSums[0])
        }
    }
}

/**
 * Fully fused RMS normalization + FFN W1/W3 matmul with SiLU/GLU for Q8_0 weights. Each
 * workgroup redundantly computes RMS scale to avoid cross-workgroup sync.
 */
fun fullyFusedRmsNormFFNGateUpQ8(
        context: KernelContext,
        x: FloatArray, // raw input (FP32)
        hb: FloatArray, // output
        rmsWeights: FloatArray, // RMS norm weights
        w1: ByteArray, // Q8_0 quantized
        w3: ByteArray, // Q8_0 quantized
        dim: Int, // input dimension
        hiddenDim: Int, // output dimension
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= hiddenDim) {
        return
    }

    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localWorkGroupSize)

    // ========== RMS Norm: Compute scale (each workgroup does this redundantly) ==========
    var sumSquares: Float = 0.0f
    run {
        var j: Int = localId
        while (j < dim) {
            var `val`: Float = x.get(j)
            sumSquares += `val` * `val`
            j += localWorkGroupSize
        }
    }

    localSum[localId] = sumSquares
    context.localBarrier()

    run {
        var stride: Int = localWorkGroupSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    var scale: Float = 1.0f / TornadoMath.sqrt(localSum[0] / dim + 1e-5f)

    // ========== W1 matmul with inline RMS normalization ==========
    var blockSize: Int = 32
    var Q8_0_BLOCK_BYTES: Int = 34
    var blocksPerRow: Int = (dim + blockSize - 1) / blockSize
    var rowBlockOffset: Int = rowId * blocksPerRow

    var partialSum1_a: Float = 0.0f
    var partialSum1_b: Float = 0.0f
    var partialSum1_c: Float = 0.0f
    var partialSum1_d: Float = 0.0f

    run {
        var j: Int = localId * 4
        while (j < dim - 3) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize
            var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

            var w1Scale: HalfFloat = w1.getHalfFloat(blockByteOffset)
            var w1ScaleFloat: Float = w1Scale.getFloat32()

            var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
            var q1: Byte = w1.get(quantsOffset)
            var q2: Byte = w1.get(quantsOffset + 1)
            var q3: Byte = w1.get(quantsOffset + 2)
            var q4: Byte = w1.get(quantsOffset + 3)

            var norm0: Float = rmsWeights.get(j) * scale * x.get(j)
            var norm1: Float = rmsWeights.get(j + 1) * scale * x.get(j + 1)
            var norm2: Float = rmsWeights.get(j + 2) * scale * x.get(j + 2)
            var norm3: Float = rmsWeights.get(j + 3) * scale * x.get(j + 3)

            partialSum1_a += (q1.toFloat() * w1ScaleFloat) * norm0
            partialSum1_b += (q2.toFloat() * w1ScaleFloat) * norm1
            partialSum1_c += (q3.toFloat() * w1ScaleFloat) * norm2
            partialSum1_d += (q4.toFloat() * w1ScaleFloat) * norm3
            j += localWorkGroupSize * 4
        }
    }

    var partialSum1: Float = partialSum1_a + partialSum1_b + partialSum1_c + partialSum1_d

    run {
        var j: Int = ((dim / 4) * 4) + localId
        while (j < dim) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize
            var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

            var w1Scale: HalfFloat = w1.getHalfFloat(blockByteOffset)
            var w1ScaleFloat: Float = w1Scale.getFloat32()

            var quant: Byte = w1.get(blockByteOffset + 2 + withinBlockIdx)
            var normalized: Float = rmsWeights.get(j) * scale * x.get(j)
            partialSum1 += (quant.toFloat() * w1ScaleFloat) * normalized
            j += localWorkGroupSize
        }
    }

    localSum[localId] = partialSum1
    context.localBarrier()

    run {
        var stride: Int = localWorkGroupSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }
    var result1: Float = localSum[0]

    // ========== W3 matmul with inline RMS normalization ==========
    var partialSum3_a: Float = 0.0f
    var partialSum3_b: Float = 0.0f
    var partialSum3_c: Float = 0.0f
    var partialSum3_d: Float = 0.0f

    run {
        var j: Int = localId * 4
        while (j < dim - 3) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize
            var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

            var w3Scale: HalfFloat = w3.getHalfFloat(blockByteOffset)
            var w3ScaleFloat: Float = w3Scale.getFloat32()

            var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
            var q1: Byte = w3.get(quantsOffset)
            var q2: Byte = w3.get(quantsOffset + 1)
            var q3: Byte = w3.get(quantsOffset + 2)
            var q4: Byte = w3.get(quantsOffset + 3)

            var norm0: Float = rmsWeights.get(j) * scale * x.get(j)
            var norm1: Float = rmsWeights.get(j + 1) * scale * x.get(j + 1)
            var norm2: Float = rmsWeights.get(j + 2) * scale * x.get(j + 2)
            var norm3: Float = rmsWeights.get(j + 3) * scale * x.get(j + 3)

            partialSum3_a += (q1.toFloat() * w3ScaleFloat) * norm0
            partialSum3_b += (q2.toFloat() * w3ScaleFloat) * norm1
            partialSum3_c += (q3.toFloat() * w3ScaleFloat) * norm2
            partialSum3_d += (q4.toFloat() * w3ScaleFloat) * norm3
            j += localWorkGroupSize * 4
        }
    }

    var partialSum3: Float = partialSum3_a + partialSum3_b + partialSum3_c + partialSum3_d

    run {
        var j: Int = ((dim / 4) * 4) + localId
        while (j < dim) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize
            var blockByteOffset: Int = (rowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

            var w3Scale: HalfFloat = w3.getHalfFloat(blockByteOffset)
            var w3ScaleFloat: Float = w3Scale.getFloat32()

            var quant: Byte = w3.get(blockByteOffset + 2 + withinBlockIdx)
            var normalized: Float = rmsWeights.get(j) * scale * x.get(j)
            partialSum3 += (quant.toFloat() * w3ScaleFloat) * normalized
            j += localWorkGroupSize
        }
    }

    localSum[localId] = partialSum3
    context.localBarrier()

    run {
        var stride: Int = localWorkGroupSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }
    var result3: Float = localSum[0]

    // ========== SiLU + GLU ==========
    if (localId == 0) {
        var silu: Float = result1 / (1.0f + TornadoMath.exp(-result1))
        hb.set(rowId, silu * result3)
    }
}

/**
 * Fused RMSNorm apply + Gate/Up Q8 projection + SiLU + GLU in one kernel.
 *
 * <p>Each workgroup computes one output element by:
 *
 * <ul>
 *   <li>gate[i] = dot(wUp[i], RMSNorm(x))
 *   <li>up[i] = dot(wUp[hiddenDim + i], RMSNorm(x))
 *   <li>output[i] = SiLU(gate[i]) × up[i]
 * </ul>
 *
 * @param context Kernel execution context
 * @param x Input hidden state (FP32) [dim]
 * @param output Output buffer (FP32) [hiddenDim] - ready for wDown
 * @param rmsWeights RMS normalization weights (FP32) [dim]
 * @param rmsScale Precomputed RMS scale factor [1]
 * @param wUp Combined gate+up weight matrix (Q8) [2×hiddenDim × dim]
 * @param dim Input dimension
 * @param hiddenDim Hidden dimension (output size)
 * @param localWorkGroupSize Local work group size for reduction
 */
fun fusedRmsNormFFNGateUpSiLUQ8(
        context: KernelContext,
        x: FloatArray,
        output: FloatArray,
        rmsWeights: FloatArray,
        rmsScale: FloatArray,
        wUp: ByteArray, // Q8 quantized [2×hiddenDim × dim]
        dim: Int,
        hiddenDim: Int,
        localWorkGroupSize: Int,
) {

    var rowId: Int = context.groupIdx
    var localId: Int = context.localIdx

    if (rowId >= hiddenDim) {
        return
    }

    var scale: Float = rmsScale.get(0)

    // Q8_0 format constants
    var blockSize: Int = 32
    var Q8_0_BLOCK_BYTES: Int = 34

    var localSum: kotlin.FloatArray = context.allocateFloatLocalArray(localWorkGroupSize)

    var blocksPerRow: Int = (dim + blockSize - 1) / blockSize

    // ═══════════════════════════════════════════════════════════════════════
    //                         GATE PROJECTION (row i)
    // ═══════════════════════════════════════════════════════════════════════
    var gateRowBlockOffset: Int = rowId * blocksPerRow

    var gateSum1: Float = 0.0f
    var gateSum2: Float = 0.0f
    var gateSum3: Float = 0.0f
    var gateSum4: Float = 0.0f

    var mainLoopEnd: Int = (dim / 4) * 4
    run {
        var j: Int = localId * 4
        while (j < mainLoopEnd) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize
            var blockByteOffset: Int = (gateRowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

            var qScale: HalfFloat = wUp.getHalfFloat(blockByteOffset)
            var qScaleFloat: Float = qScale.getFloat32()

            var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
            var q1: Byte = wUp.get(quantsOffset)
            var q2: Byte = wUp.get(quantsOffset + 1)
            var q3: Byte = wUp.get(quantsOffset + 2)
            var q4: Byte = wUp.get(quantsOffset + 3)

            // Inline RMS normalization
            var norm1: Float = rmsWeights.get(j) * scale * x.get(j)
            var norm2: Float = rmsWeights.get(j + 1) * scale * x.get(j + 1)
            var norm3: Float = rmsWeights.get(j + 2) * scale * x.get(j + 2)
            var norm4: Float = rmsWeights.get(j + 3) * scale * x.get(j + 3)

            gateSum1 += (q1.toFloat() * qScaleFloat) * norm1
            gateSum2 += (q2.toFloat() * qScaleFloat) * norm2
            gateSum3 += (q3.toFloat() * qScaleFloat) * norm3
            gateSum4 += (q4.toFloat() * qScaleFloat) * norm4
            j += localWorkGroupSize * 4
        }
    }

    // Tail for gate
    var tailIdx: Int = mainLoopEnd + localId
    if (tailIdx < dim) {
        var blockIdx: Int = tailIdx / blockSize
        var withinBlockIdx: Int = tailIdx % blockSize
        var blockByteOffset: Int = (gateRowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

        var qScale: HalfFloat = wUp.getHalfFloat(blockByteOffset)
        var qScaleFloat: Float = qScale.getFloat32()
        var quant: Byte = wUp.get(blockByteOffset + 2 + withinBlockIdx)

        var normalized: Float = rmsWeights.get(tailIdx) * scale * x.get(tailIdx)
        gateSum1 += (quant.toFloat() * qScaleFloat) * normalized
    }

    localSum[localId] = gateSum1 + gateSum2 + gateSum3 + gateSum4
    context.localBarrier()

    run {
        var stride: Int = localWorkGroupSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    var gateResult: Float = localSum[0]

    // ═══════════════════════════════════════════════════════════════════════
    //                      UP PROJECTION (row hiddenDim + i)
    // ═══════════════════════════════════════════════════════════════════════
    var upRowBlockOffset: Int = (hiddenDim + rowId) * blocksPerRow

    var upSum1: Float = 0.0f
    var upSum2: Float = 0.0f
    var upSum3: Float = 0.0f
    var upSum4: Float = 0.0f

    run {
        var j: Int = localId * 4
        while (j < mainLoopEnd) {
            var blockIdx: Int = j / blockSize
            var withinBlockIdx: Int = j % blockSize
            var blockByteOffset: Int = (upRowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

            var qScale: HalfFloat = wUp.getHalfFloat(blockByteOffset)
            var qScaleFloat: Float = qScale.getFloat32()

            var quantsOffset: Int = blockByteOffset + 2 + withinBlockIdx
            var q1: Byte = wUp.get(quantsOffset)
            var q2: Byte = wUp.get(quantsOffset + 1)
            var q3: Byte = wUp.get(quantsOffset + 2)
            var q4: Byte = wUp.get(quantsOffset + 3)

            // Inline RMS normalization (same values as gate)
            var norm1: Float = rmsWeights.get(j) * scale * x.get(j)
            var norm2: Float = rmsWeights.get(j + 1) * scale * x.get(j + 1)
            var norm3: Float = rmsWeights.get(j + 2) * scale * x.get(j + 2)
            var norm4: Float = rmsWeights.get(j + 3) * scale * x.get(j + 3)

            upSum1 += (q1.toFloat() * qScaleFloat) * norm1
            upSum2 += (q2.toFloat() * qScaleFloat) * norm2
            upSum3 += (q3.toFloat() * qScaleFloat) * norm3
            upSum4 += (q4.toFloat() * qScaleFloat) * norm4
            j += localWorkGroupSize * 4
        }
    }

    // Tail for up
    if (tailIdx < dim) {
        var blockIdx: Int = tailIdx / blockSize
        var withinBlockIdx: Int = tailIdx % blockSize
        var blockByteOffset: Int = (upRowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES

        var qScale: HalfFloat = wUp.getHalfFloat(blockByteOffset)
        var qScaleFloat: Float = qScale.getFloat32()
        var quant: Byte = wUp.get(blockByteOffset + 2 + withinBlockIdx)

        var normalized: Float = rmsWeights.get(tailIdx) * scale * x.get(tailIdx)
        upSum1 += (quant.toFloat() * qScaleFloat) * normalized
    }

    localSum[localId] = upSum1 + upSum2 + upSum3 + upSum4
    context.localBarrier()

    run {
        var stride: Int = localWorkGroupSize / 2
        while (stride > 0) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride]
            }
            context.localBarrier()
            stride = stride shr (1)
        }
    }

    var upResult: Float = localSum[0]

    // ═══════════════════════════════════════════════════════════════════════
    //                         SiLU(gate) × up
    // ═══════════════════════════════════════════════════════════════════════
    if (localId == 0) {
        var silu: Float = gateResult / (1.0f + TornadoMath.exp(-gateResult))
        output.set(rowId, silu * upResult)
    }
}
