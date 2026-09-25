# Kotlin kernels spike

Can jitllm's GPU kernels be written in Kotlin? This spike ports one kernel file,
`TransformerComputeKernelsLayered` (53 kernels and helpers, 3,033 lines of Java), to Kotlin and
compares it with the Java original. Everything else in jitllm stays Java.

**Result: yes.** The Kotlin kernels generate the same GPU code as the Java ones on OpenCL, Metal
and CUDA. jitllm produces the same answers with them, and runs at the same speed.

## What changed

- `src/main/kotlin/.../kernels/TransformerComputeKernelsLayered.kt` replaces the Java file.
  `@file:JvmName("TransformerComputeKernelsLayered")` keeps the class name, so the Java code that
  builds the task-graphs (`TransformerComputeKernelsLayered::fusedRmsNormFFNGateUp`, ...) is
  unchanged.
- `pom.xml`:
  - `kotlin-maven-plugin` compiles `src/main/kotlin` before javac.
  - `kotlin-stdlib` and `tornado-kotlin-api` are compile dependencies, shaded into the jar, so any
    TornadoVM SDK with Kotlin runtime support can run it.
- Requires a TornadoVM with Kotlin support (7.0.2-dev, beehive-lab/TornadoVM#1123). Build with
  `-Dtornadovm.version=7.0.2-jdk22plus-dev`.

## Results

Llama 3.2 1B Instruct, greedy (`--temperature 0 --seed 42`), same prompt, Java build vs Kotlin
build of the same commit.

| Backend | Model | Answer | Generated kernels | Kernel tests (`-Paccel-tests`) | tok/s, Kotlin/Java |
|---|---|---|---|---|---|
| Metal (M4 Pro) | Q8_0 | identical | 11/12 byte-identical; the 12th differs only in a declaration's order and a comment | 3 pass, 7 skipped (CUDA/tensor-core only), same as Java | 0.97 |
| OpenCL (M4 Pro) | Q8_0 | identical | 10/11 byte-identical; the 11th differs only in a comment | — | 1.02 |
| CUDA (RTX 5070 Ti) | Q8_0 | identical | 9/10 byte-identical; the 10th differs only in a comment | 9 pass, 1 opt-in skip, same as Java | 0.99 |
| CUDA (RTX 5070 Ti) | F16 | identical | 9/10 byte-identical; the 10th differs only in a comment | — | 1.03 |

The comment is Graal's `// BLOCK n MERGES [...]`, whose predecessor list comes out in a different
order. tok/s is the median of 3 alternating runs; the spread between runs is 2–5%.

## Porting rules learned

- **Keep Java's loop scoping.** Each `for (int j = ...)` becomes
  `run { var j = ...; while (...) { ...; j += ... } }`. `run` is inline, so there is no object and
  no call. Hoisting `j` into the function scope instead gives it an earlier local slot, which
  renumbers loop variables in the generated kernel: still correct, but no longer byte-identical.
- **Strided loops must be `while` loops.** `for (j in a until b step s)` creates `IntRange`/
  `IntProgression` objects and calls the Kotlin standard library, including a check that can
  throw. TornadoVM cannot compile that.
- **`@Parallel` loops become `parallelFor(start, end) { i -> }`** from `tornado-kotlin-api`.
  `tornado-kotlin-api` is compiled for JVM 22, so the Kotlin code must target 22 or newer to
  inline it.
- **`float[]` becomes `kotlin.FloatArray`.** The TornadoVM `FloatArray` import owns the short name.
- **Casts become `.toFloat()`, shifts `shr`/`shl`, ternaries `if`/`else`, `~n` becomes `n.inv()`.**
  Identifiers that are Kotlin keywords (`val`) need backticks.
- **Join multi-line statements.** Kotlin ends a statement at a newline before `*`.
- kotlinc's default null checks are fine: TornadoVM's Kotlin support removes them.

## Tools

- `scripts/kotlin/j2k_kernels.py <Java file> <Kotlin file> <JvmName>` converts a kernel file.
  It covers the subset used by these files; the compiler flags anything it does not cover.
- `scripts/kotlin/kernel_equiv.py <java log> <kotlin log> <kernel regex>` compares the kernels
  printed with `--print-kernel` by two runs: byte-identical, or equivalent up to variable renaming.

## Next steps

Port the remaining 27 kernel files the same way (FP16 families, quantized formats, MoE, paged KV,
MMA, cuDNN prefill), running the kernel tests, the golden tests and the kernel comparison for each.
