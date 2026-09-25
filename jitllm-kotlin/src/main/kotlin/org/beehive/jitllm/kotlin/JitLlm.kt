package org.beehive.jitllm.kotlin

import java.nio.file.Path
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.beehive.jitllm.api.LocalModel
import org.beehive.jitllm.api.LocalModels
import org.beehive.jitllm.api.ModelOptions
import org.beehive.jitllm.api.TextGenerationModel
import org.beehive.jitllm.api.ThinkingMode
import org.beehive.jitllm.runtime.backend.BackendId
import org.beehive.jitllm.runtime.backend.DeviceSelector
import org.beehive.jitllm.runtime.policy.ExecutionPolicy
import org.beehive.jitllm.runtime.policy.StorageOptions

/** Marks the builder scopes of this API, so an inner block cannot call an outer scope by accident. */
@DslMarker
public annotation class JitLlmDsl

/**
 * Loads models.
 *
 * ```
 * JitLlm.load(Path.of("model.gguf")) { contextLength = 4096 }.use { model ->
 *     model.session().use { session ->
 *         println(session.generate("Name three colours.").text())
 *     }
 * }
 * ```
 */
public object JitLlm {

    /**
     * Loads [modelFile] on [Dispatchers.IO]: loading reads the whole file and, on an accelerator,
     * uploads the weights, which blocks for seconds.
     *
     * @throws java.io.IOException if the file cannot be read or is not a model jitllm understands
     * @throws IllegalArgumentException if the model is not a text-generation model
     */
    public suspend fun load(
        modelFile: Path,
        configure: ModelOptionsScope.() -> Unit = {},
    ): JitLlmModel {
        val options = ModelOptionsScope().apply(configure).build()
        return withContext(Dispatchers.IO) { open(LocalModels.load(modelFile, options), modelFile) }
    }

    internal fun open(model: LocalModel, modelFile: Path): JitLlmModel {
        if (model !is TextGenerationModel) {
            model.close()
            throw IllegalArgumentException("$modelFile is not a text-generation model")
        }
        return JitLlmModel(model)
    }
}

/**
 * The options [JitLlm.load] passes to [ModelOptions]. A property left `null` (or `0` for
 * [contextLength]) keeps jitllm's default.
 */
@JitLlmDsl
public class ModelOptionsScope internal constructor() {

    /** Context length to load with, or 0 for the model's own. Sizes the key/value cache. */
    public var contextLength: Int = 0

    /** How many sessions may generate at the same time. */
    public var maxConcurrentSessions: Int? = null

    /** The backend to run on, for example [BackendId.CUDA]; `null` lets jitllm choose. */
    public var backend: BackendId? = null

    /** The device to run on; `null` lets jitllm choose. */
    public var device: DeviceSelector? = null

    /** Whether reasoning models think before answering. */
    public var thinkingMode: ThinkingMode? = null

    /** How weights and caches are stored on the device, for example `StorageOptions.fp32()`. */
    public var storageOptions: StorageOptions? = null

    /** The default execution policy of every session on the model. */
    public var executionPolicy: ExecutionPolicy? = null

    internal fun build(): ModelOptions {
        val builder = ModelOptions.builder().contextLength(contextLength)
        maxConcurrentSessions?.let(builder::maxConcurrentSessions)
        backend?.let(builder::backend)
        device?.let(builder::device)
        thinkingMode?.let(builder::thinkingMode)
        storageOptions?.let(builder::storageOptions)
        executionPolicy?.let(builder::executionPolicy)
        return builder.build()
    }
}
