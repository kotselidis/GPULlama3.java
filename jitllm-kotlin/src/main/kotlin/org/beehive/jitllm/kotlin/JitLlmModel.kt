package org.beehive.jitllm.kotlin

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import org.beehive.jitllm.api.ModelConfiguration
import org.beehive.jitllm.api.ModelInfo
import org.beehive.jitllm.api.SessionOptions
import org.beehive.jitllm.api.TextGenerationModel
import org.beehive.jitllm.api.ThinkingMode
import org.beehive.jitllm.runtime.policy.ExecutionPolicy

/**
 * A loaded model. Holds the weights; generation happens in the [JitLlmSession]s opened on it.
 * Close it (or `use` it) to release them.
 */
public class JitLlmModel(
    /** The Java model, for anything this API does not cover. */
    public val delegate: TextGenerationModel,
) : AutoCloseable {

    /** Name, architecture, context length and weight types. */
    public val info: ModelInfo
        get() = delegate.info()

    /** The options the model was loaded with, as resolved. */
    public val configuration: ModelConfiguration
        get() = delegate.configuration()

    /**
     * Opens a session: one conversation, with its own position and key/value state.
     *
     * @param dispatcher where the session runs its blocking generation calls
     */
    public fun session(
        dispatcher: CoroutineDispatcher = Dispatchers.IO,
        configure: SessionOptionsScope.() -> Unit = {},
    ): JitLlmSession {
        val options = SessionOptionsScope().apply(configure).build()
        return JitLlmSession(delegate.newSession(options), dispatcher)
    }

    override fun close() {
        delegate.close()
    }
}

/** The options [JitLlmModel.session] passes to [SessionOptions]. `null` keeps the model's value. */
@JitLlmDsl
public class SessionOptionsScope internal constructor() {

    /** Context length of this session, or 0 for the model's. */
    public var contextLength: Int = 0

    /** Whether reasoning models think before answering, in this session. */
    public var thinkingMode: ThinkingMode? = null

    /** Changes to the model's execution policy for this session. */
    public var executionPolicy: ExecutionPolicy.Overrides? = null

    internal fun build(): SessionOptions {
        val builder = SessionOptions.builder().contextLength(contextLength)
        thinkingMode?.let(builder::thinkingMode)
        executionPolicy?.let(builder::executionPolicy)
        return builder.build()
    }
}
