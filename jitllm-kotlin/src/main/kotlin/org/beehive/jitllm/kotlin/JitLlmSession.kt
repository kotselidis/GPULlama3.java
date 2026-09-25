package org.beehive.jitllm.kotlin

import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.beehive.jitllm.api.CancellationToken
import org.beehive.jitllm.api.ChatMessage
import org.beehive.jitllm.api.GenerationEvent
import org.beehive.jitllm.api.GenerationRequest
import org.beehive.jitllm.api.GenerationResult
import org.beehive.jitllm.api.GenerationSession
import org.beehive.jitllm.api.ToolSpec

/**
 * One conversation on a model, with suspend functions and flows.
 *
 * Generation blocks a thread for as long as it runs, so the session runs it on its dispatcher
 * ([kotlinx.coroutines.Dispatchers.IO] by default), never on the caller's. Calls on one session
 * run one at a time, in the order they arrive: a session is a single sequence of tokens, so two
 * generations cannot share it. Open several sessions to generate in parallel.
 *
 * Cancelling the calling coroutine (`job.cancel()`, `withTimeout`, a collector that stops early)
 * stops the generation at the next token. The session stays usable, and keeps what was generated
 * up to that point.
 */
public class JitLlmSession internal constructor(
    /** The Java session, for anything this API does not cover. */
    public val delegate: GenerationSession,
    private val dispatcher: CoroutineDispatcher,
) : AutoCloseable {

    private val mutex = Mutex()

    /** Number of tokens in this session's sequence so far. */
    public val position: Int
        get() = delegate.position()

    /** Continues the conversation with [prompt] and returns the whole answer. */
    public suspend fun generate(
        prompt: String,
        configure: RequestScope.() -> Unit = {},
    ): GenerationResult = execute(RequestScope(prompt, null).apply(configure), null)

    /**
     * Answers the conversation [messages], which states the whole of it. The session reuses the
     * part of its state that matches, so resending a growing conversation costs only the new turns.
     */
    public suspend fun generate(
        messages: List<ChatMessage>,
        configure: RequestScope.() -> Unit = {},
    ): GenerationResult = execute(RequestScope(null, messages).apply(configure), null)

    /**
     * Answers [conversation] and adds the answer to it as the assistant's turn: its text, or the
     * tool calls it asked for.
     */
    public suspend fun reply(
        conversation: Conversation,
        configure: RequestScope.() -> Unit = {},
    ): GenerationResult =
        generate(conversation.messages, configure).also(conversation::addReply)

    /**
     * The answer to [prompt] as it is generated, piece by piece. Each collection runs a new
     * generation; stopping the collection early (`take`, `first`, cancellation) stops it.
     */
    public fun stream(prompt: String, configure: RequestScope.() -> Unit = {}): Flow<String> =
        events(prompt, configure).texts()

    /** The answer to [messages] as it is generated; see [stream]. */
    public fun stream(
        messages: List<ChatMessage>,
        configure: RequestScope.() -> Unit = {},
    ): Flow<String> = events(messages, configure).texts()

    /**
     * The token events of the answer to [prompt]: every generated token, including those with no
     * text of their own. See [stream].
     */
    public fun events(
        prompt: String,
        configure: RequestScope.() -> Unit = {},
    ): Flow<GenerationEvent> = eventFlow { RequestScope(prompt, null).apply(configure) }

    /** The token events of the answer to [messages]; see [events]. */
    public fun events(
        messages: List<ChatMessage>,
        configure: RequestScope.() -> Unit = {},
    ): Flow<GenerationEvent> = eventFlow { RequestScope(null, messages).apply(configure) }

    /** Forgets the conversation; the model stays loaded. */
    public suspend fun reset() {
        mutex.withLock { withContext(dispatcher) { delegate.reset() } }
    }

    /** Releases the session's state. Call it when no generation is running. */
    override fun close() {
        delegate.close()
    }

    private fun Flow<GenerationEvent>.texts(): Flow<String> =
        filter(GenerationEvent::hasText).map(GenerationEvent::text)

    private fun eventFlow(request: () -> RequestScope): Flow<GenerationEvent> = channelFlow {
        // Blocks the generating thread while the buffer is full, so a slow collector slows the
        // generation instead of queueing without bound. A collector that has gone makes the
        // send fail at once, and the cancellation below stops the generation.
        execute(request()) { event -> channel.trySendBlocking(event) }
    }

    private suspend fun execute(
        scope: RequestScope,
        stream: ((GenerationEvent) -> Unit)?,
    ): GenerationResult = mutex.withLock {
        coroutineScope {
            val token = CancellationToken()
            val request = scope.build(token, stream)
            val generation = async(dispatcher) { delegate.generate(request) }
            try {
                generation.await()
            } catch (e: CancellationException) {
                // The generation does not see coroutine cancellation; the token stops it at the
                // next token, and coroutineScope waits for it to return before rethrowing, so the
                // lock is not released while the session is still generating.
                token.cancel()
                throw e
            }
        }
    }
}

/**
 * The settings of one generation. A property left `null` keeps jitllm's default (512 new tokens,
 * temperature 0.1, top-p 0.95, a random seed).
 */
@JitLlmDsl
public class RequestScope internal constructor(
    private val prompt: String?,
    private val messages: List<ChatMessage>?,
) {

    /** A system prompt for a [prompt][JitLlmSession.generate] request; `messages` carry their own. */
    public var systemPrompt: String? = null

    /** The most tokens to generate. */
    public var maxNewTokens: Int? = null

    /** Sampling temperature; 0 always picks the most likely token. */
    public var temperature: Double? = null

    /** Nucleus sampling: sample from the most likely tokens whose probabilities add up to this. */
    public var topP: Double? = null

    /** Sampling seed, for repeatable answers. */
    public var seed: Long? = null

    private val stopSequences = mutableListOf<String>()
    private val tools = mutableListOf<ToolSpec>()
    private var onEvent: ((GenerationEvent) -> Unit)? = null

    /** Ends the answer at the first of these strings, which is left out of the result. */
    public fun stop(vararg sequences: String) {
        stopSequences += sequences
    }

    /** Tools the model may call instead of answering; see [GenerationResult.toolCalls]. */
    public fun tools(vararg tools: ToolSpec) {
        this.tools += tools
    }

    /** Tools the model may call instead of answering; see [GenerationResult.toolCalls]. */
    public fun tools(tools: Collection<ToolSpec>) {
        this.tools += tools
    }

    /**
     * Called with every token event while [JitLlmSession.generate] runs, on the generating thread.
     * Keep it short; a flow from [JitLlmSession.events] is usually the better choice.
     */
    public fun onEvent(block: (GenerationEvent) -> Unit) {
        onEvent = block
    }

    internal fun build(
        token: CancellationToken,
        stream: ((GenerationEvent) -> Unit)?,
    ): GenerationRequest {
        val builder = GenerationRequest.builder()
        prompt?.let(builder::prompt)
        messages?.let(builder::messages)
        systemPrompt?.let(builder::systemPrompt)
        maxNewTokens?.let(builder::maxNewTokens)
        temperature?.let { builder.temperature(it.toFloat()) }
        topP?.let { builder.topP(it.toFloat()) }
        seed?.let(builder::seed)
        if (stopSequences.isNotEmpty()) builder.stopSequences(stopSequences.toList())
        if (tools.isNotEmpty()) builder.tools(tools.toList())
        val callback = onEvent
        when {
            callback != null && stream != null -> builder.onEvent { callback(it); stream(it) }
            callback != null -> builder.onEvent { callback(it) }
            stream != null -> builder.onEvent { stream(it) }
        }
        return builder.cancellation(token).build()
    }
}
