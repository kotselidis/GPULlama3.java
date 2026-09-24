package org.beehive.jitllm.kotlin

import java.time.Duration
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.Dispatchers
import org.beehive.jitllm.api.FinishReason
import org.beehive.jitllm.api.GenerationEvent
import org.beehive.jitllm.api.GenerationRequest
import org.beehive.jitllm.api.GenerationResult
import org.beehive.jitllm.api.GenerationSession
import org.beehive.jitllm.api.GenerationTimings

/**
 * A session that "generates" [tokens] one by one, [millisPerToken] apart, and behaves as the real
 * one does towards a request: it sends each token to the request's event consumer and stops at
 * the next token once the request's cancellation token is cancelled.
 */
class FakeSession(
    private val tokens: List<String> = listOf("Red", ",", " green", ",", " blue", "."),
    private val millisPerToken: Long = 0,
) : GenerationSession {

    val requests: MutableList<GenerationRequest> = CopyOnWriteArrayList()
    val generated = AtomicInteger()
    private val running = AtomicInteger()
    @Volatile var mostAtOnce = 0
    @Volatile var resets = 0
    @Volatile var closed = false
    @Volatile private var tokensSoFar = 0

    fun asKotlin(): JitLlmSession = JitLlmSession(this, Dispatchers.IO)

    override fun generate(request: GenerationRequest): GenerationResult {
        requests += request
        mostAtOnce = maxOf(mostAtOnce, running.incrementAndGet())
        try {
            val text = StringBuilder()
            var count = 0
            var reason = FinishReason.STOP_TOKEN
            for ((id, piece) in tokens.take(request.maxNewTokens()).withIndex()) {
                if (request.cancellation()?.isCancelled == true) {
                    reason = FinishReason.CANCELLED
                    break
                }
                if (millisPerToken > 0) Thread.sleep(millisPerToken)
                request.onEvent()?.accept(GenerationEvent(id, piece))
                text.append(piece)
                count++
                generated.incrementAndGet()
            }
            if (reason != FinishReason.CANCELLED && request.maxNewTokens() < tokens.size) {
                reason = FinishReason.MAX_TOKENS
            }
            tokensSoFar += count
            return GenerationResult(
                text.toString(), 1, count, reason, GenerationTimings(Duration.ofMillis(3), Duration.ofMillis(7), 1, count),
            )
        } finally {
            running.decrementAndGet()
        }
    }

    override fun position(): Int = tokensSoFar

    override fun reset() {
        resets++
        tokensSoFar = 0
    }

    override fun close() {
        closed = true
    }
}
