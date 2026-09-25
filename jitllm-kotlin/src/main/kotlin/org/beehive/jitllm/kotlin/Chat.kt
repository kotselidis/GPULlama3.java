package org.beehive.jitllm.kotlin

import kotlin.time.Duration
import kotlin.time.toKotlinDuration
import org.beehive.jitllm.api.ChatContent
import org.beehive.jitllm.api.ChatMessage
import org.beehive.jitllm.api.ChatRole
import org.beehive.jitllm.api.GenerationResult
import org.beehive.jitllm.api.GenerationTimings

/**
 * Builds a list of chat messages.
 *
 * ```
 * session.generate(chat {
 *     system("Answer in one word.")
 *     user("What colour is the sky?")
 * })
 * ```
 */
public fun chat(block: ChatScope.() -> Unit): List<ChatMessage> =
    ChatScope().apply(block).messages.toList()

/** Starts a [Conversation] with the messages [block] adds. */
public fun conversation(block: ChatScope.() -> Unit = {}): Conversation =
    Conversation(chat(block))

/** The turns of a chat, added in order. */
@JitLlmDsl
public class ChatScope internal constructor() {

    internal val messages: MutableList<ChatMessage> = mutableListOf()

    /** Instructions for the whole conversation. */
    public fun system(text: String) {
        messages += ChatMessage.of(ChatRole.SYSTEM, text)
    }

    /** What the user says. */
    public fun user(text: String) {
        messages += ChatMessage.of(ChatRole.USER, text)
    }

    /** What the assistant said. */
    public fun assistant(text: String) {
        messages += ChatMessage.of(ChatRole.ASSISTANT, text)
    }

    /** An assistant turn that called tools instead of answering. */
    public fun assistant(toolCalls: List<ChatContent.ToolCall>) {
        messages += ChatMessage(ChatRole.ASSISTANT, toolCalls.toList())
    }

    /** The result of the tool [call], as JSON. */
    public fun toolResult(call: ChatContent.ToolCall, resultJson: String) {
        messages +=
            ChatMessage(ChatRole.TOOL, listOf(ChatContent.ToolResult(call.id(), call.name(), resultJson)))
    }

    /** Any other message. */
    public fun message(message: ChatMessage) {
        messages += message
    }
}

/**
 * A chat that grows turn by turn. [JitLlmSession.reply] answers it and adds the answer, so a
 * conversation is: [add] the user's turn, reply, repeat.
 */
public class Conversation(messages: List<ChatMessage> = emptyList()) {

    private val turns = messages.toMutableList()

    /** The messages so far. */
    public val messages: List<ChatMessage>
        get() = turns.toList()

    /** Adds the messages [block] builds. */
    public fun add(block: ChatScope.() -> Unit) {
        turns += chat(block)
    }

    /** Adds [message]. */
    public fun add(message: ChatMessage) {
        turns += message
    }

    /** Adds the assistant's turn from [result]: the tool calls it asked for, or else its text. */
    public fun addReply(result: GenerationResult) {
        turns +=
            if (result.toolCalls().isNotEmpty()) {
                ChatMessage(ChatRole.ASSISTANT, result.toolCalls().toList())
            } else {
                ChatMessage.of(ChatRole.ASSISTANT, result.text())
            }
    }
}

/** Tool calls the model asked for; empty unless the finish reason is `TOOL_CALL`. */
public val GenerationResult.toolCalls: List<ChatContent.ToolCall>
    get() = toolCalls()

/** Time spent reading the prompt. */
public val GenerationTimings.prefillTime: Duration
    get() = prefill().toKotlinDuration()

/** Time spent generating the answer. */
public val GenerationTimings.decodeTime: Duration
    get() = decode().toKotlinDuration()
