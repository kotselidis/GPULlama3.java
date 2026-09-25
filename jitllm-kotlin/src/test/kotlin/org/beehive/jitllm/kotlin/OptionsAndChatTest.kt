package org.beehive.jitllm.kotlin

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import org.beehive.jitllm.api.ChatContent
import org.beehive.jitllm.api.ChatMessage
import org.beehive.jitllm.api.ChatRole
import org.beehive.jitllm.api.FinishReason
import org.beehive.jitllm.api.GenerationResult
import org.beehive.jitllm.api.GenerationTimings
import org.beehive.jitllm.api.ThinkingMode

class OptionsAndChatTest {

    @Test
    fun modelOptionsDefaultToJitllmsDefaults() {
        val options = ModelOptionsScope().build()
        assertEquals(0, options.contextLength())
        assertEquals(1, options.maxConcurrentSessions())
        assertEquals(ThinkingMode.DEFAULT, options.thinkingMode())
    }

    @Test
    fun modelOptionsCarryWhatIsSet() {
        val options = ModelOptionsScope().apply {
            contextLength = 4096
            maxConcurrentSessions = 2
            thinkingMode = ThinkingMode.DISABLED
        }.build()
        assertEquals(4096, options.contextLength())
        assertEquals(2, options.maxConcurrentSessions())
        assertEquals(ThinkingMode.DISABLED, options.thinkingMode())
    }

    @Test
    fun invalidOptionsFailWhereTheyAreWritten() {
        assertFailsWith<IllegalArgumentException> { ModelOptionsScope().apply { contextLength = -1 }.build() }
        assertFailsWith<IllegalArgumentException> { SessionOptionsScope().apply { contextLength = -1 }.build() }
    }

    @Test
    fun sessionOptionsCarryWhatIsSet() {
        val options = SessionOptionsScope().apply {
            contextLength = 1024
            thinkingMode = ThinkingMode.ENABLED
        }.build()
        assertEquals(1024, options.contextLength())
        assertEquals(ThinkingMode.ENABLED, options.thinkingMode())
    }

    @Test
    fun chatBuildsTheMessagesInOrder() {
        val call = ChatContent.ToolCall("call-1", "weather", """{"city":"Athens"}""")
        val messages = chat {
            system("Be brief.")
            user("Weather in Athens?")
            assistant(listOf(call))
            toolResult(call, """{"temperature_c":24}""")
            assistant("24 degrees and clear.")
        }
        assertEquals(
            listOf(ChatRole.SYSTEM, ChatRole.USER, ChatRole.ASSISTANT, ChatRole.TOOL, ChatRole.ASSISTANT),
            messages.map(ChatMessage::role),
        )
        assertEquals(ChatContent.ToolResult("call-1", "weather", """{"temperature_c":24}"""), messages[3].content().single())
    }

    @Test
    fun aReplyWithToolCallsIsAddedAsTheCalls() {
        val call = ChatContent.ToolCall("call-1", "weather", "{}")
        val result = GenerationResult("", 1, 1, FinishReason.TOOL_CALL, GenerationTimings(java.time.Duration.ZERO, java.time.Duration.ZERO, 1, 1), listOf(call))
        val conversation = conversation { user("Weather?") }
        conversation.addReply(result)
        assertEquals(listOf<ChatContent>(call), conversation.messages.last().content())
        assertEquals(listOf(call), result.toolCalls)
    }
}
