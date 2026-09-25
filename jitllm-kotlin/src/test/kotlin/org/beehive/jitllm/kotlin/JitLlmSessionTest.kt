package org.beehive.jitllm.kotlin

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.beehive.jitllm.api.FinishReason
import org.beehive.jitllm.api.ToolSpec

class JitLlmSessionTest {

    @Test
    fun generateReturnsTheWholeAnswer() = runBlocking {
        val result = FakeSession().asKotlin().generate("Name three colours.")
        assertEquals("Red, green, blue.", result.text())
        assertEquals(FinishReason.STOP_TOKEN, result.finishReason())
    }

    @Test
    fun theScopeSetsTheRequest() = runBlocking {
        val fake = FakeSession()
        val weather = ToolSpec("weather", "Current weather in a city", """{"type":"object"}""")
        fake.asKotlin().generate("Name three colours.") {
            systemPrompt = "Be brief."
            maxNewTokens = 3
            temperature = 0.0
            topP = 0.5
            seed = 42
            stop("\n", "END")
            tools(weather)
        }
        val request = fake.requests.single()
        assertEquals("Name three colours.", request.prompt())
        assertEquals("Be brief.", request.systemPrompt())
        assertEquals(3, request.maxNewTokens())
        assertEquals(0.0f, request.temperature())
        assertEquals(0.5f, request.topP())
        assertEquals(42L, request.seed())
        assertEquals(listOf("\n", "END"), request.stopSequences())
        assertEquals(listOf(weather), request.tools())
    }

    @Test
    fun unsetPropertiesKeepJitllmsDefaults() = runBlocking {
        val fake = FakeSession()
        fake.asKotlin().generate("hi")
        val request = fake.requests.single()
        assertEquals(512, request.maxNewTokens())
        assertEquals(0.1f, request.temperature())
        assertEquals(0.95f, request.topP())
        assertNull(request.seed())
        assertNull(request.systemPrompt())
    }

    @Test
    fun messagesAreSentAsTheConversation() = runBlocking {
        val fake = FakeSession()
        val messages = chat {
            system("Answer in one word.")
            user("What colour is the sky?")
        }
        fake.asKotlin().generate(messages)
        assertEquals(messages, fake.requests.single().messages())
    }

    @Test
    fun streamEmitsTheTextAsItIsGenerated() = runBlocking {
        val pieces = FakeSession().asKotlin().stream("Name three colours.").toList()
        assertEquals(listOf("Red", ",", " green", ",", " blue", "."), pieces)
    }

    @Test
    fun streamSkipsEventsWithoutText() = runBlocking {
        val session = FakeSession(listOf("a", "", "b")).asKotlin()
        assertEquals(listOf("a", "b"), session.stream("x").toList())
        assertEquals(3, session.events("x").toList().size)
    }

    @Test
    fun onEventSeesEveryTokenOfGenerate() = runBlocking {
        val seen = mutableListOf<String>()
        FakeSession().asKotlin().generate("x") { onEvent { seen += it.text() } }
        assertEquals(6, seen.size)
    }

    @Test
    fun aCollectorThatStopsEarlyStopsTheGeneration() = runBlocking {
        val fake = FakeSession(List(1000) { "t" }, millisPerToken = 1)
        val session = fake.asKotlin()
        assertEquals(listOf("t", "t"), session.stream("x").take(2).toList())
        assertEquals("t", session.stream("x").first())
        assertTrue(fake.generated.get() < 100, "generated ${fake.generated.get()} tokens")
    }

    @Test
    fun cancellingTheCallerStopsTheGeneration() = runBlocking {
        val fake = FakeSession(List(10_000) { "t" }, millisPerToken = 1)
        val session = fake.asKotlin()
        val job = launch { session.generate("x") }
        delay(50)
        job.cancel()
        job.join()
        assertTrue(fake.requests.single().cancellation().isCancelled)
        assertTrue(fake.generated.get() < 5_000, "generated ${fake.generated.get()} tokens")
    }

    @Test
    fun aTimeoutStopsTheGenerationAndTheSessionStaysUsable() = runBlocking {
        val fake = FakeSession(List(10_000) { "t" }, millisPerToken = 1)
        val session = fake.asKotlin()
        assertFailsWith<TimeoutCancellationException> {
            withTimeout(50.milliseconds) { session.generate("x") }
        }
        assertTrue(fake.requests.single().cancellation().isCancelled)
        val next = session.generate("y") { maxNewTokens = 2 }
        assertEquals("tt", next.text())
    }

    @Test
    fun generationsOnOneSessionRunOneAtATime() = runBlocking {
        val fake = FakeSession(List(20) { "t" }, millisPerToken = 1)
        val session = fake.asKotlin()
        List(4) { async { session.generate("x") } }.awaitAll()
        assertEquals(4, fake.requests.size)
        assertEquals(1, fake.mostAtOnce)
    }

    @Test
    fun replyAddsTheAnswerToTheConversation() = runBlocking {
        val conversation = conversation { user("Name three colours.") }
        FakeSession().asKotlin().reply(conversation)
        conversation.add { user("And three more?") }
        val turns = conversation.messages.map { it.role().name to it.content().size }
        assertEquals(listOf("USER" to 1, "ASSISTANT" to 1, "USER" to 1), turns)
    }

    @Test
    fun resetAndCloseReachTheJavaSession() = runBlocking {
        val fake = FakeSession()
        val session = fake.asKotlin()
        session.generate("x")
        assertEquals(6, session.position)
        session.reset()
        assertEquals(0, session.position)
        session.close()
        assertEquals(1, fake.resets)
        assertTrue(fake.closed)
    }

    @Test
    fun timingsConvertToKotlinDurations() = runBlocking {
        val timings = FakeSession().asKotlin().generate("x").timings()
        assertEquals(3.milliseconds, timings.prefillTime)
        assertEquals(7.milliseconds, timings.decodeTime)
    }
}
