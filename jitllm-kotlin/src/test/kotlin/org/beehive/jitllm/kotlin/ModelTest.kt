package org.beehive.jitllm.kotlin

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.beehive.jitllm.api.FinishReason
import org.beehive.jitllm.runtime.policy.StorageOptions
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.BeforeAll

/**
 * The API on a real model: Llama 3.2 1B Instruct Q8_0 from `$JITLLM_TEST_MODELS`. Skipped when the
 * file is not there. Runs on the CPU unless the JVM is started with TornadoVM's flags and
 * `-Duse.tornadovm=true`.
 */
class ModelTest {

    companion object {
        private const val FILE = "Llama-3.2-1B-Instruct-Q8_0.gguf"
        private var model: JitLlmModel? = null

        @JvmStatic
        @BeforeAll
        fun load() {
            val root = System.getenv("JITLLM_TEST_MODELS")
            val file = root?.let { Path.of(it, FILE) }
            assumeTrue(file != null && Files.isRegularFile(file), "set JITLLM_TEST_MODELS to a directory with $FILE")
            model = runBlocking {
                JitLlm.load(file!!) {
                    contextLength = 1024
                    storageOptions = StorageOptions.fp32()
                }
            }
        }

        @JvmStatic
        @AfterAll
        fun close() {
            model?.close()
        }
    }

    private val session by lazy { model!!.session() }

    @AfterTest
    fun closeSession() {
        session.close()
    }

    @Test
    fun streamAndGenerateGiveTheSameAnswer() = runBlocking {
        val streamed = session.stream("Name three colours.") { temperature = 0.0; maxNewTokens = 24 }.toList()
        session.reset()
        val result = session.generate("Name three colours.") { temperature = 0.0; maxNewTokens = 24 }
        assertTrue(result.text().isNotBlank())
        assertEquals(result.text(), streamed.joinToString(""))
    }

    @Test
    fun stoppingAStreamCancelsAndTheSessionGoesOn() = runBlocking {
        val first = session.stream("Count from one to fifty in words.") { temperature = 0.0; maxNewTokens = 200 }
            .take(3).toList()
        assertEquals(3, first.size)
        val next = session.generate("Say OK.") { temperature = 0.0; maxNewTokens = 8 }
        assertTrue(next.text().isNotBlank())
    }

    @Test
    fun aConversationRemembersItsTurns() = runBlocking {
        val chat = conversation {
            system("Answer in one short sentence.")
            user("My favourite number is 7. Remember it.")
        }
        session.reply(chat) { temperature = 0.0; maxNewTokens = 32 }
        chat.add { user("What is my favourite number? Reply with the digit only.") }
        val answer = session.reply(chat) { temperature = 0.0; maxNewTokens = 8 }
        assertTrue("7" in answer.text(), answer.text())
        assertEquals(5, chat.messages.size)
        assertTrue(answer.finishReason() != FinishReason.CANCELLED)
    }
}
