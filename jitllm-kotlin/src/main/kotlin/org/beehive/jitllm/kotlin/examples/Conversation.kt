@file:JvmName("Conversation")

package org.beehive.jitllm.kotlin.examples

import java.nio.file.Path
import kotlin.system.exitProcess
import org.beehive.jitllm.kotlin.JitLlm
import org.beehive.jitllm.kotlin.JitLlmSession
import org.beehive.jitllm.kotlin.conversation

/** A conversation over several turns, then a reset that forgets it. */
public suspend fun main(args: Array<String>) {
    if (args.isEmpty()) {
        System.err.println("usage: Conversation <model.gguf>")
        exitProcess(2)
    }

    JitLlm.load(Path.of(args[0])) { contextLength = 4096 }.use { model ->
        model.session().use { session ->
            val chat = conversation { system("Answer in one short sentence.") }
            session.ask(chat, "My favourite number is 7. Remember it.")
            session.ask(chat, "What is my favourite number times 6?")

            println("\n-- reset: the session forgets, the model stays loaded --")
            session.reset()
            session.ask(conversation(), "What is my favourite number?")
        }
    }
}

private suspend fun JitLlmSession.ask(chat: org.beehive.jitllm.kotlin.Conversation, question: String) {
    println("\n> $question")
    chat.add { user(question) }
    println(reply(chat) { temperature = 0.0; maxNewTokens = 96 }.text())
}
