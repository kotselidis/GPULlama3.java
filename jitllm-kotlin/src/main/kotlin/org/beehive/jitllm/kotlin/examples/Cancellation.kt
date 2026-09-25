@file:JvmName("Cancellation")

package org.beehive.jitllm.kotlin.examples

import java.nio.file.Path
import kotlin.system.exitProcess
import kotlin.time.Duration.Companion.seconds
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.withTimeout
import org.beehive.jitllm.kotlin.JitLlm

/** Stops generations early, with a timeout and with a collector that has seen enough. */
public suspend fun main(args: Array<String>) {
    if (args.isEmpty()) {
        System.err.println("usage: Cancellation <model.gguf>")
        exitProcess(2)
    }

    JitLlm.load(Path.of(args[0])) { contextLength = 4096 }.use { model ->
        model.session().use { session ->
            try {
                withTimeout(2.seconds) {
                    session.generate("Write a long story about a lighthouse.") { maxNewTokens = 2048 }
                }
            } catch (e: TimeoutCancellationException) {
                println("stopped after 2 s, at position ${session.position}")
            }

            print("first five pieces:")
            session.stream("Count from one to one hundred in words.").take(5).collect { print(" [$it]") }
            println()

            // The session is still usable after both.
            println(session.generate("Say OK.") { maxNewTokens = 8 }.text())
        }
    }
}
