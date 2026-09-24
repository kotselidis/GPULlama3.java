@file:JvmName("StreamingTokens")

package org.beehive.jitllm.kotlin.examples

import java.nio.file.Path
import kotlin.system.exitProcess
import org.beehive.jitllm.kotlin.JitLlm

/** Prints the answer as it is generated, from a `Flow<String>`. */
public suspend fun main(args: Array<String>) {
    if (args.isEmpty()) {
        System.err.println("usage: StreamingTokens <model.gguf> [prompt]")
        exitProcess(2)
    }
    val prompt = args.getOrElse(1) { "Name three colours, one per line." }

    JitLlm.load(Path.of(args[0])) { contextLength = 4096 }.use { model ->
        model.session().use { session ->
            session.stream(prompt) { maxNewTokens = 128 }.collect { text ->
                print(text)
                System.out.flush()
            }
            println()
        }
    }
}
