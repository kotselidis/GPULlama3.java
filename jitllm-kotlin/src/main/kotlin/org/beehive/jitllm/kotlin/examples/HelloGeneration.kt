@file:JvmName("HelloGeneration")

package org.beehive.jitllm.kotlin.examples

import java.nio.file.Path
import kotlin.system.exitProcess
import org.beehive.jitllm.kotlin.JitLlm
import org.beehive.jitllm.kotlin.decodeTime

/** Loads a model, asks one question and prints the answer. */
public suspend fun main(args: Array<String>) {
    if (args.isEmpty()) {
        System.err.println("usage: HelloGeneration <model.gguf> [prompt]")
        exitProcess(2)
    }
    val prompt = args.getOrElse(1) { "Why is the sky blue? Answer in two sentences." }

    JitLlm.load(Path.of(args[0])) { contextLength = 4096 }.use { model ->
        model.session().use { session ->
            val result = session.generate(prompt) { maxNewTokens = 128 }
            println(result.text())
            println(
                "${result.generatedTokens()} tokens in ${result.timings().decodeTime}, " +
                    "finished with ${result.finishReason()}",
            )
        }
    }
}
