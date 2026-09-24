@file:JvmName("ToolCalling")

package org.beehive.jitllm.kotlin.examples

import java.nio.file.Path
import kotlin.system.exitProcess
import org.beehive.jitllm.api.ChatContent
import org.beehive.jitllm.api.FinishReason
import org.beehive.jitllm.api.ToolSpec
import org.beehive.jitllm.kotlin.JitLlm
import org.beehive.jitllm.kotlin.conversation
import org.beehive.jitllm.kotlin.toolCalls

private val WEATHER =
    ToolSpec(
        "get_weather",
        "Get the current temperature in celsius for a city",
        """
        {"type":"object",
         "properties":{"city":{"type":"string","description":"City name"}},
         "required":["city"]}
        """.trimIndent() + "\n",
    )

/** The model calls a tool, gets its result, and answers with it. */
public suspend fun main(args: Array<String>) {
    if (args.isEmpty()) {
        System.err.println("usage: ToolCalling <model.gguf>")
        exitProcess(2)
    }

    JitLlm.load(Path.of(args[0])) { contextLength = 4096 }.use { model ->
        model.session().use { session ->
            val chat = conversation { user("What is the weather in Athens?") }
            repeat(3) {
                val result = session.reply(chat) {
                    tools(WEATHER)
                    maxNewTokens = 256
                    temperature = 0.0
                }
                if (result.finishReason() != FinishReason.TOOL_CALL) {
                    println("answer: ${result.text()}")
                    return
                }
                // reply() has added the assistant's tool calls; add a result for each.
                for (call in result.toolCalls) {
                    println("model called ${call.name()}(${call.argumentsJson()})")
                    chat.add { toolResult(call, execute(call)) }
                }
            }
            println("gave up after three tool rounds")
        }
    }
}

/** Stands in for real work; a production tool would parse the arguments and call something. */
private fun execute(call: ChatContent.ToolCall): String = """{"temperature_c": 24, "conditions": "clear"}"""
