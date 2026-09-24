# jitllm-kotlin

A Kotlin API for jitllm: load a model, generate with `suspend` functions, stream with `Flow`, and
stop a generation by cancelling its coroutine.

```kotlin
JitLlm.load(Path.of("Llama-3.2-1B-Instruct-Q8_0.gguf")) { contextLength = 4096 }.use { model ->
    model.session().use { session ->
        println(session.generate("Why is the sky blue?") { temperature = 0.0 }.text())

        session.stream("Name three colours.").collect { print(it) }
    }
}
```

It is a thin layer over jitllm's Java API (`org.beehive.jitllm.api`), with the same types for
results, events, messages and tools. Everything the Java API offers stays reachable through
`JitLlmModel.delegate` and `JitLlmSession.delegate`.

## Build

The library depends on the jitllm artifact, so install that first from the repository root, then
build this directory:

```bash
./mvnw install -DskipTests
cd jitllm-kotlin && ../mvnw install
```

By default it wraps `jitllm-1.0.0-jdk25` and targets JVM 25. For the JDK 21 build of jitllm, add
`-Djdk.version.suffix=-jdk21 -Djvm.target=21`.

Dependencies: Kotlin 2.4.20 and kotlinx-coroutines 1.11.0.

## API

| | |
|---|---|
| `JitLlm.load(path) { ... }` | Loads a model on `Dispatchers.IO`. The block sets `contextLength`, `backend`, `device`, `thinkingMode`, `storageOptions`, `executionPolicy`, `maxConcurrentSessions`. |
| `model.session { ... }` | Opens a session: `contextLength`, `thinkingMode`, `executionPolicy`. |
| `session.generate(prompt) { ... }` | `suspend`; returns a `GenerationResult`. The block sets `systemPrompt`, `maxNewTokens`, `temperature`, `topP`, `seed`, `stop(...)`, `tools(...)`, `onEvent { }`. |
| `session.generate(messages) { ... }` | The same for a whole conversation. |
| `session.stream(...)` | `Flow<String>`: the answer's text as it is generated. |
| `session.events(...)` | `Flow<GenerationEvent>`: every generated token, including those without text. |
| `session.reply(conversation)` | Answers a `Conversation` and adds the answer to it. |
| `chat { system(); user(); assistant(); toolResult() }` | Builds a `List<ChatMessage>`. |
| `conversation { ... }` | A `Conversation` that grows turn by turn. |
| `result.toolCalls`, `timings.prefillTime`, `timings.decodeTime` | Kotlin properties and `kotlin.time.Duration`s. |

### Threads

Generation blocks a thread while it runs, so a session runs it on its dispatcher
(`Dispatchers.IO` unless `model.session(dispatcher)` says otherwise), never on the caller's. Calls
on one session run one at a time, in order: a session is one sequence of tokens. Open several
sessions to generate in parallel.

### Cancellation

Cancelling the calling coroutine stops the generation at the next token: `job.cancel()`,
`withTimeout`, or a flow collector that stops early (`take`, `first`). The session stays usable.
Underneath, this uses jitllm's `CancellationToken`.

```kotlin
withTimeout(2.seconds) { session.generate("Write a long story.") }
session.stream("Count to one hundred.").take(5).toList()
```

## Examples

`src/main/kotlin/org/beehive/jitllm/kotlin/examples`, the Kotlin versions of jitllm's Java
examples: `HelloGeneration`, `StreamingTokens`, `Conversation`, `ToolCalling`, and
`Cancellation`. Each takes a model file:

```bash
../mvnw -q dependency:build-classpath -Dmdep.outputFile=target/cp.txt
java --add-modules jdk.incubator.vector -cp target/classes:$(cat target/cp.txt) \
     org.beehive.jitllm.kotlin.examples.HelloGeneration model.gguf
```

That runs on the CPU. To run on the GPU, add TornadoVM's flags, as the `jitllm` launcher does:
`@$TORNADOVM_HOME/tornado-argfile -Duse.tornadovm=true -Dtornado.device.memory=20GB`, plus
`-Djitllm.kvcache.fp32=true` on Metal and OpenCL.

## Tests

```bash
../mvnw test                                   # API tests, no model needed
JITLLM_TEST_MODELS=/path/to/models ../mvnw test  # also ModelTest, on the CPU
JITLLM_TEST_MODELS=/path/to/models ../mvnw test -Paccel-tests -Dtest=ModelTest  # on the GPU
```

`ModelTest` uses `Llama-3.2-1B-Instruct-Q8_0.gguf` and is skipped when the file is not there.
