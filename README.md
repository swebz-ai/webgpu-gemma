# @bilkobibitkov/webgpu-gemma

Browser-native, WebGPU-accelerated Gemma inference. Tiny SDK (≤45 KB gz) + React hook, zero server.

Documents and text never leave the device. The model runs in the browser tab via [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API).

## Requirements

- Chrome ≥ 113 or Edge ≥ 113 (desktop)
- A GPU with WebGPU support

Safari < 18 and all mobile browsers are not supported.

## Install

```bash
pnpm add @bilkobibitkov/webgpu-gemma
# react subpath requires react ≥ 18
```

## Usage

### Core (framework-agnostic)

```ts
import { GemmaSession, isWebGPUAvailable } from '@bilkobibitkov/webgpu-gemma';
import { MODELS } from './models'; // see MODELS.md

if (!isWebGPUAvailable()) {
  console.error('WebGPU not available');
} else {
  const session = await GemmaSession.load({
    modelUrl: 'gemma-2-2b-it-q4f16_1-MLC', // see MODELS.md
    quantization: 'q4_k_m',
    contextLength: 4096,
    onProgress: (loaded, total) => console.log(`${loaded}/${total}`),
  });

  // Streaming generation
  for await (const token of session.generate('Summarize this text: ...', { maxTokens: 512 })) {
    process.stdout.write(token);
  }

  // Embeddings (where supported by model)
  const vec = await session.embed('Hello world');

  session.dispose();
}
```

### React hook

```tsx
import { useGemma } from '@bilkobibitkov/webgpu-gemma/react';
import { useState } from 'react';

function App() {
  const [started, setStarted] = useState(false);

  const { session, loading, error, progress } = useGemma(
    started
      ? { modelUrl: 'gemma-2-2b-it-q4f16_1-MLC', quantization: 'q4_k_m', contextLength: 4096 }
      : null,
  );

  if (loading) return <p>Loading model… {Math.round(progress * 100)}%</p>;
  if (error) return <p>Error: {error.message}</p>;

  return (
    <div>
      {!started && <button onClick={() => setStarted(true)}>Load Gemma</button>}
      {session && <YourInferenceUI session={session} />}
    </div>
  );
}
```

## API

### `isWebGPUAvailable(): boolean`

Returns `true` when `navigator.gpu` is present. Call this before rendering any load UI.

### `GemmaSession.load(config): Promise<GemmaSession>`

Downloads and initialises the model. Throws `WebGPUUnavailableError` when WebGPU is absent.

| Field | Type | Description |
|---|---|---|
| `modelUrl` | `string` | web-llm model ID or HF URL — see **MODELS.md** |
| `quantization` | `'q4_0' \| 'q4_k_m' \| 'fp16'` | Must match the model |
| `contextLength` | `number` | Token context window (e.g. 4096) |
| `onProgress` | `(loaded, total) => void` | Called with 0–100 progress integers |

### `session.generate(prompt, opts?): AsyncIterable<string>`

Streams tokens. Opts: `maxTokens` (default 2048), `temperature` (default 0.7), `stop`.

### `session.embed(text): Promise<Float32Array>`

Returns a dense embedding vector. Not all models support this; throws if unsupported.

### `session.dispose(): void`

Releases GPU memory. Always call when done.

### `useGemma(config | null): { session, loading, error, progress }`

React hook. Pass `null` to skip loading. Config changes (by value) restart the session automatically.

## Supported models

See **[MODELS.md](./MODELS.md)** for the three recommended model IDs and their HuggingFace sources.

## License notice

This SDK is MIT licensed. However, **the Gemma model weights you load at runtime are subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms)**. By using this library to load Gemma weights, you agree to abide by that license. The weights are not bundled in this package — they are downloaded from HuggingFace at runtime.
