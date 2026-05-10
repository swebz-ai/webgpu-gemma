# Supported Models

All models are Google Gemma weights distributed through HuggingFace and compiled for WebGPU by the MLC team. Pass the **model ID** as the `modelUrl` field in `GemmaConfig`.

> **License**: Gemma weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Downloading them implies acceptance of that license.

## Recommended — Gemma 2B Q4_K_M (~1.5 GB)

**Model ID:** `gemma-2-2b-it-q4f16_1-MLC`

```ts
{
  modelUrl: 'gemma-2-2b-it-q4f16_1-MLC',
  quantization: 'q4_k_m',
  contextLength: 4096,
}
```

- Works on mid-range laptops and desktops with 4 GB+ VRAM
- First load downloads ~1.5 GB; cached in browser IndexedDB afterwards
- HuggingFace source: https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f16_1-MLC

## Desktop only — Gemma 2B FP16 (~5 GB)

**Model ID:** `gemma-2-2b-it-q0f16-MLC`

```ts
{
  modelUrl: 'gemma-2-2b-it-q0f16-MLC',
  quantization: 'fp16',
  contextLength: 4096,
}
```

- Higher quality responses; requires 8 GB+ VRAM
- Not recommended for laptops or integrated GPUs
- HuggingFace source: https://huggingface.co/mlc-ai/gemma-2-2b-it-q0f16-MLC

## High-end devices — Gemma 9B Q4_K_M (~4.5 GB)

**Model ID:** `gemma-2-9b-it-q4f16_1-MLC`

```ts
{
  modelUrl: 'gemma-2-9b-it-q4f16_1-MLC',
  quantization: 'q4_k_m',
  contextLength: 8192,
}
```

- Best quality of the three; requires 8 GB+ VRAM and a fast GPU
- First load downloads ~4.5 GB
- HuggingFace source: https://huggingface.co/mlc-ai/gemma-2-9b-it-q4f16_1-MLC

## Notes

- Model weights are downloaded once and cached in the browser's IndexedDB by web-llm. Subsequent loads are instant.
- `contextLength` must not exceed what the model supports (4096 for 2B, 8192 for 9B).
- The `quantization` field in `GemmaConfig` is metadata describing what you loaded — it does not affect how web-llm fetches the model (the model ID determines that).
