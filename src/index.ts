export interface GemmaConfig {
  /** web-llm model ID or HuggingFace model URL. See MODELS.md for supported values. */
  modelUrl: string;
  quantization: 'q4_0' | 'q4_k_m' | 'fp16';
  contextLength: number;
  onProgress?: (loaded: number, total: number) => void;
}

export class WebGPUUnavailableError extends Error {
  override readonly name = 'WebGPUUnavailableError';
  constructor() {
    super(
      'WebGPU is not available in this browser. ' +
        'Use Chrome ≥113 or Edge ≥113 on a desktop with a compatible GPU.',
    );
    Object.setPrototypeOf(this, WebGPUUnavailableError.prototype);
  }
}

export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export class GemmaSession {
  private constructor(private readonly engine: WebLLMEngine) {}

  static async load(config: GemmaConfig): Promise<GemmaSession> {
    if (!isWebGPUAvailable()) throw new WebGPUUnavailableError();

    const { CreateMLCEngine } = await import('@mlc-ai/web-llm');

    const engine = await CreateMLCEngine(config.modelUrl, {
      initProgressCallback: (info: { text: string; progress: number }) => {
        const pct = Math.round((info.progress ?? 0) * 100);
        config.onProgress?.(pct, 100);
      },
    });

    config.onProgress?.(100, 100);
    return new GemmaSession(engine as unknown as WebLLMEngine);
  }

  async *generate(
    prompt: string,
    opts: { maxTokens?: number; temperature?: number; stop?: string[] } = {},
  ): AsyncIterable<string> {
    const stream = await this.engine.chat.completions.create({
      messages: [{ role: 'user', content: prompt }],
      max_tokens: opts.maxTokens ?? 2048,
      temperature: opts.temperature ?? 0.7,
      stop: opts.stop,
      stream: true,
    });
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) yield delta;
    }
  }

  async embed(text: string): Promise<Float32Array> {
    const result = await this.engine.embeddings.create({ input: text });
    return new Float32Array(result.data[0].embedding);
  }

  dispose(): void {
    this.engine.unload?.();
  }
}

type WebLLMEngine = {
  chat: {
    completions: {
      create: (params: {
        messages: Array<{ role: string; content: string }>;
        max_tokens?: number;
        temperature?: number;
        stop?: string[];
        stream: true;
      }) => Promise<AsyncIterable<{ choices: Array<{ delta: { content?: string } }> }>>;
    };
  };
  embeddings: {
    create: (params: { input: string }) => Promise<{ data: Array<{ embedding: number[] }> }>;
  };
  unload?: () => void;
};
