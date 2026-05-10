import { vi, describe, it, expect, beforeEach } from 'vitest';

async function* mockStream() {
  yield { choices: [{ delta: { content: 'Hello' } }] };
  yield { choices: [{ delta: { content: ' world' } }] };
}

vi.mock('@mlc-ai/web-llm', () => ({
  CreateMLCEngine: vi.fn().mockImplementation(async (_modelUrl: string, opts: { initProgressCallback?: (info: { text: string; progress: number }) => void }) => {
    opts.initProgressCallback?.({ text: '[1/1] Loading...', progress: 1 });
    return {
      chat: {
        completions: {
          create: vi.fn().mockResolvedValue(mockStream()),
        },
      },
      embeddings: {
        create: vi.fn().mockResolvedValue({ data: [{ embedding: [0.1, 0.2, 0.3] }] }),
      },
      unload: vi.fn(),
    };
  }),
}));

describe('GemmaSession (with WebGPU)', () => {
  beforeEach(() => {
    vi.stubGlobal('navigator', { gpu: {} });
  });

  it('load() resolves to a GemmaSession', async () => {
    const { GemmaSession } = await import('../src/index.js');
    const session = await GemmaSession.load({
      modelUrl: 'gemma-2-2b-it-q4f16_1-MLC',
      quantization: 'q4_k_m',
      contextLength: 4096,
    });
    expect(session).toBeDefined();
    expect(typeof session.generate).toBe('function');
    expect(typeof session.embed).toBe('function');
    expect(typeof session.dispose).toBe('function');
    session.dispose();
  });

  it('onProgress is called during load', async () => {
    const { GemmaSession } = await import('../src/index.js');
    const progress: Array<[number, number]> = [];
    await GemmaSession.load({
      modelUrl: 'gemma-2-2b-it-q4f16_1-MLC',
      quantization: 'q4_k_m',
      contextLength: 4096,
      onProgress: (loaded, total) => progress.push([loaded, total]),
    });
    expect(progress.length).toBeGreaterThan(0);
  });

  it('generate() produces an async iterable of tokens', async () => {
    const { GemmaSession } = await import('../src/index.js');
    const session = await GemmaSession.load({
      modelUrl: 'gemma-2-2b-it-q4f16_1-MLC',
      quantization: 'q4_k_m',
      contextLength: 4096,
    });
    const tokens: string[] = [];
    for await (const token of session.generate('Say hello')) {
      tokens.push(token);
    }
    expect(tokens).toEqual(['Hello', ' world']);
    session.dispose();
  });

  it('generate() with opts produces tokens without throwing', async () => {
    const { GemmaSession } = await import('../src/index.js');
    const session = await GemmaSession.load({
      modelUrl: 'gemma-2-2b-it-q4f16_1-MLC',
      quantization: 'q4_k_m',
      contextLength: 4096,
    });
    const tokens: string[] = [];
    for await (const t of session.generate('Hello', { maxTokens: 512, temperature: 0.3, stop: ['</s>'] })) {
      tokens.push(t);
    }
    expect(tokens.length).toBeGreaterThan(0);
    session.dispose();
  });

  it('embed() returns a Float32Array', async () => {
    const { GemmaSession } = await import('../src/index.js');
    const session = await GemmaSession.load({
      modelUrl: 'gemma-2-2b-it-q4f16_1-MLC',
      quantization: 'q4_k_m',
      contextLength: 4096,
    });
    const vec = await session.embed('Hello world');
    expect(vec).toBeInstanceOf(Float32Array);
    expect(vec.length).toBe(3);
    session.dispose();
  });
});

describe('isWebGPUAvailable', () => {
  it('returns true when navigator.gpu exists', () => {
    vi.stubGlobal('navigator', { gpu: {} });
    // Import synchronously already cached from the mock
    return import('../src/index.js').then(({ isWebGPUAvailable }) => {
      expect(isWebGPUAvailable()).toBe(true);
    });
  });
});
