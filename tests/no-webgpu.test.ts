import { vi, describe, it, expect, beforeEach } from 'vitest';

vi.mock('@mlc-ai/web-llm', () => ({
  CreateMLCEngine: vi.fn(),
}));

describe('without WebGPU', () => {
  beforeEach(() => {
    // Remove gpu from navigator so isWebGPUAvailable() returns false.
    vi.stubGlobal('navigator', {});
  });

  it('isWebGPUAvailable() returns false', async () => {
    const { isWebGPUAvailable } = await import('../src/index.js');
    expect(isWebGPUAvailable()).toBe(false);
  });

  it('GemmaSession.load() rejects with WebGPUUnavailableError', async () => {
    const { GemmaSession, WebGPUUnavailableError } = await import('../src/index.js');
    await expect(
      GemmaSession.load({
        modelUrl: 'gemma-2-2b-it-q4f16_1-MLC',
        quantization: 'q4_k_m',
        contextLength: 4096,
      }),
    ).rejects.toThrow(WebGPUUnavailableError);
  });

  it('the rejection is instanceof WebGPUUnavailableError', async () => {
    const { GemmaSession, WebGPUUnavailableError } = await import('../src/index.js');
    try {
      await GemmaSession.load({
        modelUrl: 'gemma-2-2b-it-q4f16_1-MLC',
        quantization: 'q4_k_m',
        contextLength: 4096,
      });
      expect.fail('Should have thrown');
    } catch (err) {
      expect(err).toBeInstanceOf(WebGPUUnavailableError);
      expect((err as Error).name).toBe('WebGPUUnavailableError');
    }
  });

  it('CreateMLCEngine is never called when WebGPU unavailable', async () => {
    const { CreateMLCEngine } = await import('@mlc-ai/web-llm');
    const { GemmaSession } = await import('../src/index.js');
    await GemmaSession.load({
      modelUrl: 'gemma-2-2b-it-q4f16_1-MLC',
      quantization: 'q4_k_m',
      contextLength: 4096,
    }).catch(() => {});
    expect(CreateMLCEngine).not.toHaveBeenCalled();
  });
});
