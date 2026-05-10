import { useEffect, useRef, useState } from 'react';
import { GemmaSession, type GemmaConfig } from './index.js';

export function useGemma(config: GemmaConfig | null): {
  session: GemmaSession | null;
  loading: boolean;
  error: Error | null;
  progress: number;
} {
  const [session, setSession] = useState<GemmaSession | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [progress, setProgress] = useState(0);
  const activeSession = useRef<GemmaSession | null>(null);

  const modelUrl = config?.modelUrl;
  const quantization = config?.quantization;
  const contextLength = config?.contextLength;

  useEffect(() => {
    if (!config) {
      activeSession.current?.dispose();
      activeSession.current = null;
      setSession(null);
      setLoading(false);
      setError(null);
      setProgress(0);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    setProgress(0);

    GemmaSession.load({
      ...config,
      onProgress: (loaded, total) => {
        if (!cancelled && total > 0) setProgress(loaded / total);
        config.onProgress?.(loaded, total);
      },
    })
      .then((s) => {
        if (cancelled) {
          s.dispose();
          return;
        }
        activeSession.current?.dispose();
        activeSession.current = s;
        setSession(s);
        setLoading(false);
        setProgress(1);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
    // Intentionally keying on primitive fields, not config object identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelUrl, quantization, contextLength]);

  return { session, loading, error, progress };
}
