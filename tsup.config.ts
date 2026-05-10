import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts', 'src/react.ts'],
  format: ['esm'],
  dts: true,
  clean: true,
  sourcemap: false,
  external: ['react', 'react-dom', '@mlc-ai/web-llm'],
  target: 'esnext',
});
