declare module '@tensorflow/tfjs-core/dist/hash_util' {
  import Long from 'long';

  export function hexToLong(hex: string): Long;
  export function fingerPrint64(s: Uint8Array, len?: number): Long;
}
