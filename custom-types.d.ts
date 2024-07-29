declare module '@tensorflow/tfjs-core/dist/hash_util' {
  export function hexToLong(hex: string): any;
  export function fingerPrint64(s: Uint8Array, len?: number): any;
}

declare module '@tensorflow/tfjs-core' {
  export interface Tensor {
    dataSync(): Float32Array;
    // Add any other methods or properties you're using from Tensor
  }
}

declare module '@tensorflow-models/universal-sentence-encoder' {
  import {Tensor} from "@tensorflow/tfjs-core";

  export interface UniversalSentenceEncoder {
    embed(inputs: string[]): Promise<Tensor>;
    // Add any other methods or properties you're using from UniversalSentenceEncoder
  }

  export function load(): Promise<UniversalSentenceEncoder>;
}
