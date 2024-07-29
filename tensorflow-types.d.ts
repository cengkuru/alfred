declare module '@tensorflow/tfjs-core/dist/hash_util' {
  type Long = any;
  export function hexToLong(hex: string): Long;
  export function fingerPrint64(s: Uint8Array, len?: number): Long;
}

declare module '@tensorflow/tfjs-core' {
  export interface Tensor {
    dataSync(): Float32Array;
    // Add other properties and methods as needed
  }

  export type Shape = number[];
  export type DataType = 'float32' | 'int32' | 'bool' | 'complex64' | 'string';
  export interface DataTypeMap {
    float32: Float32Array;
    int32: Int32Array;
    bool: Uint8Array;
    complex64: Float32Array;
    string: string[];
  }
}

declare module '@tensorflow/tfjs-layers' {
  import { Tensor, Shape, DataType } from '@tensorflow/tfjs-core';

  export interface SymbolicTensor {
    shape: Shape;
    // Add other properties as needed
  }

  export interface ModelTensorInfo {
    shape: number[];
    // Add other properties as needed
  }

  export interface LayersModel {
    inputs: SymbolicTensor[];
    // Add other properties and methods as needed
  }

  export interface InferenceModel {
    inputs: ModelTensorInfo[];
    // Add other properties and methods as needed
  }

  export type PyJsonValue = any;

  export interface NodeConfig {
    // Define properties as needed
  }

  export type MetricsIdentifier = string;
  export type SampleWeightMode = 'temporal' | string;
  export type LossWeights = any;
}

declare module '@tensorflow-models/universal-sentence-encoder' {
  import { Tensor } from '@tensorflow/tfjs-core';

  export interface UniversalSentenceEncoder {
    embed(inputs: string[]): Promise<Tensor>;
    // Add other methods or properties as needed
  }

  export function load(): Promise<UniversalSentenceEncoder>;
}
