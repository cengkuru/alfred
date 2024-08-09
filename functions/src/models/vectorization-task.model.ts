export interface VectorizationTask {
    id: string;
    status: VectorizationStatus;
    progress: number;
    createdAt: Date;
    updatedAt: Date;
    batchIndex: number;
    texts: string[];
    metadata: {
        intent: string;
        sentiment: string;
        entities: {
            places: string[];
            people: string[];
            organizations: string[];
            dates: string[];
        };
        numericValues: number[];
    };
}

export enum VectorizationStatus {
    PENDING = 'pending',
    PROCESSING = 'processing',
    COMPLETED = 'completed',
    FAILED = 'failed'
}
