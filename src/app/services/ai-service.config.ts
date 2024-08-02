import { InjectionToken } from '@angular/core';

export interface AiServiceConfig {
    timeoutDuration: number;
    maxRetries: number;
    feedbackTimeout: number;
}

export const AI_SERVICE_CONFIG = new InjectionToken<AiServiceConfig>('ai.service.config');

export const DEFAULT_AI_SERVICE_CONFIG: AiServiceConfig = {
    timeoutDuration: 30000,
    maxRetries: 2,
    feedbackTimeout: 5000
};




