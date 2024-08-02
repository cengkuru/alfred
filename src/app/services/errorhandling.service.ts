import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import { LoggingService } from './logging.service';

@Injectable({
    providedIn: 'root'
})
export class ErrorHandlingService {
    constructor(private loggingService: LoggingService) {}

    handleError(error: any): Observable<never> {
        let errorMessage = 'An error occurred in the AI service';

        if (error.code === 'ETIMEDOUT') {
            errorMessage = 'The request to the AI service timed out. Please try again.';
        } else if (error.code === 'permission-denied') {
            errorMessage = 'You do not have permission to use this service.';
        } else if (error.code === 'unavailable') {
            errorMessage = 'The AI service is currently unavailable. Please try again later.';
        } else if (error.message) {
            errorMessage = error.message;
        }

        this.loggingService.error('Error:', errorMessage, error);
        return throwError(() => new Error(errorMessage));
    }
}
