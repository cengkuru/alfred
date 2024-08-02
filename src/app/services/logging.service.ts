import { Injectable } from '@angular/core';

@Injectable({
    providedIn: 'root'
})
export class LoggingService {
    log(...args: any[]): void {
        console.log(...args);
    }

    error(...args: any[]): void {
        console.error(...args);
    }

    warn(...args: any[]): void {
        console.warn(...args);
    }

    info(...args: any[]): void {
        console.info(...args);
    }

    debug(...args: any[]): void {
        console.debug(...args);
    }
}
