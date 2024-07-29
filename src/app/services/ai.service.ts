import { Injectable } from '@angular/core';
import { AngularFireFunctions } from '@angular/fire/compat/functions';
import { AngularFirestore } from '@angular/fire/compat/firestore';
import {Observable, throwError, of, take, BehaviorSubject, from, switchMap} from 'rxjs';
import { catchError, map, retry, timeout, shareReplay, tap, finalize } from 'rxjs/operators';
import firebase from 'firebase/compat/app';
import {TextVectorizationService} from "./ textVectorization.service";

export interface AiResponse {
  answer: string;
  pineconeResults: any;
  relevantContext: string;
  contextRelevance: string;
}

@Injectable({
  providedIn: 'root'
})
export class AiService {
  private readonly timeoutDuration = 30000; // 30 seconds
  private readonly maxRetries = 2;
  private sessionKey = 'ai_session_id';
  private visitLogged = false;
  private cache = new Map<string, Observable<AiResponse>>();

  private searchCountSubject = new BehaviorSubject<number>(0);
  private visitCountSubject = new BehaviorSubject<number>(0);
  private loadingSubject = new BehaviorSubject<boolean>(false);

  loading$ = this.loadingSubject.asObservable();

  constructor(
      private functions: AngularFireFunctions,
      private firestore: AngularFirestore,
      private textVectorizationService: TextVectorizationService
  ) {
    this.initializeCounts();
  }

  private initializeCounts() {
    this.getSearchCount().pipe(take(1)).subscribe(count => this.searchCountSubject.next(count));
    this.getVisitCount().pipe(take(1)).subscribe(count => this.visitCountSubject.next(count));
  }

  askQuestion(question: string): Observable<AiResponse> {
    this.loadingSubject.next(true);
    this.logSearchQuery(question);

    if (this.cache.has(question)) {
      this.loadingSubject.next(false);
      return this.cache.get(question)!;
    }

    const response$ = from(this.textVectorizationService.enrichQuery(question)).pipe(
        switchMap(enrichedQuestion => {
          const callable = this.functions.httpsCallable('askAI');
          return callable({ question: enrichedQuestion }).pipe(
              timeout(this.timeoutDuration),
              retry({
                count: this.maxRetries,
                delay: (error, retryCount) => {
                  console.log(`Retrying question (${retryCount}/${this.maxRetries}): ${enrichedQuestion}`);
                  return of(1000 * Math.pow(2, retryCount)); // Exponential backoff
                }
              }),
              map(response => this.processResponse(response, enrichedQuestion))
          );
        }),
        catchError(this.handleError),
        tap(response => {
          // Log successful responses
          console.log('AI Response:', response);
        }),
        finalize(() => this.loadingSubject.next(false)),
        shareReplay(1)
    );

    this.cache.set(question, response$);
    return response$;
  }

  private processResponse(response: any, enrichedQuestion: string): AiResponse {
    if (!response || !response.answer) {
      throw new Error('Invalid response from AI service');
    }
    return {
      ...response,
      enrichedQuestion
    } as AiResponse;
  }

  private handleError = (error: any): Observable<never> => {
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

    console.error('Error in AiService:', error);
    this.logError(errorMessage, error);
    return throwError(() => new Error(errorMessage));
  }

  getRelevantDocuments(topic: string): Observable<string[]> {
    const callable = this.functions.httpsCallable('getRelevantDocuments');

    return callable({ topic }).pipe(
        timeout(this.timeoutDuration),
        retry(this.maxRetries),
        map(response => this.processDocumentsResponse(response)),
        catchError(this.handleError)
    );
  }

  private processDocumentsResponse(response: any): string[] {
    if (!response || !Array.isArray(response)) {
      throw new Error('Invalid response from document retrieval service');
    }
    return response as string[];
  }

  provideFeedback(questionId: string, isHelpful: boolean, comment?: string): Observable<void> {
    const callable = this.functions.httpsCallable('provideFeedback');

    return callable({ questionId, isHelpful, comment }).pipe(
        timeout(5000), // shorter timeout for feedback
        catchError(this.handleError)
    );
  }

  logSearchQuery(query: string): void {
    this.firestore.collection('searchQueries').add({
      query,
      timestamp: firebase.firestore.FieldValue.serverTimestamp()
    }).then(() => {
      const currentCount = this.searchCountSubject.value;
      this.searchCountSubject.next(currentCount + 1);
    });
  }

  logPageVisit(): void {
    if (this.visitLogged) return;

    const sessionId = this.getOrCreateSessionId();
    const visitRef = this.firestore.collection('visits').doc(sessionId);

    visitRef.get().pipe(take(1)).subscribe(doc => {
      if (!doc.exists) {
        visitRef.set({ timestamp: firebase.firestore.FieldValue.serverTimestamp() })
            .then(() => {
              this.visitLogged = true;
              const currentCount = this.visitCountSubject.value;
              this.visitCountSubject.next(currentCount + 1);
            });
      } else {
        this.visitLogged = true;
      }
    });
  }

  private getOrCreateSessionId(): string {
    let sessionId = localStorage.getItem(this.sessionKey);
    if (!sessionId) {
      sessionId = Math.random().toString(36).substr(2, 9);
      localStorage.setItem(this.sessionKey, sessionId);
    }
    return sessionId;
  }

  getSearchCountExt(): Observable<number> {
    return this.searchCountSubject.asObservable();
  }

  getVisitCountExt(): Observable<number> {
    return this.visitCountSubject.asObservable();
  }

  private getSearchCount(): Observable<number> {
    return this.firestore.collection('searchQueries').valueChanges().pipe(
        map(queries => queries.length)
    );
  }

  private getVisitCount(): Observable<number> {
    return this.firestore.collection('visits').valueChanges().pipe(
        map(visits => visits.length)
    );
  }

  private logError(message: string, error: any): void {
    this.firestore.collection('errors').add({
      message,
      error: JSON.stringify(error),
      timestamp: firebase.firestore.FieldValue.serverTimestamp()
    });
  }

  clearCache(): void {
    this.cache.clear();
  }

  getLoadingState(): Observable<boolean> {
    return this.loading$;
  }
}
