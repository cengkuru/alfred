import { Injectable, Inject } from '@angular/core';
import { AngularFireFunctions } from '@angular/fire/compat/functions';
import { AngularFirestore } from '@angular/fire/compat/firestore';
import { Observable, throwError, of, BehaviorSubject, from } from 'rxjs';
import { catchError, map, retry, timeout, shareReplay, tap, finalize, switchMap, take } from 'rxjs/operators';
import firebase from 'firebase/compat/app';
import {TextVectorizationService} from "./ textVectorization.service";
import {LoggingService} from "./logging.service";
import {ErrorHandlingService} from "./errorhandling.service";
import {AI_SERVICE_CONFIG, AiServiceConfig} from "./ai-service.config";

export interface AiResponse {
  answer: string;
  pineconeResults: any;
  relevantContext: string;
  contextRelevance: string;
  enrichedQuestion?: string;
}

export interface FeedbackData {
  questionId: string;
  isHelpful: boolean;
  comment?: string;
}

export interface ConversationMessage {
  role: string;
  content: string;
  id: string;
  contextRelevance?: string; // Added this line
}

@Injectable({
  providedIn: 'root'
})
export class AiService {
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
      private textVectorizationService: TextVectorizationService,
      private loggingService: LoggingService,
      private errorHandlingService: ErrorHandlingService,
      @Inject(AI_SERVICE_CONFIG) private config: AiServiceConfig
  ) {
    this.initializeCounts();
  }

  private initializeCounts(): void {
    this.getSearchCount().pipe(take(1)).subscribe(
        count => this.searchCountSubject.next(count),
        error => this.errorHandlingService.handleError(error)
    );
    this.getVisitCount().pipe(take(1)).subscribe(
        count => this.visitCountSubject.next(count),
        error => this.errorHandlingService.handleError(error)
    );
  }

  askQuestion(question: string): Observable<AiResponse> {
    this.loadingSubject.next(true);
    this.logSearchQuery(question);

    const sessionId = this.getOrCreateSessionId();
    const cacheKey = `${sessionId}:${question}`;

    return this.getCachedResponse(cacheKey).pipe(
        finalize(() => this.loadingSubject.next(false))
    );
  }

  private getCachedResponse(cacheKey: string): Observable<AiResponse> {
    if (!this.cache.has(cacheKey)) {
      const response$ = this.fetchResponse(cacheKey).pipe(
          shareReplay(1)
      );
      this.cache.set(cacheKey, response$);
    }
    return this.cache.get(cacheKey)!;
  }

  private fetchResponse(cacheKey: string): Observable<AiResponse> {
    const [sessionId, question] = cacheKey.split(':');
    return from(this.textVectorizationService.enrichQuery(question)).pipe(
        switchMap(enrichedQuestion => {
          const callable = this.functions.httpsCallable('askAlfred');
          return callable({ question: enrichedQuestion, sessionId }).pipe(
              timeout(this.config.timeoutDuration),
              retry({
                count: this.config.maxRetries,
                delay: (error, retryCount) => {
                  this.loggingService.log(`Retrying question (${retryCount}/${this.config.maxRetries}): ${enrichedQuestion}`);
                  return of(1000 * Math.pow(2, retryCount)); // Exponential backoff
                }
              }),
              map(response => this.processResponse(response, enrichedQuestion)),
              catchError(error => this.errorHandlingService.handleError(error))
          );
        }),
        tap(response => {
          this.loggingService.log('AI Response:', response);
        })
    );
  }

  private processResponse(response: any, enrichedQuestion: string): AiResponse {
    if (!response || !response.answer) {
      throw new Error('Invalid response from AI service');
    }
    return {
      ...response,
      enrichedQuestion
    };
  }

  provideFeedback(feedbackData: FeedbackData): Observable<void> {
    const callable = this.functions.httpsCallable('provideFeedback');
    const sessionId = this.getOrCreateSessionId();

    return callable({ ...feedbackData, sessionId }).pipe(
        timeout(this.config.feedbackTimeout),
        tap(() => this.logFeedback(feedbackData)),
        catchError(error => this.errorHandlingService.handleError(error))
    );
  }

  private logFeedback(feedbackData: FeedbackData): void {
    this.firestore.collection('feedback').add({
      ...feedbackData,
      sessionId: this.getOrCreateSessionId(),
      timestamp: firebase.firestore.FieldValue.serverTimestamp()
    }).catch(error => this.errorHandlingService.handleError(error));
  }

  clearConversationHistory(): Observable<void> {
    const callable = this.functions.httpsCallable('clearConversationHistory');
    const sessionId = this.getOrCreateSessionId();

    return callable({ sessionId }).pipe(
        tap(() => this.cache.clear()),
        catchError(error => this.errorHandlingService.handleError(error))
    );
  }

  logSearchQuery(query: string): void {
    const sessionId = this.getOrCreateSessionId();
    this.firestore.collection('searchQueries').add({
      query,
      sessionId,
      timestamp: firebase.firestore.FieldValue.serverTimestamp()
    }).then(() => {
      const currentCount = this.searchCountSubject.value;
      this.searchCountSubject.next(currentCount + 1);
    }).catch(error => this.errorHandlingService.handleError(error));
  }

  logPageVisit(): void {
    if (this.visitLogged) return;

    const sessionId = this.getOrCreateSessionId();
    const visitRef = this.firestore.collection('visits').doc(sessionId);

    visitRef.get().pipe(take(1)).subscribe(
        doc => {
          if (!doc.exists) {
            visitRef.set({ timestamp: firebase.firestore.FieldValue.serverTimestamp() })
                .then(() => {
                  this.visitLogged = true;
                  const currentCount = this.visitCountSubject.value;
                  this.visitCountSubject.next(currentCount + 1);
                })
                .catch(error => this.errorHandlingService.handleError(error));
          } else {
            this.visitLogged = true;
          }
        },
        error => this.errorHandlingService.handleError(error)
    );
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

  clearCache(): void {
    this.cache.clear();
  }

  saveMessageToHistory(message: ConversationMessage): Observable<void> {
    const sessionId = this.getOrCreateSessionId();
    return from(this.firestore.collection('conversations').doc(sessionId).set({
      messages: firebase.firestore.FieldValue.arrayUnion(message)
    }, { merge: true })).pipe(
        catchError(error => this.errorHandlingService.handleError(error))
    );
  }

  getLoadingState(): Observable<boolean> {
    return this.loading$;
  }

  getConversationHistory(): Observable<ConversationMessage[]> {
    const sessionId = this.getOrCreateSessionId();
    return this.firestore.collection('conversations').doc(sessionId).valueChanges().pipe(
        map(doc => doc ? (doc as any).messages : []),
        catchError(error => this.errorHandlingService.handleError(error))
    );
  }
}
