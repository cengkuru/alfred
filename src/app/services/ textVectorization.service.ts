import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import {Observable, from, throwError, BehaviorSubject, of} from 'rxjs';
import {mergeMap, catchError, tap, finalize, map} from 'rxjs/operators';
import { AngularFireFunctions } from '@angular/fire/compat/functions';
import nlp from "compromise";
import {AngularFirestore} from "@angular/fire/compat/firestore";
import {VectorizationStatus, VectorizationTask} from "../models/vectorization-task.model";

import firebase from 'firebase/compat/app';

interface TextItem {
  text: string;
  metadata: Record<string, any>;
}

interface VectorizationState {
  totalItems: number;
  processedItems: number;
  continuationToken: string | null;
}



@Injectable({
  providedIn: 'root'
})
export class TextVectorizationService {
  private chunkSize = 2000; // Process approximately 2000 characters at a time
  private overlapSize = 200;
  private maxChunksPerBatch = 10; // Process 10 chunks per batch
  private progressSubject = new BehaviorSubject<number>(0);
  public progress$ = this.progressSubject.asObservable();

  private stopwords: Set<string> = new Set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'
  ]);

  constructor(
      private http: HttpClient,
      private fns: AngularFireFunctions,
      private firestore: AngularFirestore
  ) {}

  processDocument(filePath: string): Observable<any> {
    this.progressSubject.next(0);
    return this.readTextFile(filePath).pipe(
        tap(text => console.log(`File read successfully. Content length: ${text.length}`)),
        mergeMap(text => from(this.processText(text))),
        finalize(() => this.progressSubject.next(100)),
        catchError(error => this.handleError('processDocument', error))
    );
  }

  processJsonDataset(datasetPath: string): Observable<any> {
    this.progressSubject.next(0);
    return this.readJsonFile(datasetPath).pipe(
        mergeMap(data => this.processJsonData(data)),
        finalize(() => this.progressSubject.next(100)),
        catchError(error => this.handleError('processJsonDataset', error))
    );
  }


  public clearVectorizationProgress(): Observable<any> {
    const clearProgressFunction = this.fns.httpsCallable('clearVectorizationProgress');
    return clearProgressFunction({}).pipe(
        tap(result => console.log('Cleared vectorization progress:', result)),
        catchError(error => {
          console.error('Error clearing vectorization progress:', error);
          return throwError(() => new Error(`Failed to clear vectorization progress: ${this.getErrorMessage(error)}`));
        })
    );
  }

  private async processText(text: string): Promise<any> {
    try {
      const preprocessedText = await this.preprocessText(text);
      const chunks = this.chunkTextWithOverlap(preprocessedText);
      const totalChunks = chunks.length;

      console.log(`Total chunks to process: ${totalChunks}`);

      const textItems: TextItem[] = chunks.map(chunk => ({
        text: chunk,
        metadata: this.extractMetadata(chunk)
      }));

      const state: VectorizationState = {
        totalItems: totalChunks,
        processedItems: 0,
        continuationToken: null
      };

      const result = await this.startVectorization(textItems, state);

      if (result && result.success) {
        console.log(`Processed ${result.state.processedItems} out of ${result.state.totalItems} chunks`);
        this.progressSubject.next(100); // All chunks processed
      } else {
        throw new Error('Unexpected result structure from vectorization');
      }

      console.log('Text processing completed');
      return { success: true, message: 'Text processing completed' };
    } catch (error) {
      console.error('Error in processText:', error);
      throw error;
    }
  }

  private async startVectorization(texts: TextItem[], state: VectorizationState): Promise<any> {
    console.log(`Attempting to start vectorization for ${texts.length} chunks`);
    const startVectorizationFunction = this.fns.httpsCallable('startVectorization');

    try {
      const result = await startVectorizationFunction({ texts, state }).toPromise();
      console.log('Vectorization result:', result);

      if (result && result.data && result.data.success) {
        return result.data;
      } else {
        throw new Error('Failed to start vectorization process');
      }
    } catch (error) {
      console.error('Error in startVectorization:', error);
      if (error instanceof Error) {
        throw new Error(`Vectorization failed: ${error.message}`);
      } else {
        throw new Error('Vectorization failed: Unknown error');
      }
    }
  }
  private async preprocessText(text: string): Promise<string> {
    const doc = nlp(text);

    doc.normalize({
      whitespace: true,
      punctuation: true,
      case: true,
      unicode: true,
      contractions: true,
      acronyms: true,
      parentheses: true,
      quotations: true,
      emoji: true,
    });

    const terms: string[] = doc.terms().out('array');
    const processedTerms = terms.filter((term: string) => !this.stopwords.has(term.toLowerCase()));

    return processedTerms.join(' ');
  }

  private chunkTextWithOverlap(text: string): string[] {
    const words = text.split(/\s+/);
    const chunks: string[] = [];
    let currentChunk: string[] = [];

    for (const word of words) {
      if (currentChunk.join(' ').length + word.length > this.chunkSize) {
        chunks.push(currentChunk.join(' '));
        currentChunk = currentChunk.slice(-this.overlapSize);
      }
      currentChunk.push(word);
    }

    if (currentChunk.length > 0) {
      chunks.push(currentChunk.join(' '));
    }

    console.log(`Text chunked into ${chunks.length} parts`);
    return chunks;
  }

  private extractMetadata(chunk: string): { [key: string]: any } {
    return {
      entities: this.extractEntities(chunk),
      numericValues: this.extractNumericValues(chunk),
      sentiment: this.analyzeSentiment(chunk),
      intent: this.classifyIntent(chunk)
    };
  }


  private extractEntities(text: string): { [key: string]: string[] } {
    const words = text.split(/\s+/);
    return {
      people: words.filter(word => word.charAt(0) === word.charAt(0).toUpperCase() && word.length > 1),
      organizations: words.filter(word => word.toUpperCase() === word && word.length > 1),
      places: [],
      dates: this.extractDates(text),
    };
  }

  private extractDates(text: string): string[] {
    const dateRegex = /\b\d{1,2}\/\d{1,2}\/\d{2,4}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}\b/gi;
    return text.match(dateRegex) || [];
  }

  private extractNumericValues(text: string): number[] {
    return text.match(/\d+(\.\d+)?/g)?.map(Number) || [];
  }

  private analyzeSentiment(text: string): string {
    const positiveWords = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic'];
    const negativeWords = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing'];

    const words = text.toLowerCase().split(/\s+/);
    const positiveCount = words.filter(word => positiveWords.includes(word)).length;
    const negativeCount = words.filter(word => negativeWords.includes(word)).length;

    if (positiveCount > negativeCount) return 'positive';
    if (negativeCount > positiveCount) return 'negative';
    return 'neutral';
  }

  private classifyIntent(text: string): string {
    const intents = {
      greeting: ['hello', 'hi', 'hey', 'greetings'],
      farewell: ['bye', 'goodbye', 'see you', 'farewell'],
      question: ['what', 'why', 'how', 'when', 'where', 'who', 'which'],
      request: ['can you', 'could you', 'please', 'help'],
      complaint: ['problem', 'issue', 'wrong', 'bad', 'terrible'],
      gratitude: ['thanks', 'thank you', 'appreciate', 'grateful']
    };

    const lowercaseText = text.toLowerCase();
    for (const [intent, keywords] of Object.entries(intents)) {
      if (keywords.some(keyword => lowercaseText.includes(keyword))) {
        return intent;
      }
    }
    return 'unknown';
  }

  private readTextFile(filePath: string): Observable<string> {
    return this.http.get(filePath, { responseType: 'text' }).pipe(
        catchError(error => this.handleError('readTextFile', error))
    );
  }

  private readJsonFile(filePath: string): Observable<any> {
    return this.http.get(filePath).pipe(
        catchError(error => this.handleError('readJsonFile', error))
    );
  }

  private processJsonData(data: any): Observable<any> {
    const jsonString = JSON.stringify(data);
    return from(this.processText(jsonString));
  }

  private checkVectorizationProgress(): Observable<any> {
    const getProgressFunction = this.fns.httpsCallable('getVectorizationProgress');
    return getProgressFunction({}).pipe(
        tap(progress => {
          console.log('Vectorization progress:', progress);
          this.progressSubject.next(progress.data.percentage);
        }),
        catchError(error => {
          console.error('Error checking vectorization progress:', error);
          return throwError(() => new Error(`Failed to check vectorization progress: ${this.getErrorMessage(error)}`));
        })
    );
  }

  private handleError(operation: string, error: unknown): Observable<never> {
    console.error(`${operation} failed:`, error);
    return throwError(() => new Error(`${operation} failed: ${this.getErrorMessage(error)}`));
  }

  private getErrorMessage(error: unknown): string {
    if (error instanceof Error) return error.message;
    return String(error);
  }


  getAllVectorizationTasks(): Observable<VectorizationTask[]> {
    console.log('Fetching vectorization tasks');
    return this.firestore.collection<VectorizationTask>('vectorizationTasks').snapshotChanges().pipe(
        tap(actions => console.log('Raw Firestore actions:', actions)),
        map(actions => actions.map(a => {
          const data = a.payload.doc.data() as VectorizationTask;
          const id = a.payload.doc.id;
          return { id, ...data };
        })),
        tap(tasks => console.log('Mapped vectorization tasks:', tasks)),
        catchError(error => {
          console.error('Error in getAllVectorizationTasks:', error);
          return this.handleError('getAllVectorizationTasks', error);
        })
    );
  }

  getRecentVectorizationTasks(limit: number = 2): Observable<VectorizationTask[]> {
    console.log(`Attempting to fetch the ${limit} most recent vectorization tasks`);

    return this.firestore.collection<VectorizationTask>('vectorizationTasks').get().pipe(
        tap(snapshot => {
          console.log(`Firestore query returned ${snapshot.size} documents`);
          if (snapshot.empty) {
            console.log('The vectorizationTasks collection is empty');
          } else {
            console.log('Collection is not empty. Checking individual documents...');
            snapshot.docs.forEach((doc, index) => {
              console.log(`Document ${index + 1}:`, doc.id, doc.data());
            });
          }
        }),
        map(snapshot => snapshot.docs.map(doc => {
          const data = doc.data() as VectorizationTask;
          console.log(`Processing document ${doc.id}:`, data);
          return {
            id: doc.id,
            ...data,
            createdAt: this.ensureDate(data.createdAt),
            updatedAt: this.ensureDate(data.updatedAt)
          };
        })),
        tap(tasks => {
          console.log(`Mapped ${tasks.length} vectorization tasks`);
          tasks.forEach((task, index) => {
            console.log(`Task ${index + 1}:`, task);
          });
        }),
        catchError(error => {
          console.error('Error in getRecentVectorizationTasks:', error);
          if (error.code) {
            console.error('Firestore error code:', error.code);
          }
          return this.handleError('getRecentVectorizationTasks', error);
        })
    );
  }

  private ensureDate(date: any): Date {
    if (date instanceof firebase.firestore.Timestamp) {
      return date.toDate();
    } else if (date instanceof Date) {
      return date;
    } else if (typeof date === 'string') {
      return new Date(date);
    }
    console.warn('Unknown date format:', date);
    return new Date(); // Return current date as a fallback
  }

  reprocessFailedAndPendingTasks(): Observable<any> {
    const reprocessTasks = this.fns.httpsCallable('reprocessFailedAndPendingTasks');
    return from(reprocessTasks({})).pipe(
        tap((result: { data: any }) => {
          console.log('Reprocessing result:', result.data);
        }),
        catchError((error: Error) => {
          console.error('Error reprocessing tasks:', error);
          return throwError(() => new Error(`Failed to reprocess tasks: ${this.getErrorMessage(error)}`));
        })
    );
  }

}
