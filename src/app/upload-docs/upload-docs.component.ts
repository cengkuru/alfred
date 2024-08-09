import {Component, ElementRef, OnDestroy, OnInit, ViewChild} from '@angular/core';
import { CommonModule } from '@angular/common';
import {from, Subscription} from 'rxjs';
import {AiService} from "../services/ai.service";
import {ProcessingProgress} from "../models/processing-progress.models";
import { trigger, transition, style, animate, keyframes } from '@angular/animations';
import {TextVectorizationService} from "../services/ textVectorization.service";
import {VectorizationStatus, VectorizationTask} from "../models/vectorization-task.model";
import {Functions, httpsCallable} from "@angular/fire/functions";
import {catchError, finalize, tap} from "rxjs/operators";


@Component({
  selector: 'app-upload-docs',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './upload-docs.component.html',
  animations: [
    trigger('fadeInOut', [
      transition(':enter', [
        style({ opacity: 0 }),
        animate('500ms ease-out', style({ opacity: 1 }))
      ]),
      transition(':leave', [
        animate('500ms ease-in', style({ opacity: 0 }))
      ])
    ]),
    trigger('scaleIn', [
      transition(':enter', [
        style({ transform: 'scale(0.95)', opacity: 0 }),
        animate('300ms ease-out', style({ transform: 'scale(1)', opacity: 1 }))
      ])
    ]),
    trigger('pulseAnimation', [
      transition('* => *', [
        animate('2s ease-in-out', keyframes([
          style({ opacity: 1, offset: 0 }),
          style({ opacity: 0.5, offset: 0.5 }),
          style({ opacity: 1, offset: 1 })
        ]))
      ])
    ])
  ]
})
export class UploadDocsComponent implements OnInit, OnDestroy   {
  failedTasks: VectorizationTask[] = [];
  private failedTasksSub: Subscription | null = null;
  vectorizationTasks: VectorizationTask[] = [];
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  selectedFile: File | null = null;
  processingProgress: ProcessingProgress | null = null;
  isProcessing = false;
  private processingSub: Subscription | null = null;
  isReprocessing = false;
  reprocessingSuccess = false;
  errorMessage: string | null = null;  // Properly declared errorMessage property


  constructor(
      private aiService: AiService,
      private textVectorizationService: TextVectorizationService,
      // private fns: Functions
  ) {}

  ngOnInit() {
    // this.getVectorizations(1)
    this.reprocessFailedAndPendingTasks();


  }

  getVectorizations(limit=2) {
    this.textVectorizationService.getRecentVectorizationTasks(limit).subscribe(
        tasks => {
          this.vectorizationTasks = tasks;
          console.log('Vectorization tasks:', this.vectorizationTasks);
        },
        // Error handling is now done in the service, so this error callback is optional
        // but can be used for component-specific error handling if needed
        error => console.error('Unexpected error in component:', error)
    );
  }
  checkFailedVectorizations() {
    this.failedTasksSub = this.textVectorizationService.getAllVectorizationTasks().subscribe(
        tasks => {
          this.failedTasks = tasks.filter(task => task.status === VectorizationStatus.FAILED);
          console.log('Failed vectorization tasks:', this.failedTasks);
        },
        error => console.error('Error fetching failed vectorization tasks:', error)
    );
  }

  reprocessFailedAndPendingTasks() {
    this.isReprocessing = true;
    this.reprocessingSuccess = false;

    this.textVectorizationService.reprocessFailedAndPendingTasks().subscribe({
      next: (result) => {
        console.log('Reprocessing result:', result);
        this.reprocessingSuccess = true;
        this.checkFailedVectorizations();
      },
      error: (error) => {
        console.error('Error reprocessing tasks:', error);
        // Handle the error (e.g., show an error message to the user)
      },
      complete: () => {
        this.isReprocessing = false;
      }
    });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
    } else {
      this.selectedFile = null;
    }
  }

  async uploadFile(): Promise<void> {
    if (!this.selectedFile) {
      console.error('No file selected');
      this.errorMessage = 'Please select a file before uploading.';
      return;
    }

    this.isProcessing = true;
    this.processingProgress = null;
    this.errorMessage = null;

    try {
      const text = await this.readFileAsText(this.selectedFile);
      this.processDocument(text);
    } catch (error) {
      console.error('Error reading file:', error);
      this.errorMessage = 'Failed to read the file. Please try again.';
      this.isProcessing = false;
    }
  }

  private processDocument(text: string): void {
    this.processingSub = this.aiService.processDocument(text).pipe(
        tap((progress: ProcessingProgress) => {
          this.processingProgress = progress;
        }),
        catchError((error) => {
          console.error('Error processing document:', error);
          this.errorMessage = 'An error occurred while processing the document. Please try again.';
          return [];
        }),
        finalize(() => {
          this.isProcessing = false;
          this.selectedFile = null;
          if (this.fileInput) {
            this.fileInput.nativeElement.value = '';
          }
        })
    ).subscribe();
  }


  private readFileAsText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = (e) => reject(e);
      reader.readAsText(file);
    });
  }

  ngOnDestroy(): void {
    if (this.processingSub) {
      this.processingSub.unsubscribe();
    }

    if (this.failedTasksSub) {
      this.failedTasksSub.unsubscribe();
    }
  }



}
