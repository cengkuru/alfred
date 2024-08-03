import {Component, ElementRef, OnDestroy, ViewChild} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import {AiService} from "../services/ai.service";
import {ProcessingProgress} from "../models/processing-progress.models";
import { trigger, transition, style, animate, keyframes } from '@angular/animations';


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
export class UploadDocsComponent implements OnDestroy {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  selectedFile: File | null = null;
  processingProgress: ProcessingProgress | null = null;
  isProcessing = false;
  private processingSub: Subscription | null = null;

  constructor(private aiService: AiService) {}

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
      return;
    }

    this.isProcessing = true;
    this.processingProgress = null;
    const text = await this.readFileAsText(this.selectedFile);

    this.processingSub = this.aiService.processDocument(text).subscribe(
        (progress: ProcessingProgress) => {
          this.processingProgress = progress;
        },
        (error) => {
          console.error('Error processing document:', error);
          this.isProcessing = false;
          this.processingProgress = null;
        },
        () => {
          console.log('Document processing complete');
          this.isProcessing = false;
          this.selectedFile = null;
          this.fileInput.nativeElement.value = '';
        }
    );
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
  }
}
