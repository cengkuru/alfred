import { Component, OnInit, OnDestroy } from '@angular/core';
import { animate, style, transition, trigger } from "@angular/animations";
import { Subscription } from 'rxjs';
import {TextVectorizationService} from "./services/ textVectorization.service";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
  animations: [
    trigger('fadeInUp', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(20px)' }),
        animate('500ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
    ]),
  ],
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'itiReport';
  isMobileMenuOpen = false;
  processingProgress = 0;
  private progressSubscription: Subscription | null = null;

  constructor(
      private textService: TextVectorizationService
  ) {}

  ngOnInit(): void {
    this.progressSubscription = this.textService.progress$.subscribe(progress => {
      this.processingProgress = progress;
      console.log(`Processing progress: ${progress}%`);
    });


    // Uncomment these lines when you want to process the documents
    // this.processDocument();
    //this.processJson();
  }

  ngOnDestroy(): void {
    if (this.progressSubscription) {
      this.progressSubscription.unsubscribe();
    }
  }

  processDocument() {
    this.textService.processDocument('assets/data/impactStories2024.txt').subscribe({
      next: (response) => {
        console.log('Vectorization started:', response);
        // Start monitoring progress
        this.monitorVectorizationProgress();
      },
      error: (error) => {
        console.error('Error starting document processing:', error);
      },
      complete: () => {
        console.log('Document processing initiation completed');
      }
    });
  }

  processJson() {
    this.textService.processJsonDataset('assets/data/projectScores.json').subscribe({
      next: (response) => {
        console.log('JSON processing started:', response);
        // Start monitoring progress
        this.monitorVectorizationProgress();
      },
      error: (error) => {
        console.error('Error starting JSON processing:', error);
      },
      complete: () => {
        console.log('JSON processing initiation completed');
      }
    });
  }


  private monitorVectorizationProgress() {
    const progressSubscription = this.textService.progress$.subscribe({
      next: (progress) => {
        console.log('Vectorization progress:', progress);
        // Update your UI with the progress
        // For example: this.progressPercentage = progress;

        if (progress >= 100) {
          console.log('Vectorization completed');
          progressSubscription.unsubscribe();
        }
      },
      error: (error) => {
        console.error('Error monitoring vectorization progress:', error);
        progressSubscription.unsubscribe();
      }
    });
  }

}
