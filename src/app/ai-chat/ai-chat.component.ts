import { AfterViewChecked, Component, ElementRef, OnInit, ViewChild, OnDestroy } from '@angular/core';
import { CommonModule } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { AiService, AiResponse } from "../services/ai.service";
import { trigger, state, style, animate, transition } from '@angular/animations';
import {interval, Observable, Subscription, takeWhile} from 'rxjs';
import { catchError, finalize, tap } from 'rxjs/operators';
import firebase from 'firebase/compat/app';
import { AngularFireStorage } from "@angular/fire/compat/storage";
import { AngularFirestore } from '@angular/fire/compat/firestore';
import { RouterLink } from "@angular/router";

interface ChatMessage {
  content: string;
  sender: 'user' | 'ai';
  isError?: boolean;
  id?: string;
  feedback?: 'positive' | 'negative' | null;
  isTyping?: boolean;
}

@Component({
  selector: 'app-ai-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './ai-chat.component.html',
  styleUrls: ['./ai-chat.component.scss'],
  animations: [
    trigger('fadeInOut', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(10px)' }),
        animate('300ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
      transition(':leave', [
        animate('300ms ease-in', style({ opacity: 0, transform: 'translateY(10px)' })),
      ]),
    ]),
    trigger('slideInOut', [
      transition(':enter', [
        style({ transform: 'translateY(100%)' }),
        animate('300ms ease-out', style({ transform: 'translateY(0)' })),
      ]),
      transition(':leave', [
        animate('300ms ease-in', style({ transform: 'translateY(100%)' })),
      ]),
    ]),
    trigger('pulse', [
      state('inactive', style({ transform: 'scale(1)' })),
      state('active', style({ transform: 'scale(1.1)' })),
      transition('inactive <=> active', animate('300ms ease-in-out')),
    ]),
  ],
})
export class AiChatComponent implements OnInit, AfterViewChecked, OnDestroy {
  @ViewChild('chatContainer') private chatContainer!: ElementRef;
  @ViewChild('chatInput') private chatInput!: ElementRef;

  question = '';
  chatHistory: ChatMessage[] = [];
  isLoading = false;
  pulseState = 'inactive';
  relevantDocuments: string[] = [];
  private typingSpeed = 10; // ms per character

  visitCount$: Observable<number>;
  searchCount$: Observable<number>;
  loadingSubscription: Subscription;

  // Toggle state for statistics visibility
  showStats = false;

  constructor(
      private aiService: AiService,
      private storage: AngularFireStorage,
      private firestore: AngularFirestore,
  ) {
    this.visitCount$ = this.aiService.getVisitCountExt();
    this.searchCount$ = this.aiService.getSearchCountExt();
    this.loadingSubscription = this.aiService.getLoadingState().subscribe(
        isLoading => this.isLoading = isLoading
    );
  }


  ngOnInit() {
    this.addWelcomeMessage();
    this.aiService.logPageVisit();
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  ngOnDestroy() {
    if (this.loadingSubscription) {
      this.loadingSubscription.unsubscribe();
    }
  }

  toggleStats() {
    this.showStats = !this.showStats;
  }

  askQuestion() {
    if (this.question.trim()) {
      this.addUserMessage(this.question);
      this.pulseState = 'active';
      this.getAiResponse();
    }
  }

  provideFeedback(messageId: string, isPositive: boolean) {
    const message = this.chatHistory.find(msg => msg.id === messageId);
    if (message) {
      message.feedback = isPositive ? 'positive' : 'negative';
      this.aiService.provideFeedback(messageId, isPositive).subscribe(
          () => console.log('Feedback submitted successfully'),
          (error) => console.error('Error submitting feedback:', error)
      );
    }
  }

  private addWelcomeMessage() {
    this.chatHistory.push({
      content: "<i class='bi bi-emoji-smile mr-2'></i>Hello! I'm Alfred, an AI assistant here to help you with information about the Infrastructure Transparency. What would you like to know?",
      sender: 'ai'
    });
  }

  private addUserMessage(content: string) {
    this.chatHistory.push({ content, sender: 'user' });
  }

  private getAiResponse() {
    this.aiService.askQuestion(this.question).pipe(
        finalize(() => {
          this.question = '';
          this.pulseState = 'inactive';
        })
    ).subscribe(
        (response: AiResponse) => this.handleAiResponse(response),
        (error) => this.handleError(error)
    );
  }

  private handleAiResponse(response: AiResponse) {
    const formattedAnswer = this.formatAnswer(response.answer);
    const messageId = `msg-${Date.now()}`;
    this.addAiMessage('', messageId); // Start with an empty message
    this.simulateTyping(formattedAnswer, messageId);
    this.addContextInfo(response, messageId);
  }

  private addAiMessage(content: string, id: string) {
    this.chatHistory.push({
      id,
      content: `<div class="bg-gray-100 rounded-lg p-4 mb-4 shadow-sm">${content}</div>`,
      sender: 'ai',
      isTyping: true
    });
  }

  private simulateTyping(content: string, messageId: string) {
    let index = 0;
    const message = this.chatHistory.find(msg => msg.id === messageId);
    if (!message) return;

    const typingInterval = interval(this.typingSpeed).pipe(
        takeWhile(() => index < content.length)
    ).subscribe(() => {
      message.content += content[index];
      index++;
      if (index === content.length) {
        message.isTyping = false;
      }
      this.scrollToBottom();
    });
  }

  private addContextInfo(response: AiResponse, messageId: string) {
    if (response.contextRelevance) {
      this.chatHistory.push({
        content: `<i class='bi bi-info-circle mr-2'></i>Context Relevance: ${response.contextRelevance}`,
        sender: 'ai',
        id: `${messageId}-context`
      });
    }
  }

  private formatAnswer(answer: string): string {
    // Split the answer into paragraphs
    const paragraphs = answer.split('\n\n');

    // Process each paragraph
    const formattedParagraphs = paragraphs.map(paragraph => {
      // Check if the paragraph is a list
      if (paragraph.match(/^(\d+\.|-)\s/m)) {
        const listItems = paragraph.split('\n');
        const listType = listItems[0].startsWith('-') ? 'ul' : 'ol';

        const formattedList = listItems.map(item => {
          const cleanItem = item.replace(/^(\d+\.|-)\s/, '').trim();
          return `<li class="ml-5 mb-2">${cleanItem}</li>`;
        }).join('');

        return `<${listType} class="list-disc list-inside mb-4">${formattedList}</${listType}>`;
      } else {
        // Regular paragraph
        return `<p class="mb-4">${paragraph}</p>`;
      }
    });

    // Join the formatted paragraphs
    let formattedAnswer = formattedParagraphs.join('');

    // Apply additional formatting
    formattedAnswer = formattedAnswer
        .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
        .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
        .replace(/(\w+):/g, '<span class="font-semibold secondary">$1:</span>');

    // Add an icon to the beginning of the answer
    return `<div class="flex items-start">
      <i class="bi bi-chat-left-text mr-2 mt-1 secondary"></i>
      <div>${formattedAnswer}</div>
    </div>`;
  }

  private handleError(error: any) {
    const errorMessage = this.createErrorMessage(error);
    this.chatHistory.push({
      content: "<i class='bi bi-exclamation-triangle mr-2'></i>" + errorMessage,
      sender: 'ai',
      isError: true
    });
  }

  private createErrorMessage(error: any): string {
    let message = "I apologize, but I encountered an issue. ";
    if (error.message.includes('404')) {
      message += "The requested information couldn't be found. Could you please rephrase your question or ask about a different topic?";
    } else {
      message += error.message || 'An unexpected error occurred. Please try again later.';
    }
    return message;
  }

  private scrollToBottom(): void {
    try {
      this.chatContainer.nativeElement.scrollTop = this.chatContainer.nativeElement.scrollHeight;
    } catch(err) {
      console.error('Error scrolling to bottom:', err);
    }
  }

  // New method to trigger pulse animation
  triggerPulseAnimation() {
    this.pulseState = 'active';
    setTimeout(() => {
      this.pulseState = 'inactive';
    }, 300);
  }

  // New method to focus on the input field
  focusInput() {
    if (this.chatInput) {
      this.chatInput.nativeElement.focus();
    }
  }

  clearChat() {
    this.chatHistory = [];
    this.addWelcomeMessage();
    this.aiService.clearCache();
  }
}
