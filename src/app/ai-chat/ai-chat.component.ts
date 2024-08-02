import {AfterViewChecked, Component, ElementRef, OnInit, ViewChild, OnDestroy, SecurityContext} from '@angular/core';
import { CommonModule } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { AiService, AiResponse, FeedbackData } from "../services/ai.service";
import { trigger, state, style, animate, transition } from '@angular/animations';
import {Observable, Subscription, interval, Subject, takeUntil} from 'rxjs';
import {take, takeWhile, debounceTime, finalize} from 'rxjs/operators';
import { AngularFireStorage } from "@angular/fire/compat/storage";
import { AngularFirestore } from '@angular/fire/compat/firestore';
import { RouterLink } from "@angular/router";
import {DomSanitizer} from "@angular/platform-browser";

interface ChatMessage {
  content: string;
  sender: 'user' | 'ai';
  isError?: boolean;
  id?: string;
  feedback?: 'positive' | 'negative' | null;
  isTyping?: boolean;
  contextRelevance?: string;
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
        animate('200ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })),
      ]),
      transition(':leave', [
        animate('150ms ease-in', style({ opacity: 0, transform: 'translateY(10px)' })),
      ]),
    ]),
    trigger('slideInOut', [
      transition(':enter', [
        style({ transform: 'translateY(100%)' }),
        animate('200ms ease-out', style({ transform: 'translateY(0)' })),
      ]),
      transition(':leave', [
        animate('150ms ease-in', style({ transform: 'translateY(100%)' })),
      ]),
    ]),
    trigger('pulse', [
      state('inactive', style({ transform: 'scale(1)' })),
      state('active', style({ transform: 'scale(1.05)' })),
      transition('inactive <=> active', animate('150ms ease-in-out')),
    ]),
    trigger('scaleIn', [
      transition(':enter', [
        style({ opacity: 0, transform: 'scale(0.95)' }),
        animate('200ms ease-out', style({ opacity: 1, transform: 'scale(1)' })),
      ]),
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
  rotateState = 'default';
  private typingSpeed = 15; // ms per character

  visitCount$!: Observable<number>;
  searchCount$!: Observable<number>;
  private destroy$ = new Subject<void>();
  private debounceSubject = new Subject<string>();
  showStats = false;

  constructor(
      private aiService: AiService,
      private storage: AngularFireStorage,
      private firestore: AngularFirestore,
      private sanitizer: DomSanitizer // For sanitizing HTML
  ) {}

  ngOnInit() {
    this.addWelcomeMessage();
    this.aiService.logPageVisit();
    this.initializeConversation();

    this.visitCount$ = this.aiService.getVisitCountExt();
    this.searchCount$ = this.aiService.getSearchCountExt();

    this.aiService.getLoadingState()
        .pipe(takeUntil(this.destroy$))
        .subscribe(isLoading => this.isLoading = isLoading);

    this.debounceSubject.pipe(
        debounceTime(300),
        takeUntil(this.destroy$)
    ).subscribe(() => this.askQuestion());
  }

  ngAfterViewChecked() {
    this.scrollToBottom();
  }

  ngOnDestroy() {
    this.destroy$.next();
    this.destroy$.complete();
  }
  askQuestion() {
    if (this.question.trim()) {
      this.addUserMessage(this.question);
      this.pulseState = 'active';
      this.getAiResponse();
    }
  }

  debounceAskQuestion() {
    this.debounceSubject.next(this.question);
  }

  toggleStats() {
    this.showStats = !this.showStats;
    this.rotateState = this.showStats ? 'rotated' : 'default';
  }

  provideFeedback(messageId: string, isPositive: boolean) {
    const message = this.chatHistory.find(msg => msg.id === messageId);
    if (message) {
      message.feedback = isPositive ? 'positive' : 'negative';
      const feedbackData: FeedbackData = {
        questionId: messageId,
        isHelpful: isPositive,
        comment: ''
      };
      this.aiService.provideFeedback(feedbackData).pipe(
          takeUntil(this.destroy$)
      ).subscribe(
          () => console.log('Feedback submitted successfully'),
          (error) => console.error('Error submitting feedback:', error)
      );
    } else {
      console.error('Message not found for feedback');
    }
  }
  private initializeConversation() {
    this.aiService.getConversationHistory().pipe(
        take(1),
        takeUntil(this.destroy$)
    ).subscribe(
        history => {
          if (history.length === 0) {
            this.addWelcomeMessage();
          } else {
            this.chatHistory = history.map(msg => ({
              content: msg.content,
              sender: msg.role === 'user' ? 'user' : 'ai',
              id: msg.id,
              contextRelevance: msg.contextRelevance || undefined // Handle possible undefined
            }));
          }
          this.scrollToBottom();
        },
        error => {
          console.error('Error loading conversation history:', error);
          this.addWelcomeMessage();
        }
    );
  }


  private addWelcomeMessage() {
    const welcomeMessage = "<i class='bi bi-emoji-smile mr-2'></i>Hello! I'm Alfred, an AI assistant here to help you with information about Infrastructure Transparency. What would you like to know?";
    this.addMessage(welcomeMessage, 'ai', false, `welcome-${Date.now()}`);
    this.aiService.saveMessageToHistory({ role: 'assistant', content: welcomeMessage, id: `welcome-${Date.now()}` })
        .pipe(takeUntil(this.destroy$))
        .subscribe(
            () => console.log('Welcome message saved to history'),
            error => console.error('Error saving welcome message:', error)
        );
  }

  private addUserMessage(content: string) {
    this.addMessage(content, 'user');
  }

  private addMessage(content: string, sender: 'user' | 'ai', isError: boolean = false, id: string = '', feedback: 'positive' | 'negative' | null = null, isTyping: boolean = false, contextRelevance?: string) {
    this.chatHistory.push({ content, sender, isError, id, feedback, isTyping, contextRelevance });
  }


  private getAiResponse() {
    this.aiService.askQuestion(this.question).pipe(
        takeUntil(this.destroy$),
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
    this.addAiMessage('', messageId, response.contextRelevance);
    this.simulateTyping(formattedAnswer, messageId);
  }

  private addAiMessage(content: string, id: string, contextRelevance?: string) {
    const sanitizedContent = this.sanitizer.sanitize(SecurityContext.HTML, content) || '';
    this.addMessage(`<div class="bg-primary-200 rounded-apple-lg p-4 mb-4 shadow-apple-sm">${sanitizedContent}</div>`, 'ai', false, id, null, true, contextRelevance);
  }

  private async simulateTyping(content: string, messageId: string) {
    const message = this.chatHistory.find(msg => msg.id === messageId);
    if (!message) return;

    for (let i = 0; i < content.length; i++) {
      await new Promise(resolve => setTimeout(resolve, this.typingSpeed));
      message.content += content[i];
      this.scrollToBottom();
    }
    message.isTyping = false;
  }

  private formatAnswer(answer: string): string {
    let formattedAnswer = answer;

    // Split the answer into paragraphs
    const paragraphs = formattedAnswer.split('\n\n');

    // Process each paragraph
    const formattedParagraphs = paragraphs.map(paragraph => {
      // Check if the paragraph is a list
      if (paragraph.match(/^(\d+\.|-)\s/m)) {
        const listItems = paragraph.split('\n');
        const listType = listItems[0].startsWith('-') ? 'ul' : 'ol';

        const formattedList = listItems.map(item => {
          const cleanItem = item.replace(/^(\d+\.|-)\s/, '').trim();
          return `<li class="ml-4 mb-1">${cleanItem}</li>`;
        }).join('');

        return `<${listType} class="list-disc list-inside mb-3">${formattedList}</${listType}>`;
      } else {
        // Regular paragraph
        return `<p class="mb-3">${paragraph}</p>`;
      }
    });

    // Join the formatted paragraphs
    formattedAnswer = formattedParagraphs.join('');

    // Apply additional formatting
    formattedAnswer = formattedAnswer
        .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
        .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
        .replace(/(\w+):/g, '<span class="font-semibold text-gray-700">$1:</span>');

    // Handle table formatting
    formattedAnswer = formattedAnswer.replace(
        /<table>([\s\S]*?)<\/table>/g,
        (match, tableContent) => {
          const rows = tableContent.trim().split('\n');
          const formattedRows = rows.map((row: string) => {
            const cells = row.split('|').filter((cell: string) => cell.trim() !== '');
            return `<tr>${cells.map((cell: string) => `<td class="border px-3 py-2">${cell.trim()}</td>`).join('')}</tr>`;
          }).join('');
          return `<table class="table-auto border-collapse border border-gray-300 my-3">${formattedRows}</table>`;
        }
    );

    return `<div class="text-gray-800">${formattedAnswer}</div>`;
  }

  private handleError(error: any) {
    const errorMessage = this.createErrorMessage(error);
    this.addMessage("<i class='bi bi-exclamation-triangle mr-2 text-secondary-300'></i>" + errorMessage, 'ai', true);
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
    } catch (err) {
      console.error('Error scrolling to bottom:', err);
    }
  }

  triggerPulseAnimation() {
    this.pulseState = 'active';
    setTimeout(() => {
      this.pulseState = 'inactive';
    }, 300);
  }

  focusInput() {
    if (this.chatInput) {
      this.chatInput.nativeElement.focus();
    }
  }

  async clearChat() {
    const confirmed = await this.confirmClearChat();
    if (confirmed) {
      this.chatHistory = [];
      this.addWelcomeMessage();
      this.aiService.clearCache();
      this.aiService.clearConversationHistory().pipe(
          takeUntil(this.destroy$)
      ).subscribe(
          () => console.log('Conversation history cleared successfully'),
          error => console.error('Error clearing conversation history:', error)
      );
    }
  }

  private confirmClearChat(): Promise<boolean> {
    return new Promise(resolve => {
      const confirmed = window.confirm('Are you sure you want to clear the chat history?');
      resolve(confirmed);
    });
  }
}
