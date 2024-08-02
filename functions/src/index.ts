import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Configuration, OpenAIApi } from 'openai';
import axios from 'axios';
import {https, logger} from "firebase-functions";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as natural from 'natural';


// Initialize Firebase Admin SDK if not already done
if (!admin.apps.length) {
  admin.initializeApp();
}



const OPENAI_API_KEY = functions.config().openai.apikey;
const PINECONE_API_KEY = functions.config().pinecone.apikey;
const PINECONE_INDEX_NAME = 'alfred';

const pinecone = new Pinecone({
  apiKey: PINECONE_API_KEY,
});

const BATCH_SIZE = 100;

const embeddings = new OpenAIEmbeddings({ openAIApiKey: OPENAI_API_KEY });

const openai = new OpenAIApi(new Configuration({ apiKey: OPENAI_API_KEY }));

interface TextItem {
  text: string;
  metadata: Record<string, any>;
}



async function retryOperation<T>(operation: () => Promise<T>, maxRetries = 3): Promise<T> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, i)));
    }
  }
  throw new Error('Max retries reached');
}

async function batchEmbedTexts(texts: string[]): Promise<number[][]> {
  return retryOperation(async () => {
    const embeddingResponse = await openai.createEmbedding({
      model: "text-embedding-ada-002",
      input: texts,
    });
    return embeddingResponse.data.data.map(item => item.embedding);
  });
}
async function updateProgress(processedItems: number, totalItems: number): Promise<void> {
  await admin.firestore().collection('vectorizationProgress').doc('current').set({
    processedItems,
    totalItems,
    percentage: (processedItems / totalItems) * 100
  }, { merge: true });
}


interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ConversationHistory {
  messages: Message[];
}

interface RelevantContext {
  context: string;
  pineconeResults: any;
}

// Improved keyword search function using TF-IDF
function keywordSearch(question: string, documents: string[]): string[] {
  const TfIdf = natural.TfIdf;
  const tfidf = new TfIdf();

  documents.forEach(doc => tfidf.addDocument(doc));

  const tokenizer = new natural.WordTokenizer();
  const questionTokens = tokenizer.tokenize(question.toLowerCase());

  const scores = documents.map((_, index) => {
    return questionTokens.reduce((score, token) => {
      return score + tfidf.tfidf(token, index);
    }, 0);
  });

  const topIndices = scores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map(item => item.index);

  return topIndices.map(index => documents[index]);
}

function rerank(matches: any[], question: string): any[] {
  const keywords = question.toLowerCase().split(' ');

  return matches.map(match => {
    const text = match.metadata?.text?.toLowerCase() || '';
    const keywordScore = keywords.reduce((score, keyword) =>
        score + (text.includes(keyword) ? 1 : 0), 0);
    return { ...match, keywordScore };
  }).sort((a, b) => {
    // Combine vector similarity score and keyword score
    const scoreA = (a.score || 0) + a.keywordScore;
    const scoreB = (b.score || 0) + b.keywordScore;
    return scoreB - scoreA;
  });
}

async function hybridSearch(question: string, conversationHistory: ConversationHistory): Promise<RelevantContext> {
  const index = pinecone.Index(PINECONE_INDEX_NAME);
  const expandedQuery = await expandQuery(question, conversationHistory);
  const queryEmbedding = await embeddings.embedQuery(expandedQuery);

  const queryResponse = await index.query({
    vector: queryEmbedding,
    topK: 20,
    includeValues: true,
    includeMetadata: true,
  });

  const documents = queryResponse.matches
      .map(match => match.metadata?.text)
      .filter((text): text is string => typeof text === 'string');

  const keywordResults = keywordSearch(question, documents);

  const combinedResults = rerank(queryResponse.matches, question)
      .filter(match => match.metadata?.text && keywordResults.includes(match.metadata.text))
      .slice(0, 10);  // Keep top 10 results after reranking

  return {
    context: combinedResults.map(match => match.metadata?.text || '').join('\n\n'),
    pineconeResults: combinedResults
  };
}

function postProcessAnswer(rawAnswer: string): string {
  let processedAnswer = rawAnswer.trim();

  if (!/[.!?]$/.test(processedAnswer)) {
    processedAnswer += '.';
  }

  processedAnswer = processedAnswer.charAt(0).toUpperCase() + processedAnswer.slice(1);

  return processedAnswer;
}
async function expandQuery(question: string, conversationHistory: ConversationHistory): Promise<string> {
  const recentMessages = conversationHistory.messages.slice(-3).map(m => m.content).join(' ');
  const expandedQuery = `${recentMessages} ${question}`;

  const response = await axios.post(ANTHROPIC_API_URL, {
    model: "claude-3-haiku-20240307",
    max_tokens: 50,
    messages: [{
      role: "user",
      content: `Given the conversation context and question, generate an expanded search query to find relevant information:
      
      Context: ${recentMessages}
      Question: ${question}
      
      Expanded query:`
    }]
  }, {
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01'
    },
  });

  return response.data?.content?.[0]?.text || expandedQuery;
}

const ANTHROPIC_API_KEY = functions.config().anthropic.apikey;
const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';


async function chunkText(text: string, chunkSize: number = 600, overlap: number = 50): Promise<string[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: chunkSize,
    chunkOverlap: overlap,
  });
  return await splitter.splitText(text);
}
// Improved: Prompt generation function
async function generatePrompt(question: string, context: string, conversationHistory: ConversationHistory): Promise<string> {
  const chunks = await chunkText(context);
  const contextPrompt = chunks.map(chunk => `Context: ${chunk}\n`).join('\n');

  const recentConversation = conversationHistory.messages.slice(-5).map(m => `${m.role}: ${m.content}`).join('\n');

  return `
${contextPrompt}

Recent Conversation:
${recentConversation}

You are Alfred, an AI assistant specializing in Infrastructure Transparency and the Construction Sector Transparency Initiative (CoST). Your knowledge spans infrastructure project lifecycles, transparency in public contracting, and socio-economic impacts of infrastructure investments.

Current Question: ${question}

Instructions:
1. Analyze the question, context, and recent conversation to provide a coherent and relevant response.
2. If this is a follow-up question, make sure to reference and build upon the previous parts of the conversation.
3. Provide a comprehensive answer that directly addresses the user's query, using multiple paragraphs if necessary.
4. Use markdown formatting to enhance readability, including headers, lists, and emphasis where appropriate.
5. Include relevant statistics, examples, or case studies from the provided context when applicable.
6. If the information provided is insufficient or unclear, acknowledge this and suggest potential avenues for further research.
7. End your response with a thought-provoking question or suggestion for further exploration of the topic.
8. Important: Do not generate additional questions or responses on behalf of the user. Wait for the user's next input.

Begin your response now:`;
}



export const askAlfred = functions.runWith({
  timeoutSeconds: 300,
  memory: '1GB'
}).https.onCall(async (data, context) => {
  const { question, sessionId } = data;

  if (!question || typeof question !== 'string') {
    throw new functions.https.HttpsError('invalid-argument', 'Please provide a question as a string.');
  }

  if (!sessionId || typeof sessionId !== 'string') {
    throw new functions.https.HttpsError('invalid-argument', 'Please provide a valid sessionId.');
  }

  try {
    const conversationHistory = await getConversationHistory(sessionId);
    const { context: relevantContext } = await hybridSearch(question, conversationHistory);
    const prompt = await generatePrompt(question, relevantContext, conversationHistory);

    // Ensure messages alternate between user and assistant
    const formattedMessages = strictlyAlternateMessages(conversationHistory.messages, prompt);

    console.log('Formatted messages:', JSON.stringify(formattedMessages, null, 2)); // Debug log

    const response = await axios.post(ANTHROPIC_API_URL, {
      model: "claude-3-haiku-20240307",
      max_tokens: 500,
      messages: formattedMessages
    }, {
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
    });

    if (response.data?.content?.[0]?.text) {
      const rawAnswer: string = response.data.content[0].text.trim();
      const processedAnswer = postProcessAnswer(rawAnswer);

      // Update conversation history
      await updateConversationHistory(sessionId, { role: 'user', content: question });
      await updateConversationHistory(sessionId, { role: 'assistant', content: processedAnswer });

      return {
        answer: processedAnswer,
        contextRelevance: calculateContextRelevance(relevantContext, processedAnswer)
      };
    } else {
      throw new Error('Unexpected response structure from AI');
    }
  } catch (error) {
    console.error('Error in askAlfred function:', error);
    if (axios.isAxiosError(error) && error.response) {
      console.error('Anthropic API error response:', error.response.data);
      throw new functions.https.HttpsError('unavailable', `AI service error: ${error.response.data.error?.message || 'Unknown error'}`);
    }
    throw new functions.https.HttpsError('internal', 'Processing error. Please try again later.');
  }
});


function calculateContextRelevance(context: string, answer: string): string {
  // Implement a simple relevance calculation (e.g., based on keyword overlap)
  const contextKeywords = new Set(context.toLowerCase().split(/\W+/));
  const answerKeywords = new Set(answer.toLowerCase().split(/\W+/));
  const overlapCount = [...contextKeywords].filter(word => answerKeywords.has(word)).length;
  const relevancePercentage = (overlapCount / contextKeywords.size) * 100;

  if (relevancePercentage > 80) return "High";
  if (relevancePercentage > 50) return "Medium";
  return "Low";
}

function strictlyAlternateMessages(history: Message[], currentPrompt: string): Message[] {
  const formattedMessages: Message[] = [];

  // Always start with a user message
  if (history.length === 0 || history[0].role !== 'user') {
    formattedMessages.push({ role: 'user', content: 'Hello, I have a question.' });
  }

  // Process historical messages, ensuring strict alternation
  for (const message of history) {
    if (formattedMessages.length === 0 || message.role !== formattedMessages[formattedMessages.length - 1].role) {
      formattedMessages.push(message);
    }
  }

  // Ensure the last message before the new prompt is from the assistant
  if (formattedMessages.length % 2 === 1) {
    formattedMessages.push({ role: 'assistant', content: 'How can I assist you?' });
  }

  // Add the current prompt as a user message
  formattedMessages.push({ role: 'user', content: currentPrompt });

  return formattedMessages;
}



async function getConversationHistory(sessionId: string): Promise<ConversationHistory> {
  const doc = await admin.firestore().collection('conversations').doc(sessionId).get();
  if (!doc.exists) {
    return { messages: [] };
  }
  const data = doc.data();
  return {
    messages: (data?.messages || []).map((msg: Message) => ({
      role: msg.role,
      content: msg.content
    }))
  };
}



async function updateConversationHistory(sessionId: string, newMessage: Message): Promise<void> {
  const conversationRef = admin.firestore().collection('conversations').doc(sessionId);

  await admin.firestore().runTransaction(async (transaction) => {
    const doc = await transaction.get(conversationRef);
    let messages: Message[] = [];

    if (doc.exists) {
      messages = doc.data()?.messages || [];
    }

    // Ensure the conversation starts with a user message if it's empty
    if (messages.length === 0 && newMessage.role === 'assistant') {
      messages.push({ role: 'user', content: 'Hello, I have a question.' });
    }

    messages.push(newMessage);

    // Keep only the last 10 messages
    messages = messages.slice(-10);

    transaction.set(conversationRef, {
      messages: messages,
      lastUpdated: admin.firestore.FieldValue.serverTimestamp()
    }, { merge: true });
  });
}


export const clearConversationHistory = functions.https.onCall(async (data, context) => {
  const sessionId: string = data?.sessionId || 'default';

  try {
    await admin.firestore().collection('conversations').doc(sessionId).delete();
    return { success: true, message: 'Conversation history cleared' };
  } catch (error) {
    logger.error('Error clearing conversation history:', error);
    throw new functions.https.HttpsError('internal', 'Failed to clear conversation history. Please try again.');
  }
});




// Contextual questions
// Cloud Function (add to your existing functions file)
export const getContextualQuestions = functions.https.onCall(async (data, context) => {
  try {
    const prompt = `You are Alfred, an AI assistant specializing in Infrastructure Transparency and the Construction Sector Transparency Initiative (CoST). Your purpose is to help users understand and implement transparency in infrastructure projects.

Given this context, generate 4 engaging and relevant questions that users might want to ask about infrastructure transparency. Each question should be concise (no more than 6 words) and paired with an appropriate Bootstrap icon name.

Format your response as a JSON array of objects, each with 'text' and 'icon' properties. For example:
[
  { "text": "CoST impact on corruption?", "icon": "bi-shield-check" },
  ...
]

Focus on key areas such as:
1. Transparency in public contracting
2. Benefits of the CoST initiative
3. Best practices in infrastructure project management
4. Socio-economic impacts of transparent infrastructure investments

Generate the questions now:`;

    const response = await axios.post(ANTHROPIC_API_URL, {
      model: "claude-3-haiku-20240307",
      max_tokens: 400,
      messages: [{ role: "user", content: prompt }]
    }, {
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
    });

    if (response.data?.content?.[0]?.text) {
      const generatedContent = response.data.content[0].text.trim();
      const questions = JSON.parse(generatedContent);

      // Validate the structure of the generated questions
      if (Array.isArray(questions) && questions.length === 4 &&
          questions.every(q => typeof q.text === 'string' && typeof q.icon === 'string')) {
        return questions;
      } else {
        throw new Error('Invalid question format generated');
      }
    } else {
      throw new Error('Unexpected response structure from AI');
    }
  } catch (error) {
    console.error('Error generating contextual questions:', error);
    // Fallback to static questions if there's an error
    return [
      { text: "CoST initiative overview?", icon: "bi-info-circle" },
      { text: "Transparency impact on projects?", icon: "bi-graph-up" },
      { text: "Public engagement in CoST?", icon: "bi-people" },
      { text: "Implementing CoST standards?", icon: "bi-clipboard-check" }
    ];
  }
});












interface VectorizationState {
  totalItems: number;
  processedItems: number;
  continuationToken: string | null;
}


function sanitizeText(text: string): string {
  return text.replace(/[\uFFFD\uFFFE\uFFFF]/g, '')
      .replace(/[\u0000-\u001F\u007F-\u009F]/g, '')
      .normalize('NFKD')
      .replace(/[^\x20-\x7E]/g, '');
}

function sanitizeMetadata(metadata: Record<string, any>): Record<string, string | number | boolean | string[]> {
  const sanitized: Record<string, string | number | boolean | string[]> = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      sanitized[key] = value;
    } else if (Array.isArray(value) && value.every(item => typeof item === 'string')) {
      sanitized[key] = value;
    } else if (typeof value === 'object' && value !== null) {
      sanitized[key] = JSON.stringify(value);
    }
  }
  return sanitized;
}


async function processChunk(chunk: TextItem, index: any, indexDimension: number): Promise<void> {
  const sanitizedText = sanitizeText(chunk.text);
  if (sanitizedText.length === 0) {
    console.warn('Text chunk was empty after sanitization');
    return;
  }
  const embedding = await embeddings.embedQuery(sanitizedText);
  if (embedding.length !== indexDimension) {
    throw new Error(`Generated embedding dimension (${embedding.length}) does not match index dimension (${indexDimension})`);
  }
  const sanitizedMetadata = sanitizeMetadata(chunk.metadata);
  await index.upsert([{
    id: `doc-${Date.now()}-${chunk.metadata.startIndex || ''}`,
    values: embedding,
    metadata: {
      ...sanitizedMetadata,
      text: sanitizedText.slice(0, 1000),
      timestamp: new Date().toISOString(),
    },
  }]);
}
export const startVectorization = functions
    .runWith({
      timeoutSeconds: 540,
      memory: '2GB'
    })
    .https.onCall(async (data: { texts: TextItem[], state: VectorizationState }, context) => {
      try {
        functions.logger.info('Function started with data:', JSON.stringify(data, null, 2));

        if (!Array.isArray(data.texts) || data.texts.length === 0) {
          throw new functions.https.HttpsError('invalid-argument', 'The function must be called with a non-empty "texts" array.');
        }

        const index = pinecone.Index(PINECONE_INDEX_NAME);
        const indexDescription = await index.describeIndexStats();
        const indexDimension = indexDescription.dimension;

        if (!indexDimension) {
          throw new Error('Could not determine index dimension');
        }

        let state: VectorizationState = data.state;

        for (const chunk of data.texts) {
          await processChunk(chunk, index, indexDimension);
          state.processedItems++;

          // Update progress periodically (e.g., every 10 chunks)
          if (state.processedItems % 10 === 0) {
            await updateProgress(state.processedItems, state.totalItems);
          }
        }

        // Final progress update
        await updateProgress(state.processedItems, state.totalItems);

        return {
          success: true,
          message: `Processed all ${state.totalItems} items`,
          state: state
        };

      } catch (error) {
        functions.logger.error('Detailed error in startVectorization:', error);
        if (error instanceof functions.https.HttpsError) {
          throw error;
        } else if (error instanceof Error) {
          throw new functions.https.HttpsError('internal', `Vectorization failed: ${error.message}`);
        } else {
          throw new functions.https.HttpsError('internal', 'Vectorization failed: Unknown error');
        }
      }
    });
export const getVectorizationProgress = functions.https.onCall(async (data, context) => {
  const progressDoc = await admin.firestore().collection('vectorizationProgress').doc('current').get();
  if (!progressDoc.exists) {
    return { processedItems: 0, totalItems: 0, percentage: 0 };
  }
  return progressDoc.data();
});

export const processVectorizationBatch = functions.firestore
    .document('vectorizationTasks/{taskId}')
    .onCreate(async (snap, context) => {
      const task = snap.data() as { batchIndex: number; texts: TextItem[]; status: string };
      if (task.status !== 'pending') return;

      const index = pinecone.Index(PINECONE_INDEX_NAME);

      try {
        console.log(`Processing batch ${task.batchIndex}`);
        const texts = task.texts.map(item => item.text);
        const metadata = task.texts.map(item => item.metadata);

        const embeddings = await batchEmbedTexts(texts);

        const vectors = embeddings.map((embedding, i) => ({
          id: `doc-${Date.now()}-${task.batchIndex}-${i}`,
          values: embedding,
          metadata: { ...metadata[i], text: texts[i].slice(0, 1000) }
        }));

        await index.upsert(vectors);

        await snap.ref.update({ status: 'completed' });

        const completedTasks = await admin.firestore().collection('vectorizationTasks')
            .where('status', '==', 'completed')
            .get();

        const totalProcessed = completedTasks.size * BATCH_SIZE;

        await updateProgress(totalProcessed, task.texts.length);
        console.log(`Completed processing batch ${task.batchIndex}. Total processed: ${totalProcessed}`);

      } catch (error) {
        console.error(`Error processing batch ${task.batchIndex}:`, error);
        await snap.ref.update({ status: 'failed', error: JSON.stringify(error) });
      }
    });



export const clearVectorizationProgress = functions.https.onCall(async (data = {}, context) => {
  await admin.firestore().collection('vectorizationProgress').doc('current').delete();
  console.log('Vectorization progress cleared');
  return { success: true, message: 'Vectorization progress cleared' };
});

export const provideFeedback = https.onCall(async (data, context) => {
  const { questionId, isHelpful, comment } = data;

  if (typeof questionId !== 'string' || typeof isHelpful !== 'boolean') {
    throw new https.HttpsError('invalid-argument', 'Invalid feedback data provided.');
  }

  try {
    await admin.firestore().collection('feedback').add({
      questionId,
      isHelpful,
      comment,
      userId: context.auth?.uid || 'anonymous',
      timestamp: admin.firestore.FieldValue.serverTimestamp()
    });
    return { success: true };
  } catch (error) {
    logger.error('Error storing feedback:', error);
    throw new https.HttpsError('internal', 'Failed to store feedback. Please try again.');
  }
});

