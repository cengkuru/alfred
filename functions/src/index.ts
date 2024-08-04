import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import {Pinecone, QueryResponse} from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Configuration, OpenAIApi } from 'openai';
import axios from 'axios';
import {https, logger} from "firebase-functions";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as natural from 'natural';
import * as zlib from 'zlib';

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

interface ProcessedChunk {
  text: string;
  metadata: Record<string, any>;
  summaries: {
    brief: string;
    medium: string;
    detailed: string;
  };
}

// Type definitions
interface SearchResult {
  text: string;
  score: number;
  documentId: string;
  chunkIndex: number;
  summaries: {
    brief: string;
    medium: string;
    detailed: string;
  };
}

interface PineconeMetadata {
  text: string;
  documentId: string;
  chunkIndex: number;
  summaries: string[]; // Changed to string[] to match the actual structure
}

interface ConversationHistory {
  messages: { role: 'user' | 'assistant'; content: string }[];
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




const ANTHROPIC_API_KEY = functions.config().anthropic.apikey;
const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
const CHUNK_SIZE = 4000; // Adjust based on Claude's token limit


async function performKeywordSearch(query: string): Promise<SearchResult[]> {
  console.log('Starting keyword search for query:', query);

  try {
    const allDocuments = await admin.firestore().collection('documents').get();

    const documents = allDocuments.docs.map(doc => {
      const data = doc.data();
      return {
        id: doc.id,
        text: data.text ? decompressData(Buffer.from(data.text, 'base64')) : '',
        summaries: {
          brief: data.summaries?.brief ? decompressData(Buffer.from(data.summaries.brief, 'base64')) : '',
          medium: data.summaries?.medium ? decompressData(Buffer.from(data.summaries.medium, 'base64')) : '',
          detailed: data.summaries?.detailed ? decompressData(Buffer.from(data.summaries.detailed, 'base64')) : ''
        }
      };
    });

    console.log(`Retrieved ${documents.length} documents for keyword search`);

    const keywordResults = keywordSearch(query, documents.map(doc => doc.text));

    console.log(`Keyword search returned ${keywordResults.length} results`);

    return keywordResults.map(result => {
      const document = documents.find(doc => doc.text === result);
      if (!document) {
        console.warn('No matching document found for keyword result');
        return null;
      }
      return {
        text: result,
        score: 1, // Default score for keyword results
        documentId: document.id,
        chunkIndex: 0, // Assuming each document is a single chunk for keyword search
        summaries: document.summaries
      };
    }).filter((result): result is SearchResult => result !== null);
  } catch (error) {
    console.error('Error in performKeywordSearch:', error);
    return [];
  }
}

// Helper function for keyword search
function keywordSearch(query: string, documents: string[]): string[] {
  console.log(`Performing keyword search with query: "${query}" on ${documents.length} documents`);

  const validDocuments = documents.filter(doc => typeof doc === 'string' && doc.trim() !== '');

  if (validDocuments.length === 0) {
    console.warn('No valid documents for keyword search');
    return [];
  }

  const TfIdf = natural.TfIdf;
  const tfidf = new TfIdf();

  validDocuments.forEach(doc => tfidf.addDocument(doc));

  const tokenizer = new natural.WordTokenizer();
  const queryTokens = tokenizer.tokenize(query.toLowerCase());

  const scores = validDocuments.map((_, index) => {
    return queryTokens.reduce((score, token) => {
      return score + tfidf.tfidf(token, index);
    }, 0);
  });

  const results = scores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map(item => validDocuments[item.index]);

  console.log(`Keyword search returned ${results.length} results`);
  return results;
}
// Updated hybridSearch function
export async function hybridSearch(question: string, conversationHistory: ConversationHistory): Promise<SearchResult[]> {
  console.log('Starting hybrid search for question:', question);

  try {
    // Expand the query
    console.log('Expanding query');
    const expandedQuery = await expandQuery(question, conversationHistory);
    console.log('Expanded query:', expandedQuery);

    // Perform vector search with expanded query
    console.log('Performing vector search');
    const vectorResults = await performVectorSearch(expandedQuery);
    console.log('Vector search results:', vectorResults.length);

    // Perform keyword search
    console.log('Performing keyword search');
    const keywordResults = await performKeywordSearch(expandedQuery);
    console.log('Keyword search results:', keywordResults.length);

    // Rerank results
    console.log('Reranking results');
    const rerankedResults = rerank(vectorResults, keywordResults, expandedQuery);
    console.log('Reranked results:', rerankedResults.length);

    return rerankedResults;
  } catch (error) {
    console.error('Error in hybrid search:', error);
    return [];
  }
}
async function expandQuery(question: string, conversationHistory: ConversationHistory): Promise<string> {
  const recentMessages = conversationHistory.messages.slice(-3).map(m => m.content).join(' ');
  const expandedQuery = `${recentMessages} ${question}`;

  try {
    const response = await axios.post(ANTHROPIC_API_URL, {
      model: "claude-3-haiku-20240307",
      max_tokens: 100,
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
  } catch (error) {
    console.error('Error expanding query:', error);
    return expandedQuery;
  }
}
// Improved keyword search function using TF-IDF

function rerank(vectorResults: SearchResult[], keywordResults: SearchResult[], query: string): SearchResult[] {
  const allResults = [...vectorResults, ...keywordResults];

  return allResults.map(result => {
    const keywordScore = keywordResults.some(r => r.documentId === result.documentId) ? 1 : 0;
    const combinedScore = (result.score || 0) + keywordScore;
    return { ...result, score: combinedScore };
  }).sort((a, b) => b.score - a.score).slice(0, 10);
}



// Improved: Prompt generation function
function generatePrompt(question: string, context: string, conversationHistory: ConversationHistory): string {
  const recentConversation = conversationHistory.messages.slice(-3).map(m => `${m.role}: ${m.content}`).join('\n');

  return `
Context:
${context}

Recent Conversation:
${recentConversation}

You are Alfred, an AI assistant specializing in Infrastructure Transparency and the Construction Sector Transparency Initiative (CoST). Your knowledge spans infrastructure project lifecycles, transparency in public contracting, and socio-economic impacts of infrastructure investments.

Current Question: ${question}

Instructions:
1. Analyze the question, context, and recent conversation to provide a coherent and relevant response.
2. If this is a follow-up question, make sure to reference and build upon the previous parts of the conversation.
3. Provide a comprehensive but concise answer that directly addresses the user's query.
4. Use markdown formatting to enhance readability, including headers and emphasis where appropriate.
5. Include relevant statistics or examples from the provided context when applicable.
6. If the information provided is insufficient, acknowledge this and suggest potential avenues for further research.
7. Aim for a response length of about 150-200 words to ensure it fits within the token limit.

Begin your response now:`;
}

function truncatePrompt(prompt: string, maxTokens: number): string {
  const tokens = prompt.split(/\s+/);
  if (tokens.length <= maxTokens) return prompt;
  return tokens.slice(0, maxTokens).join(' ') + '...';
}


export const askAlfred = functions.runWith({
  timeoutSeconds: 300,
  memory: '1GB'
}).https.onCall(async (data, context) => {
  const { question, sessionId } = data;

  console.log('askAlfred called with:', { question, sessionId });

  if (!question || typeof question !== 'string' || question.trim() === '') {
    console.error('Invalid question provided:', question);
    throw new functions.https.HttpsError('invalid-argument', 'Please provide a valid non-empty question.');
  }

  if (!sessionId || typeof sessionId !== 'string' || sessionId.trim() === '') {
    console.error('Invalid sessionId provided:', sessionId);
    throw new functions.https.HttpsError('invalid-argument', 'Please provide a valid non-empty sessionId.');
  }

  try {
    console.log('Getting conversation history for sessionId:', sessionId);
    const conversationHistory = await getConversationHistory(sessionId);

    console.log('Performing hybrid search');
    const searchResults = await hybridSearch(question, conversationHistory);
    console.log('Hybrid search results:', JSON.stringify(searchResults, null, 2));

    console.log('Performing vectorized similarity search');
    const vectorSearchResults = await performVectorSearch(question);
    console.log('Vector search results:', JSON.stringify(vectorSearchResults, null, 2));

    console.log('Combining and ranking results');
    const combinedResults = combineAndRankResults(searchResults, vectorSearchResults);
    console.log('Combined results:', JSON.stringify(combinedResults, null, 2));

    console.log('Generating prompt');
    const context = combinedResults.map(result => result.summaries.medium).join('\n\n');
    const prompt = generatePrompt(question, context, conversationHistory);
    const truncatedPrompt = truncatePrompt(prompt, 4000);
    console.log('Generated prompt:', truncatedPrompt);

    console.log('Sending request to Anthropic API');
    const response = await axios.post(ANTHROPIC_API_URL, {
      model: "claude-3-haiku-20240307",
      max_tokens: 500,
      messages: [{ role: "user", content: truncatedPrompt }]
    }, {
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
    });

    if (response.data?.content?.[0]?.text) {
      const rawAnswer: string = response.data.content[0].text.trim();
      console.log('Raw answer from Anthropic:', rawAnswer);

      const processedAnswer = postProcessAnswer(rawAnswer);
      console.log('Processed answer:', processedAnswer);

      console.log('Updating conversation history');
      await updateConversationHistory(sessionId, { role: 'user', content: question });
      await updateConversationHistory(sessionId, { role: 'assistant', content: processedAnswer });

      console.log('Calculating context relevance');
      const contextRelevance = calculateContextRelevance(context, processedAnswer);

      console.log('Returning answer');
      return {
        answer: processedAnswer,
        contextRelevance: contextRelevance
      };
    } else {
      console.error('Unexpected response structure from Anthropic API');
      throw new Error('Unexpected response structure from AI');
    }
  } catch (error: unknown) {
    console.error('Detailed error in askAlfred:', error);

    if (axios.isAxiosError(error)) {
      if (error.response) {
        console.error('Error response data:', error.response.data);
        console.error('Error response status:', error.response.status);
        console.error('Error response headers:', error.response.headers);
      } else if (error.request) {
        console.error('Error request:', error.request);
      } else {
        console.error('Error message:', error.message);
      }
    } else if (error instanceof Error) {
      console.error('Error message:', error.message);
      console.error('Error stack:', error.stack);
    } else {
      console.error('Unknown error:', error);
    }

    throw new functions.https.HttpsError('internal', 'Processing error. Please try again later.');
  }
});

// Updated performVectorSearch function
async function performVectorSearch(question: string): Promise<SearchResult[]> {
  try {
    console.log('Starting vector search for question:', question);
    const queryEmbedding = await embeddings.embedQuery(question);
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    const searchResults = await index.query({
      vector: queryEmbedding,
      topK: 10,
      includeMetadata: true
    });

    console.log('Raw search results:', JSON.stringify(searchResults, null, 2));

    return searchResults.matches
        .filter((match): match is QueryResponse['matches'][number] & { metadata: PineconeMetadata } => {
          if (!match.metadata) {
            console.warn('Match missing metadata:', match);
            return false;
          }
          const { text, documentId, chunkIndex, summaries } = match.metadata;
          const isValid =
              (typeof text === 'string' || text instanceof Buffer) &&
              typeof documentId === 'string' &&
              typeof chunkIndex === 'number' &&
              Array.isArray(summaries) &&
              summaries.length === 3;
          if (!isValid) {
            console.warn('Invalid metadata structure:', match.metadata);
          }
          return isValid;
        })
        .map(match => {
          console.log('Processing match:', JSON.stringify(match, null, 2));
          const { text, documentId, chunkIndex, summaries } = match.metadata;
          return {
            text: decompressData(text),
            score: match.score ?? 0,
            documentId,
            chunkIndex,
            summaries: {
              brief: decompressData(summaries[0]),
              medium: decompressData(summaries[1]),
              detailed: decompressData(summaries[2])
            }
          };
        });
  } catch (error) {
    console.error('Error in performVectorSearch:', error);
    return [];
  }
}

function combineAndRankResults(hybridResults: SearchResult[], vectorResults: SearchResult[]): SearchResult[] {
  const combinedResults = [...hybridResults, ...vectorResults];

  const uniqueResults = Array.from(new Map(combinedResults.map(item =>
      [`${item.documentId}_${item.chunkIndex}`, item]
  )).values());

  return uniqueResults.sort((a, b) => b.score - a.score).slice(0, 10);
}
function postProcessAnswer(rawAnswer: string): string {
  let processedAnswer = rawAnswer.trim();
  processedAnswer = processedAnswer.replace(/^Sure,?\s+/, '');
  processedAnswer = processedAnswer.charAt(0).toUpperCase() + processedAnswer.slice(1);
  return processedAnswer;
}

function calculateContextRelevance(context: string, answer: string): string {
  const contextKeywords = new Set(context.toLowerCase().split(/\W+/));
  const answerKeywords = new Set(answer.toLowerCase().split(/\W+/));
  const overlapCount = [...contextKeywords].filter(word => answerKeywords.has(word)).length;
  const relevancePercentage = (overlapCount / contextKeywords.size) * 100;

  if (relevancePercentage > 80) return "High";
  if (relevancePercentage > 50) return "Medium";
  return "Low";
}




async function getConversationHistory(sessionId: string): Promise<ConversationHistory> {
  if (!sessionId) {
    throw new Error('Invalid sessionId provided to getConversationHistory');
  }
  const doc = await admin.firestore().collection('conversations').doc(sessionId).get();
  if (!doc.exists) {
    return { messages: [] };
  }
  const data = doc.data();
  return {
    messages: (data?.messages || []).map((msg: any) => ({
      role: msg.role,
      content: msg.content
    }))
  };
}

async function updateConversationHistory(sessionId: string, newMessage: Message): Promise<void> {
  if (!sessionId) {
    throw new Error('Invalid sessionId provided to updateConversationHistory');
  }
  const conversationRef = admin.firestore().collection('conversations').doc(sessionId);

  await admin.firestore().runTransaction(async (transaction) => {
    const doc = await transaction.get(conversationRef);
    let messages = doc.exists ? (doc.data()?.messages || []) : [];
    messages.push(newMessage);
    messages = messages.slice(-10); // Keep only the last 10 messages

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



















// Intelligent Chunking
async function intelligentChunking(text: string): Promise<string[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: 200,
    separators: ['\n\n', '\n', '. ', ' ', '']
  });
  return splitter.splitText(text);
}

// Metadata Extraction
function extractMetadata(text: string): Record<string, any> {
  const metadata: Record<string, any> = {};

  // Extract dates
  const dateRegex = /\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b/g;
  metadata.dates = text.match(dateRegex) || [];

  // Extract monetary values (assuming USD)
  const moneyRegex = /\$\s?[0-9,]+\.?[0-9]*/g;
  metadata.monetaryValues = text.match(moneyRegex) || [];

  // Extract potential vendor names (capitalized words)
  const vendorRegex = /\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*(?:Inc\.|LLC|Ltd\.|Corp\.)\b/g;
  metadata.potentialVendors = text.match(vendorRegex) || [];

  return metadata;
}

// Relevance Filtering
function isRelevantChunk(chunk: string): boolean {
  const tokenizer = new natural.WordTokenizer();
  const tokens = tokenizer.tokenize(chunk);
  const uniqueTokens = new Set(tokens.map(t => t.toLowerCase()));

  // Consider a chunk relevant if it has a certain number of unique words
  return uniqueTokens.size > 20; // Adjust this threshold as needed
}

async function progressiveSummarization(text: string): Promise<{ brief: string; medium: string; detailed: string }> {
  const summaries = {
    brief: '',
    medium: '',
    detailed: ''
  };

  for (const level of ['detailed', 'medium', 'brief']) {
    const response = await axios.post(ANTHROPIC_API_URL, {
      model: "claude-3-haiku-20240307",
      max_tokens: 500,
      messages: [
        { role: "user", content: `Provide a ${level} summary of the following text:\n\n${text}` }
      ]
    }, {
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
    });

    summaries[level as keyof typeof summaries] = response.data.content[0].text.trim();
  }

  return summaries;
}


async function batchProcess<T, R>(items: T[], batchSize: number, processFn: (batch: T[]) => Promise<R[]>): Promise<R[]> {
  const results: R[] = [];
  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    const batchResults = await processFn(batch);
    results.push(...batchResults);
  }
  return results;
}

// Compress data for Firebase storage
function compressData(data: string): Buffer {
  return zlib.gzipSync(Buffer.from(data));
}

// Decompress data from Firebase storage
// Updated decompressData function
function decompressData(data: string | Buffer | undefined | null): string {
  if (!data) {
    console.warn('Attempted to decompress undefined or null data');
    return '';
  }
  try {
    if (typeof data === 'string') {
      // If it's a string, assume it's already base64 encoded
      return zlib.gunzipSync(Buffer.from(data, 'base64')).toString();
    } else {
      // If it's a Buffer, decompress directly
      return zlib.gunzipSync(data).toString();
    }
  } catch (error) {
    console.error('Error decompressing data:', error);
    return '';
  }
}


// Updated processDocument function
export const processDocument = functions.runWith({
  timeoutSeconds: 540,
  memory: '2GB'
}).https.onCall(async (data, context) => {
  const { text, documentId } = data;

  if (!text || typeof text !== 'string') {
    throw new functions.https.HttpsError('invalid-argument', 'The function must be called with a non-empty "text" string.');
  }

  try {
    const chunks = await intelligentChunking(text);
    const relevantChunks = chunks.filter(isRelevantChunk);

    const processedChunks: ProcessedChunk[] = [];

    // Batch process chunks
    const batchSize = 5;
    await batchProcess(relevantChunks, batchSize, async (batch) => {
      const batchResults = await Promise.all(batch.map(async (chunk) => {
        const metadata = extractMetadata(chunk);
        const summaries = await progressiveSummarization(chunk);
        return { text: chunk, metadata, summaries };
      }));
      processedChunks.push(...batchResults);

      // Update progress
      await admin.firestore().collection('processingProgress').doc(documentId).set({
        totalChunks: relevantChunks.length,
        processedChunks: processedChunks.length,
        percentage: (processedChunks.length / relevantChunks.length) * 100
      }, { merge: true });

      return batchResults;
    });

    // Store processed chunks in Firebase
    const compressedChunks = processedChunks.map(chunk => ({
      ...chunk,
      text: compressData(chunk.text),
      summaries: {
        brief: compressData(chunk.summaries.brief),
        medium: compressData(chunk.summaries.medium),
        detailed: compressData(chunk.summaries.detailed)
      }
    }));

    await admin.firestore().collection('documents').doc(documentId).set({
      chunks: compressedChunks,
      metadata: extractMetadata(text),
      timestamp: admin.firestore.FieldValue.serverTimestamp()
    });

    // Generate and store embeddings in Pinecone
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    const embeddingsArray = await batchProcess(processedChunks, batchSize, async (batch) => {
      const texts = batch.map(chunk => chunk.summaries.medium);
      return embeddings.embedDocuments(texts);
    });

    const vectors = processedChunks.map((chunk, i) => ({
      id: `${documentId}_chunk_${i}`,
      values: embeddingsArray[i],
      metadata: {
        ...chunk.metadata,
        documentId,
        chunkIndex: i
      }
    }));

    await index.upsert(vectors);

    return { success: true, message: `Processed ${processedChunks.length} chunks of the document` };
  } catch (error) {
    console.error('Error in processDocument:', error);
    throw new functions.https.HttpsError('internal', 'Document processing failed. Please try again.');
  }
});

