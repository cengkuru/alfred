import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import {Pinecone, QueryResponse} from '@pinecone-database/pinecone';
import { IndexStatsDescription } from '@pinecone-database/pinecone';
import { PineconeRecord, RecordMetadata } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Configuration, OpenAIApi } from 'openai';
import axios from 'axios';
import {https, logger} from "firebase-functions";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as natural from 'natural';
import * as zlib from 'zlib';
import {firestore} from "firebase-admin";
import DocumentData = firestore.DocumentData;
import {VectorizationStatus, VectorizationTask} from "./models/vectorization-task.model";

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

const CHUNK_SIZE = 1000; // Adjust based on your needs
const BATCH_SIZE = 5; // Reduced batch size
const MAX_RETRIES = 5;
const INITIAL_BACKOFF = 1000; // 1 second

interface ChunkData {
  text: string;
  metadata: any;
}

interface DocumentWithSummaries extends DocumentData {
  text: string;
  summaries?: {
    brief: string;
    medium: string;
    detailed: string;
  };
}



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


async function performKeywordSearch(expandedQuery: string): Promise<SearchResult[]> {
  console.log('Starting improved keyword search for query:', expandedQuery);

  try {
    const results: SearchResult[] = [];

    // 1. Search in documents collection
    const documentsSnapshot = await admin.firestore()
        .collection('documents')
        .where('processed', '==', true)
        .get();

    // 2. Search in vectorizationTasks collection
    const tasksSnapshot = await admin.firestore()
        .collection('vectorizationTasks')
        .where('status', '==', VectorizationStatus.COMPLETED)
        .get();

    // Combine both document sources
    const allDocuments = [
      ...documentsSnapshot.docs.map(doc => ({ id: doc.id, data: doc.data() as DocumentWithSummaries, type: 'document' as const })),
      ...tasksSnapshot.docs.map(doc => ({ id: doc.id, data: doc.data() as VectorizationTask, type: 'task' as const }))
    ];

    // Create TF-IDF model
    const TfIdf = natural.TfIdf;
    const tfidf = new TfIdf();

    // Add documents to TF-IDF model
    allDocuments.forEach((doc, index) => {
      let content = '';
      if (doc.type === 'document') {
        content = [
          decompressData(doc.data.text),
          decompressData(doc.data.summaries?.brief),
          decompressData(doc.data.summaries?.medium),
          decompressData(doc.data.summaries?.detailed)
        ].filter(Boolean).join(' ');
      } else if (doc.type === 'task') {
        content = doc.data.texts.join(' ') + JSON.stringify(doc.data.metadata);
      }
      tfidf.addDocument(content);
    });

    // Perform search
    const tokenizer = new natural.WordTokenizer();
    const queryTokens = tokenizer.tokenize(expandedQuery.toLowerCase());

    const scores = allDocuments.map((_, index) => {
      return queryTokens.reduce((score, token) => score + tfidf.tfidf(token, index), 0);
    });

    // Sort and select top results
    const topResults = scores
        .map((score, index) => ({ score, index }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 10);  // Adjust number of results as needed

    // Process results
    for (const result of topResults) {
      const doc = allDocuments[result.index];
      let searchResult: SearchResult;

      if (doc.type === 'document') {
        searchResult = {
          text: decompressData(doc.data.text),
          score: result.score,
          documentId: doc.id,
          chunkIndex: 0,
          summaries: {
            brief: decompressData(doc.data.summaries?.brief) || '',
            medium: decompressData(doc.data.summaries?.medium) || '',
            detailed: decompressData(doc.data.summaries?.detailed) || ''
          }
        };
      } else {
        // For vectorizationTasks, we'll use the first text as the main content
        searchResult = {
          text: doc.data.texts[0] || '',
          score: result.score,
          documentId: doc.id,
          chunkIndex: doc.data.batchIndex || 0,
          summaries: {
            brief: JSON.stringify(doc.data.metadata),
            medium: doc.data.texts.join(' ').substring(0, 500),
            detailed: doc.data.texts.join(' ')
          }
        };
      }

      results.push(searchResult);
    }

    return results;
  } catch (error) {
    console.error('Error in improved keyword search:', error);
    return [];
  }
}

// Helper function for keyword search

// Updated hybridSearch function
export async function hybridSearch(query: string, conversationHistory: ConversationHistory): Promise<SearchResult[]> {
  console.log('Starting improved hybrid search for query:', query);

  try {
    // Perform vector search
    console.log('Performing vector search');
    const vectorResults = await performVectorSearch(query);
    console.log('Vector search results:', vectorResults.length);

    // Perform keyword search using the new performKeywordSearch function
    console.log('Performing keyword search');
    const keywordResults = await performKeywordSearch(query);
    console.log('Keyword search results:', keywordResults.length);

    // Combine and rank results
    console.log('Combining and ranking results');
    const combinedResults = combineAndRankResults(vectorResults, keywordResults);
    console.log('Combined results:', combinedResults.length);

    // If no results, fallback to semantic search
    if (combinedResults.length === 0) {
      console.log('No results found, falling back to semantic search');
      const semanticResults = await performSemanticSearch(query);
      console.log('Semantic search results:', semanticResults.length);
      return semanticResults;
    }

    return combinedResults;
  } catch (error) {
    console.error('Error in improved hybrid search:', error);
    return [];
  }
}

async function expandQuery(question: string, conversationHistory: ConversationHistory): Promise<string> {
  const recentMessages = conversationHistory.messages.slice(-5).map(m => m.content).join(' ');
  const expandedQuery = `${recentMessages} ${question}`;

  try {
    const response = await axios.post(ANTHROPIC_API_URL, {
      model: "claude-3-haiku-20240307",
      max_tokens: 100,
      messages: [{
        role: "user",
        content: `You are an AI assistant specializing in Infrastructure Transparency and the Construction Sector Transparency Initiative (CoST). Given the conversation context and question, generate an expanded search query to find relevant information about infrastructure transparency, public contracting, and CoST initiatives:
        
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

async function performSemanticSearch(query: string): Promise<SearchResult[]> {
  try {
    const queryEmbedding = await embeddings.embedQuery(query);
    const index = pinecone.Index(PINECONE_INDEX_NAME);
    const searchResults = await index.query({
      vector: queryEmbedding,
      topK: 20,
      includeMetadata: true
    });

    return searchResults.matches
        .filter((match): match is QueryResponse['matches'][number] & { metadata: PineconeMetadata } => !!match.metadata)
        .map(match => ({
          text: decompressData(match.metadata.text),
          score: match.score ?? 0,
          documentId: match.metadata.documentId,
          chunkIndex: match.metadata.chunkIndex,
          summaries: {
            brief: decompressData(match.metadata.summaries[0]),
            medium: decompressData(match.metadata.summaries[1]),
            detailed: decompressData(match.metadata.summaries[2])
          }
        }));
  } catch (error) {
    console.error('Error in semantic search:', error);
    return [];
  }
}

// Improved: Prompt generation function
function generatePrompt(question: string, searchResults: SearchResult[], conversationHistory: ConversationHistory): string {
  const relevantContext = searchResults
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
      .map(result => `${result.summaries.detailed}\n\nSource: ${result.documentId}, Chunk: ${result.chunkIndex}`)
      .join('\n\n');

  const recentConversation = conversationHistory.messages.slice(-5).map(m => `${m.role}: ${m.content}`).join('\n');

  return `
Context:
${relevantContext}

Recent Conversation:
${recentConversation}

You are Alfred, an AI assistant specializing in Infrastructure Transparency and the Construction Sector Transparency Initiative (CoST). Your knowledge spans infrastructure project lifecycles, transparency in public contracting, and socio-economic impacts of infrastructure investments.

Current Question: ${question}

Instructions:
1. Analyze the question, context, and recent conversation to provide a coherent and relevant response.
2. If this is a follow-up question, make sure to reference and build upon the previous parts of the conversation.
3. Provide a comprehensive answer that directly addresses the user's query, using the provided context.
4. If the exact information is not available in the context, use your general knowledge to provide a relevant response, but clearly indicate when you're doing so.
5. Use markdown formatting to enhance readability, including headers and emphasis where appropriate.
6. Include relevant statistics, examples, or quotes from the provided context when applicable.
7. If the information provided is insufficient, acknowledge this and suggest potential avenues for further research or inquiry.
8. Aim for a detailed response of about 200-300 words to ensure thoroughness.

Begin your response now:`;
}



function truncatePrompt(prompt: string, maxTokens: number): string {
  const tokens = prompt.split(/\s+/);
  if (tokens.length <= maxTokens) return prompt;
  return tokens.slice(0, maxTokens).join(' ') + '...';
}

async function classifyQuestionType(question: string): Promise<string> {
  try {
    const response = await axios.post(ANTHROPIC_API_URL, {
      model: "claude-3-haiku-20240307",
      max_tokens: 50,
      messages: [{
        role: "user",
        content: `Classify the following question into one of these types: factual, definitional, procedural, or analytical.
        
        Question: ${question}
        
        Classification:`
      }]
    }, {
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
    });

    return response.data?.content?.[0]?.text.trim().toLowerCase() || 'unknown';
  } catch (error) {
    console.error('Error classifying question:', error);
    return 'unknown';
  }
}
export const askAlfred = functions.runWith({
  timeoutSeconds: 500,
  memory: '1GB'
}).https.onCall(async (data, context) => {
  console.log('askAlfred function called with data:', JSON.stringify(data));

  const { question, sessionId } = data;

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
    console.log('Conversation history retrieved:', JSON.stringify(conversationHistory));

    console.log('Classifying question type');
    const questionType = await classifyQuestionType(question);
    console.log('Question classified as:', questionType);

    console.log('Expanding query');
    const expandedQuery = await expandQuery(question, conversationHistory);
    console.log('Query expanded to:', expandedQuery);

    console.log('Performing hybrid search');
    const hybridResults = await hybridSearch(expandedQuery, conversationHistory);
    console.log('Hybrid search returned results:', hybridResults.length);

    console.log('Generating prompt');
    const prompt = generatePrompt(question, hybridResults, conversationHistory);
    const truncatedPrompt = truncatePrompt(prompt, 4000);
    console.log('Prompt generated and truncated to length:', truncatedPrompt.length);

    console.log('Sending request to Anthropic API');
    const response = await axios.post(ANTHROPIC_API_URL, {
      model: "claude-3-haiku-20240307",
      max_tokens: 1000,
      system: `You are answering a ${questionType} question about infrastructure transparency and the CoST initiative. Use the provided context to give a detailed and accurate answer. If the information is not directly available, use your knowledge to provide a relevant response.`,
      messages: [
        { role: "user", content: truncatedPrompt }
      ]
    }, {
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
    });

    console.log('Received response from Anthropic API');

    if (response.data?.content?.[0]?.text) {
      const rawAnswer: string = response.data.content[0].text.trim();
      console.log('Raw answer length:', rawAnswer.length);

      const processedAnswer = postProcessAnswer(rawAnswer);
      console.log('Processed answer length:', processedAnswer.length);

      console.log('Updating conversation history');
      await updateConversationHistory(sessionId, { role: 'user', content: question });
      await updateConversationHistory(sessionId, { role: 'assistant', content: processedAnswer });

      console.log('Calculating context relevance');
      const contextRelevance = calculateContextRelevance(hybridResults.map(r => r.text).join(' '), processedAnswer);

      console.log('Returning answer');
      return {
        answer: processedAnswer,
        contextRelevance: contextRelevance,
        questionType: questionType
      };
    } else {
      console.error('Unexpected response structure from Anthropic API:', JSON.stringify(response.data));
      throw new Error('Unexpected response structure from AI');
    }
  } catch (error: unknown) {
    console.error('Detailed error in askAlfred:', error);

    if (axios.isAxiosError(error)) {
      if (error.response) {
        console.error('Axios error response data:', error.response.data);
        console.error('Axios error response status:', error.response.status);
        console.error('Axios error response headers:', error.response.headers);
        throw new functions.https.HttpsError('internal', `API error: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
      } else if (error.request) {
        console.error('Axios error request:', error.request);
        throw new functions.https.HttpsError('internal', 'API request failed. Please check your network connection.');
      } else {
        console.error('Axios error message:', error.message);
        throw new functions.https.HttpsError('internal', `API error: ${error.message}`);
      }
    } else if (error instanceof Error) {
      console.error('Error message:', error.message);
      console.error('Error stack:', error.stack);
      throw new functions.https.HttpsError('internal', `Processing error: ${error.message}`);
    } else {
      console.error('Unknown error:', error);
      throw new functions.https.HttpsError('internal', 'An unknown error occurred. Please try again later.');
    }
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

// Updated combineAndRankResults function
function combineAndRankResults(vectorResults: SearchResult[], keywordResults: SearchResult[]): SearchResult[] {
  const allResults = [...vectorResults, ...keywordResults];

  // Remove duplicates
  const uniqueResults = Array.from(new Map(allResults.map(item =>
      [`${item.documentId}_${item.chunkIndex}`, item]
  )).values());

  // Rank results based on a combination of vector similarity and keyword matching
  const rankedResults = uniqueResults.map(result => {
    const vectorScore = vectorResults.find(r => r.documentId === result.documentId && r.chunkIndex === result.chunkIndex)?.score || 0;
    const keywordScore = keywordResults.includes(result) ? 1 : 0;

    return {
      ...result,
      score: vectorScore * 0.6 + keywordScore * 0.4
    };
  });

  // Sort by combined score and return top results
  return rankedResults.sort((a, b) => b.score - a.score).slice(0, 10);
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


export const getContextualQuestions = functions.https.onCall(async (data, context) => {
  try {
    const db = admin.firestore();
    let contextualContent = '';

    // Step 1: Fetch summaries
    const summariesSnapshot = await db.collection('summaries').limit(3).orderBy('createdAt', 'desc').get();
    summariesSnapshot.forEach(doc => {
      const summaryData = doc.data();
      contextualContent += `Summary: ${summaryData.content}\n\n`;
    });

    // Step 2: Fetch recent documents
    const documentsSnapshot = await db.collection('documents').limit(3).orderBy('createdAt', 'desc').get();
    documentsSnapshot.forEach(doc => {
      const documentData = doc.data();
      contextualContent += `Document: ${documentData.title}\nExcerpt: ${documentData.excerpt}\n\n`;
    });

    // Step 3: Fetch completed vectorization tasks
    const tasksSnapshot = await db.collection('vectorizationTasks')
        .where('status', '==', 'COMPLETED')
        .limit(3)
        .orderBy('updatedAt', 'desc')
        .get();
    tasksSnapshot.forEach(doc => {
      const taskData = doc.data();
      contextualContent += `Vectorized Content: ${taskData.texts.slice(0, 3).join(' ')}\n`;
      contextualContent += `Intent: ${taskData.metadata.intent}, Sentiment: ${taskData.metadata.sentiment}\n\n`;
    });

    // Step 4: Generate AI prompt with Firestore context
    const prompt = `You are Alfred, an AI assistant specializing in Infrastructure Transparency and the Construction Sector Transparency Initiative (CoST). Your purpose is to help users understand and implement transparency in infrastructure projects. 

Here's some context from our recent summaries, documents, and vectorized content:
${contextualContent}

Given this context, generate 4 engaging and relevant questions that users might want to ask about infrastructure transparency, focusing on our recent data. Each question should be concise (no more than 10 words) and paired with an appropriate Bootstrap icon name.

Format your response as a JSON array of objects, each with 'text' and 'icon' properties. For example:
[
  { "text": "How does recent project X enhance contracting transparency?", "icon": "bi-shield-check" },
  ...
]

Focus on key areas such as:
1. Transparency in public contracting related to our recent documents
2. Benefits of the CoST initiative as seen in our summaries
3. Best practices in infrastructure project management from our vectorized content
4. Socio-economic impacts of our transparent infrastructure investments

Generate the questions now:`;

    // Step 5: Call Anthropic API
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
      { text: "Recent transparency improvements in contracting?", icon: "bi-shield-check" },
      { text: "CoST benefits highlighted in latest summary?", icon: "bi-file-earmark-text" },
      { text: "Best practices from vectorized project data?", icon: "bi-graph-up" },
      { text: "Socio-economic impact of recent initiatives?", icon: "bi-people" }
    ];
  }
});


export const startVectorization = functions
    .runWith({
      timeoutSeconds: 540,
      memory: '2GB'
    })
    .https.onCall(async (data: { documentId: string, text: string }, context) => {
      const { documentId, text } = data;

      try {
        // Step 1: Create document chunks
        const splitter = new RecursiveCharacterTextSplitter({
          chunkSize: CHUNK_SIZE,
          chunkOverlap: 200,
        });
        const chunks = await splitter.splitText(text);

        // Step 2: Create a vectorization task
        const taskRef = await admin.firestore().collection('vectorizationTasks').add({
          documentId,
          status: 'processing',
          progress: 0,
          createdAt: admin.firestore.FieldValue.serverTimestamp(),
          updatedAt: admin.firestore.FieldValue.serverTimestamp(),
          totalChunks: chunks.length,
          processedChunks: 0,
        });

        // Step 3: Process chunks in batches
        for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
          const batch = chunks.slice(i, i + BATCH_SIZE);
          await processChunkBatch(taskRef.id, documentId, batch, i);
        }

        // Step 4: Update task status to completed
        await taskRef.update({
          status: 'completed',
          progress: 100,
          updatedAt: admin.firestore.FieldValue.serverTimestamp(),
        });

        return { success: true, message: 'Vectorization process started', taskId: taskRef.id };
      } catch (error) {
        console.error('Error in startVectorization:', error);
        throw new functions.https.HttpsError('internal', 'Vectorization process failed to start');
      }
    });

async function processChunkBatch(taskId: string, documentId: string, chunks: string[], startIndex: number) {
  const chunkData: ChunkData[] = chunks.map((chunk, index) => ({
    text: chunk,
    metadata: {
      documentId,
      chunkIndex: startIndex + index,
      // Add any other metadata extraction logic here
    },
  }));

  const index = pinecone.Index(PINECONE_INDEX_NAME);

  for (const chunk of chunkData) {
    await retryWithBackoff(async () => {
      const [embedding] = await embeddings.embedDocuments([chunk.text]);
      await index.upsert([{
        id: `${documentId}-${chunk.metadata.chunkIndex}`,
        values: embedding,
        metadata: chunk.metadata,
      }]);
    });

    // Update progress
    await admin.firestore().collection('vectorizationTasks').doc(taskId).update({
      processedChunks: admin.firestore.FieldValue.increment(1),
      progress: admin.firestore.FieldValue.increment(100 / chunks.length),
      updatedAt: admin.firestore.FieldValue.serverTimestamp(),
    });
  }
}

async function retryWithBackoff<T>(operation: () => Promise<T>): Promise<T> {
  let retries = 0;
  while (retries < MAX_RETRIES) {
    try {
      return await operation();
    } catch (error: unknown) {
      if (retries === MAX_RETRIES - 1) throw error;
      if (axios.isAxiosError(error) && error.response?.status === 429) {
        const delay = INITIAL_BACKOFF * Math.pow(2, retries);
        console.log(`Rate limited. Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        retries++;
      } else {
        throw error;
      }
    }
  }
  throw new Error('Max retries reached');
}


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


export const reprocessFailedAndPendingTasks = functions
    .runWith({
      timeoutSeconds: 540,
      memory: '2GB'
    })
    .https.onCall(async (data, context) => {
      try {
        const tasksSnapshot = await admin.firestore().collection('vectorizationTasks')
            .where('status', 'in', [VectorizationStatus.FAILED, VectorizationStatus.PENDING])
            .get();

        if (tasksSnapshot.empty) {
          console.log('No failed or pending tasks found.');
          return { success: true, message: 'No tasks to reprocess' };
        }

        const taskUpdates = tasksSnapshot.docs.map(async (doc) => {
          const task = doc.data() as VectorizationTask;
          console.log(`Reprocessing task ${doc.id} with status ${task.status}`);

          try {
            // Reprocess the task
            await processVectorizationTask(task);

            // Verify that vectors were created in Pinecone
            const vectorCount = await getVectorCount();
            if (vectorCount === 0) {
              throw new Error('No vectors created in Pinecone');
            }

            // Update task status to completed
            await doc.ref.update({
              status: VectorizationStatus.COMPLETED,
              progress: 100,
              updatedAt: admin.firestore.FieldValue.serverTimestamp(),
            });

            console.log(`Successfully reprocessed task ${doc.id}`);
            return { taskId: doc.id, success: true, vectorCount };
          } catch (error) {
            console.error(`Error reprocessing task ${doc.id}:`, error);
            await doc.ref.update({
              status: VectorizationStatus.FAILED,
              updatedAt: admin.firestore.FieldValue.serverTimestamp(),
              error: error instanceof Error ? error.message : String(error),
            });
            return { taskId: doc.id, success: false, error: error instanceof Error ? error.message : String(error) };
          }
        });

        const results = await Promise.all(taskUpdates);
        const successCount = results.filter(r => r.success).length;
        const failCount = results.length - successCount;
        const totalVectors = results.reduce((sum, r) => sum + (r.vectorCount || 0), 0);

        return {
          success: true,
          message: `Reprocessed ${results.length} tasks. ${successCount} succeeded, ${failCount} failed. Total vectors: ${totalVectors}`,
          results: results
        };
      } catch (error) {
        console.error('Error in reprocessFailedAndPendingTasks:', error);
        throw new functions.https.HttpsError('internal', 'Failed to reprocess tasks. Please try again.');
      }
    });

async function processVectorizationTask(task: VectorizationTask): Promise<void> {
  const index = pinecone.Index(PINECONE_INDEX_NAME);

  for (let i = 0; i < task.texts.length; i += BATCH_SIZE) {
    const batch = task.texts.slice(i, i + BATCH_SIZE);
    const embeddings = await batchEmbedTexts(batch);

    const vectors: PineconeRecord<RecordMetadata>[] = embeddings.map((embedding, j) => ({
      id: `${task.id}-${i + j}`,
      values: embedding,
      metadata: {
        text: batch[j].slice(0, 1000), // Truncate text to 1000 characters
        intent: task.metadata.intent,
        sentiment: task.metadata.sentiment,
        places: task.metadata.entities.places.join(','),
        people: task.metadata.entities.people.join(','),
        organizations: task.metadata.entities.organizations.join(','),
        dates: task.metadata.entities.dates.join(','),
        numericValues: task.metadata.numericValues.join(','),
      }
    }));

    await index.upsert(vectors);
  }
}

async function getVectorCount(): Promise<number> {
  const index = pinecone.Index(PINECONE_INDEX_NAME);
  const statsResponse: IndexStatsDescription = await index.describeIndexStats();
  return statsResponse.totalRecordCount || 0;
}


async function batchEmbedTexts(texts: VectorizationTask['texts']): Promise<number[][]> {
  return retryOperation(async () => {
    const embeddingResponse = await openai.createEmbedding({
      model: "text-embedding-ada-002",
      input: texts,
    });
    return embeddingResponse.data.data.map(item => item.embedding);
  });
}



















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

    // Reduce batch size
    const batchSize = 3;

    await batchProcess(relevantChunks, batchSize, async (batch) => {
      const batchResults = await Promise.all(batch.map(async (chunk) => {
        const metadata = extractMetadata(chunk);
        const summaries = await retryWithBackoff(() => progressiveSummarization(chunk));
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
      return retryWithBackoff(() => embeddings.embedDocuments(texts));
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

    await retryWithBackoff(() => index.upsert(vectors));

    return { success: true, message: `Processed ${processedChunks.length} chunks of the document` };
  } catch (error) {
    console.error('Error in processDocument:', error);
    if (axios.isAxiosError(error)) {
      console.error('Axios error details:', error.response?.data);
    }
    throw new functions.https.HttpsError('internal', 'Document processing failed. Please try again.');
  }
});


