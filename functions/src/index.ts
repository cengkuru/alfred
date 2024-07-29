import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Configuration, OpenAIApi } from 'openai';
import axios from 'axios';
import {https, logger} from "firebase-functions";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

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
interface RelevantContext {
  context: string;
  pineconeResults: any;
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


async function getRelevantContext(question: string): Promise<RelevantContext> {
  const index = pinecone.Index(PINECONE_INDEX_NAME);
  const queryEmbedding = await embeddings.embedQuery(question);

  const queryResponse = await index.query({
    vector: queryEmbedding,
    topK: 10,
    includeValues: true,
    includeMetadata: true,
  });

  const contexts = queryResponse.matches
      .map(match => match.metadata?.text || '')
      .filter(Boolean);

  return {
    context: contexts.join('\n\n'),
    pineconeResults: queryResponse.matches
  };
}


const ANTHROPIC_API_KEY = functions.config().anthropic.apikey;
const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';


async function chunkText(text: string, chunkSize: number = 400, overlap: number = 20): Promise<string[]> {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: chunkSize,
    chunkOverlap: overlap,
  });
  return await splitter.splitText(text);
}

async function generatePrompt(question: string, context: string, userLevel: string = 'intermediate'): Promise<string> {
  const chunks = await chunkText(context);
  const contextPrompt = chunks.map(chunk => `Context: ${chunk}\n`).join('\n');

  return `
${contextPrompt}

You are Alfred, an AI assistant specializing in Infrastructure Transparency and the Construction Sector Transparency Initiative (CoST). Your knowledge spans infrastructure project lifecycles, transparency in public contracting, and socio-economic impacts of infrastructure investments.

User's Expertise Level: ${userLevel}
Question: ${question}

Instructions:
1. Analyze the question and context to determine the most appropriate response structure.
2. Provide a concise, accurate answer that directly addresses the user's query.
3. Use markdown formatting to enhance readability.
4. Include relevant statistics or examples from the provided context when applicable.
5. Adjust your language and explanation depth based on the user's expertise level.
6. Highlight CoST's role and impact in promoting infrastructure transparency when relevant.
7. Relate information to real-world benefits of transparency in infrastructure projects if appropriate.
8. Use headers and sections only when they enhance the answer's clarity and organization.
9. Conclude with a brief summary and an open-ended question to encourage further engagement.

Remember:
- Not all responses require a rigid structure with predefined sections.
- Be flexible and use headers that best fit the specific question and context.
- Prioritize clarity and relevance over a one-size-fits-all format.
- Ground your answers in the provided context and CoST's methodologies.
- Be concise yet comprehensive, avoiding unnecessary verbosity.

Begin your response now:`;
}



export const askAI = functions.runWith({
  timeoutSeconds: 300,
  memory: '1GB'
}).https.onCall(async (data, context) => {
  const question: string = data?.question;
  const userLevel: string = data?.userLevel || 'intermediate';

  if (!question || typeof question !== 'string') {
    throw new functions.https.HttpsError('invalid-argument', 'Please provide a question as a string.');
  }

  try {
    const { context: relevantContext } = await getRelevantContext(question);
    const prompt = await generatePrompt(question, relevantContext, userLevel);

    const requestBody = {
      model: "claude-3-haiku-20240307",
      max_tokens: 1000,
      messages: [
        {
          role: "user",
          content: prompt
        }
      ]
    };

    const response = await axios.post(
        ANTHROPIC_API_URL,
        requestBody,
        {
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01'
          },
        }
    );

    if (response.data && response.data.content && response.data.content[0] && response.data.content[0].text) {
      const rawAnswer = response.data.content[0].text.trim();
      const processedAnswer = postProcessAnswer(rawAnswer);
      return { answer: processedAnswer };
    } else {
      throw new Error('Unexpected response structure from AI');
    }
  } catch (error) {
    functions.logger.error('Error in askAI function:', error);
    if (axios.isAxiosError(error)) {
      if (error.response) {
        functions.logger.error('Anthropic API error response:', error.response.data);
        throw new functions.https.HttpsError('unavailable', `The AI service returned an error: ${error.response.data.error?.message || 'Unknown error'}`);
      } else if (error.request) {
        throw new functions.https.HttpsError('deadline-exceeded', 'The request to the AI service timed out. Please try again.');
      }
    }
    throw new functions.https.HttpsError('internal', 'Sorry, I encountered an issue while processing your question. Please try again later.');
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
function postProcessAnswer(rawAnswer: string): string {
  let processedAnswer = rawAnswer;

  // Apply basic formatting (headers, lists, etc.)
  processedAnswer = applyBasicFormatting(processedAnswer);

  // Intelligent section detection and formatting
  processedAnswer = detectAndFormatSections(processedAnswer);

  // Add emojis to headers intelligently
  processedAnswer = addHeaderEmojis(processedAnswer);

  // Ensure proper spacing and readability
  processedAnswer = improveReadability(processedAnswer);

  // Add a dynamic concluding remark
  processedAnswer = addDynamicConclusion(processedAnswer);

  return processedAnswer;
}

function applyBasicFormatting(text: string): string {
  // Format headings
  text = text.replace(/^(#{1,6})\s*(.+)$/gm, (_, hashes, headingText) => {
    return `\n${hashes} ${headingText.trim()}\n`;
  });

  // Format lists
  text = text.replace(/^(\d+)\.\s+/gm, (_, number) => `\n${number}. `);
  text = text.replace(/^[-*]\s+/gm, '\n- ');

  // Format blockquotes
  text = text.replace(/^>\s*(.*)/gm, (_, content) => `> ${content.trim()}`);

  // Format code blocks
  text = text.replace(/```(\w+)?\n([\s\S]+?)\n```/g, (_, language, code) => {
    return `\n\`\`\`${language || ''}\n${code.trim()}\n\`\`\`\n`;
  });

  // Format inline code
  text = text.replace(/`([^`\n]+)`/g, (_, code) => `\`${code.trim()}\``);

  // Format bold and italic
  text = text.replace(/\*\*\*([^*\n]+)\*\*\*/g, (_, content) => `***${content.trim()}***`);
  text = text.replace(/\*\*([^*\n]+)\*\*/g, (_, content) => `**${content.trim()}**`);
  text = text.replace(/\*([^*\n]+)\*/g, (_, content) => `*${content.trim()}*`);

  // Format links
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, linkText, url) => `[${linkText.trim()}](${url.trim()})`);

  // Format horizontal rules
  text = text.replace(/^(-{3,}|\*{3,})$/gm, '\n---\n');

  return text;
}

function addHeaderEmojis(text: string): string {
  const emojiMap: { [key: string]: string } = {
    'Main Answer': '💡',
    'Key Points': '🔑',
    'Detailed Explanation': '📊',
    'Real-World Impact': '🌍',
    'Summary': '📝',
    'Example': '🔍',
    'Conclusion': '🎯',
    'Further Reading': '📚'
  };

  return text.replace(/^(#{1,6})\s+(.*?)$/gm, (match, hashes, header) => {
    const trimmedHeader = header.trim();
    const emoji = emojiMap[trimmedHeader] || '';
    return emoji ? `${hashes} ${emoji} ${trimmedHeader}` : match;
  });
}

function improveReadability(text: string): string {
  // Remove excessive newlines
  text = text.replace(/\n{3,}/g, '\n\n');

  // Ensure there's always a newline after a heading
  text = text.replace(/^(#{1,6}.*?)$/gm, '$1\n');

  // Add line breaks after punctuation for better readability
  text = text.replace(/([.!?])\s+(?=[A-Z])/g, '$1\n\n');

  // Capitalize first letter after punctuation
  text = text.replace(/([.!?]\s+)([a-z])/g, (_, punctuation, letter) => `${punctuation}${letter.toUpperCase()}`);

  // Ensure proper spacing around list items
  text = text.replace(/(\n[^\n]+)(\n[-\d])/g, '$1\n\n$2');

  return text;
}

function detectAndFormatSections(text: string): string {
  const sections = [
    { name: 'Main Answer', emoji: '💡' },
    { name: 'Key Points', emoji: '🔑' },
    { name: 'Detailed Explanation', emoji: '📊' },
    { name: 'Real-World Impact', emoji: '🌍' },
    { name: 'Summary', emoji: '📝' },
    { name: 'Example', emoji: '🔍' },
    { name: 'Conclusion', emoji: '🎯' },
    { name: 'Further Reading', emoji: '📚' }
  ];

  sections.forEach(section => {
    const regex = new RegExp(`(?:^|\\n)${section.name}:?\\s*(.+(?:\\n(?!\\n).+)*)`, 'i');
    const match = text.match(regex);
    if (match) {
      const content = match[1].trim();
      text = text.replace(match[0], `\n\n## ${section.emoji} ${section.name}\n\n${content}\n\n`);
    }
  });

  return text;
}

function addDynamicConclusion(text: string): string {
  const conclusions = [
    "I hope this information helps! Do you have any other questions about infrastructure transparency or CoST's work?",
    "Is there anything else you'd like to know about this topic or CoST's initiatives?",
    "How else can I assist you with understanding infrastructure transparency and CoST's methodologies?",
    "Would you like to explore any specific aspects of infrastructure project lifecycles or transparency measures further?"
  ];

  return text + "\n\n" + conclusions[Math.floor(Math.random() * conclusions.length)];
}


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

