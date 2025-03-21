import { ChatAnthropic } from '@langchain/anthropic';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Document } from '@langchain/core/documents';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';

import data from './data.js';

const video1 = data[0];

const docs = [
  new Document({
    pageContent: video1.transcript,
    metadata: { video_id: video1.video_id },
  }),
];

// Split the video into chunks
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const chunks = await splitter.splitDocuments(docs);

// console.log(chunks);

// Embed the chunks
const embeddings = new OpenAIEmbeddings({
  model: 'text-embedding-3-large',
});

const vectorStore = new MemoryVectorStore(embeddings);

await vectorStore.addDocuments(chunks);

// retrieve the most relevant chunks
const retrievedDocs = await vectorStore.similaritySearch(
  'What was the finish time of Norris?',
  1
);
// console.log('Retrieved docs: --------------------------------');
// console.log(retrievedDocs);

// retrieveal tool
const retrieveTool = tool(
  async ({ query }) => {
    console.log('Retrieving docs for query: --------------------------------');
    console.log(query);
    const retrievedDocs = await vectorStore.similaritySearch(query, 3);

    const serializedDocs = retrievedDocs
      .map((doc) => doc.pageContent)
      .join('\n ');

    return serializedDocs;
  },
  {
    name: 'retrieve',
    description:
      'Retrieve the most relevant chunks of text from the transcript of a yotube video',
    schema: z.object({
      query: z.string(),
    }),
  }
);

const llm = new ChatAnthropic({
  modelName: 'claude-3-7-sonnet-latest',
});

const agent = createReactAgent({ llm, tools: [retrieveTool] });

const results = await agent.invoke({
  messages: [{ role: 'user', content: 'What was the finish time of Norris?' }],
});

console.log(results.messages.at(-1)?.content);
