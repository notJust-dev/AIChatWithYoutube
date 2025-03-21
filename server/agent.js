import { ChatAnthropic } from '@langchain/anthropic';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { MemorySaver } from '@langchain/langgraph';

import { vectorStore, addYTVideoToVectorStore } from './embeddings.js';
import data from './data.js';

// await addYTVideoToVectorStore(data[0]);
// await addYTVideoToVectorStore(data[1]);

// retrieveal tool
const retrieveTool = tool(
  async ({ query }, { configurable: { video_id } }) => {
    const retrievedDocs = await vectorStore.similaritySearch(query, 3, {
      video_id,
    });

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

const checkpointer = new MemorySaver();

const agent = createReactAgent({
  llm,
  tools: [retrieveTool],
  checkpointer,
});

// testing the agent
const video_id = 'Pxn276cWKeI';

console.log('Q1: What will people learn from this video?');
const results = await agent.invoke(
  {
    messages: [
      {
        role: 'user',
        content:
          'What will people learn from the video based on its transcript?',
      },
    ],
  },
  { configurable: { thread_id: 1, video_id } }
);

console.log(results.messages.at(-1)?.content);
