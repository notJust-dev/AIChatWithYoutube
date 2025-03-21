import { ChatAnthropic } from '@langchain/anthropic';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { MemorySaver } from '@langchain/langgraph';

import { vectorStore, addYTVideoToVectorStore } from './embeddings.js';
import data from './data.js';
import { triggerYoutubeVideoScrape } from './brightdata.js';

// await addYTVideoToVectorStore(data[0]);
// await addYTVideoToVectorStore(data[1]);

const triggerYoutubeVideoScrapeTool = tool(
  async ({ url }) => {
    console.log('Triggering youtube video scrape', url);

    const snapshotId = await triggerYoutubeVideoScrape(url);

    console.log('Youtube video scrape triggered', snapshotId);
    return snapshotId;
  },
  {
    name: 'triggerYoutubeVideoScrape',
    description: `
    Trigger the scraping of a youtube video using the url. 
    The tool start a scraping job, that usually takes around 7 seconds
    The tool will return a snapshot/job id, that can be used to check the status of the scraping job
    Before calling this tool, make sure that it is not already in the vector store
  `,
    schema: z.object({
      url: z.string(),
    }),
  }
);
// retrieveal tool
const retrieveTool = tool(
  async ({ query, video_id }, { configurable: {} }) => {
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
      'Retrieve the most relevant chunks of text from the transcript for a specific youtube video',
    schema: z.object({
      query: z.string(),
      video_id: z.string().describe('The id of the video to retrieve'),
    }),
  }
);

// retrieveal similar videos tool
const retrieveSimilarVideosTool = tool(
  async ({ query }) => {
    const retrievedDocs = await vectorStore.similaritySearch(query, 30);

    const ids = retrievedDocs.map((doc) => doc.metadata.video_id).join('\n ');

    return ids;
  },
  {
    name: 'retrieveSimilarVideos',
    description: 'Retrieve the ids of the most similar videos to the query',
    schema: z.object({
      query: z.string(),
    }),
  }
);

const llm = new ChatAnthropic({
  modelName: 'claude-3-7-sonnet-latest',
});

const checkpointer = new MemorySaver();

export const agent = createReactAgent({
  llm,
  tools: [
    retrieveTool,
    triggerYoutubeVideoScrapeTool,
    retrieveSimilarVideosTool,
  ],
  checkpointer,
});
