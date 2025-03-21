import express from 'express';
import cors from 'cors';
import { agent } from './agent.js';

const port = process.env.PORT || 3000;

const app = express();

app.use(express.json());
app.use(cors());

app.get('/', (req, res) => {
  res.send('Hello World!');
});

// curl -X POST http://localhost:3000/generate \
// -H "Content-Type: application/json" \
// -d '{
//   "query": "What will people learn from this video?",
//   "video_id": "Pxn276cWKeI",
//   "thread_id": 1
// }'

app.post('/generate', async (req, res) => {
  const { query, video_id, thread_id } = req.body;
  console.log(query, video_id, thread_id);

  const results = await agent.invoke(
    {
      messages: [
        {
          role: 'user',
          content: query,
        },
      ],
    },
    { configurable: { thread_id, video_id } }
  );

  res.send(results.messages.at(-1)?.content);
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
