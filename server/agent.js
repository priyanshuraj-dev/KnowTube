import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from 'zod';
import { MemorySaver } from "@langchain/langgraph";
import { vectorStore, embeddings, pool } from './embeddings.js';
import { triggerYoutubeVideoScrape } from "./brightdata.js";

process.on('uncaughtException', (err) => {
    if (err.code === 'ECONNRESET' || err.message === 'Connection terminated unexpectedly') {
        console.log('Neon idle disconnect — ignoring');
        return;
    }
    console.error(err);
    process.exit(1);
});

const triggerYoutubeVideoScrapeTool = tool(async ({ url }) => {
    console.log('Triggering youtube video scrape', url);
    const snapshotId = await triggerYoutubeVideoScrape(url);
    console.log('Youtube video scrape triggered', snapshotId);
    return `Video is being indexed. Snapshot ID: ${snapshotId}. Please try again in about 1 minutes.`;
}, {
    name: 'triggerYoutubeVideoScrape',
    description: `
        Trigger the scraping of a youtube video using the url.
        Only call this if retrieve returns NO_TRANSCRIPT_FOUND.
        After calling this tool, immediately return its response to the user without any modification.
    `,
    schema: z.object({
        url: z.string(),
    }),
});

const retrieveTool = tool(
    async ({ query, video_id }) => {
        console.log('VIDEO ID: ', video_id);

        const queryEmbedding = await embeddings.embedQuery(query);

        const result = await pool.query(`
            SELECT text
            FROM transcripts
            WHERE metadata->>'video_id' = $2
            ORDER BY vector <=> $1::vector
            LIMIT 3
        `, [`[${queryEmbedding.join(',')}]`, video_id]);

        console.log('Rows found:', result.rows.length);

        if (result.rows.length === 0) {
            return "NO_TRANSCRIPT_FOUND";
        }

        return result.rows.map(row => row.text).join('\n');
    },
    {
        name: 'retrieve',
        description: 'Retrieve the most relevant chunks of text from the transcript for a specific youtube video',
        schema: z.object({
            query: z.string(),
            video_id: z.string().describe('The id of the video to retrieve')
        })
    }
);

const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.5-flash",
    apiKey: process.env.GEMINI_API_KEY,
    temperature: 0,
});
const testResponse = await model.invoke([{role: 'user', content: 'say hi'}]);
console.log('Gemini test:', testResponse.content);

const checkpointer = new MemorySaver();

export const agent = createReactAgent({ 
    llm: model, 
    tools: [retrieveTool, triggerYoutubeVideoScrapeTool], 
    checkpointer,
    prompt: `You are a RAG assistant for YouTube videos.
    You are PROHIBITED from answering any question from your own knowledge.
    You MUST call the 'retrieve' tool before saying anything.

    When a user provides a YouTube URL (e.g. https://www.youtube.com/watch?v=VIDEO_ID):
    1. Extract the video_id from the URL (the value after ?v=)
    2. CALL 'retrieve' tool immediately with the query and video_id
    3. If retrieve returns "NO_TRANSCRIPT_FOUND" → CALL 'triggerYoutubeVideoScrape' with the full URL
    4. After triggering scrape → tell the user the video is being indexed (~30 seconds) and to try again
    5. If retrieve returns content → answer ONLY using that content

    YOU MUST NEVER:
    - Answer without calling 'retrieve' first
    - Use your training knowledge about any video
    - Guess or hallucinate video content
    - Skip tool calls for any reason`,
});