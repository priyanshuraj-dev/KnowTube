import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import {RecursiveCharacterTextSplitter} from '@langchain/textsplitters'
import {Document} from '@langchain/core/documents'
import data from './data.js'
import { GoogleGenAI } from "@google/genai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { tool } from "@langchain/core/tools";
import {z} from 'zod';
const video1 = data[0];

const docs = [new Document({pageContent: video1.transcript,metadata:{video_id:video1.video_id}})]

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
})

const chunks = await splitter.splitDocuments(docs)
 
const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-embedding-001",
});

const vectorStore = new MemoryVectorStore(embeddings);

await vectorStore.addDocuments(chunks)

// retrieve the most relevant chunks
const retrievedDocs = await vectorStore.similaritySearch('What was the finish time of Norris?',1);
// console.log(retrievedDocs)

// retrieval tool
const retrieveTool = tool(
    async({query}) => {
    console.log(query)
    const retrievedDocs = await vectorStore.similaritySearch(query,3);
    const serializedDocs = retrievedDocs.map((doc) => doc.pageContent).join('\n')
    return serializedDocs;
},{
    name: 'retrieve',
    description:'Retrieve the most relevant chunks of text from the transcript of a youtube video',
    schema: z.object({
        query: z.string(),
    })
})

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  apiKey: process.env.GEMINI_API_KEY,
  temperature: 0,
});

const agent = createReactAgent({ llm: model, tools: [retrieveTool] });

const result = await agent.invoke({
  messages: [{ role: "user", content: "What was the finish time of Norris?" }],
});

console.log(result.messages.at(-1).content)