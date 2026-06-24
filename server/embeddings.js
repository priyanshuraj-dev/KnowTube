import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import { PGVectorStore } from "@langchain/pgvector";
import pg from 'pg';

export const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-embedding-001",
});

export const pool = new pg.Pool({ 
  connectionString: process.env.DB_URL 
});

export const vectorStore = await PGVectorStore.initialize(embeddings, {
    postgresConnectionOptions: {
        connectionString: process.env.DB_URL
    },
    tableName: 'transcripts',
    columns: {
        idColumnName: 'id',
        vectorColumnName: 'vector',
        contentColumnName: 'text',
        metadataColumnName: 'metadata'
    },
    distanceStrategy: 'cosine',
    dimensions: 3072
});

export const addYTVideoToVectorStore = async (videoData) => {    
    const { transcript, video_id } = videoData;

    if (!transcript || transcript.trim().length === 0) {
        console.log(`Video ${video_id} has no transcript, skipping`);
        return;
    }

    if (!video_id) {
        console.log('Missing video_id, skipping');
        return;
    }

    // duplicate check using raw SQL
    const existing = await pool.query(
        `SELECT id FROM transcripts WHERE metadata->>'video_id' = $1 LIMIT 1`,
        [video_id]
    );
    if (existing.rows.length > 0) {
        console.log(`Video ${video_id} already embedded, skipping`);
        return;
    }

    const docs = [new Document({ pageContent: transcript, metadata: { video_id } })];
    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });
    
    const chunks = await splitter.splitDocuments(docs);
    console.log(`Total chunks: ${chunks.length}`);

    const batchSize = 10;
    for (let i = 0; i < chunks.length; i += batchSize) {
        const batch = chunks.slice(i, i + batchSize);
        try {
            await vectorStore.addDocuments(batch);
            console.log(`✓ Inserted batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(chunks.length / batchSize)}`);
        } catch (e) {
            console.error(`✗ Batch ${Math.floor(i / batchSize) + 1} failed:`, e.message);
        }
        await new Promise(res => setTimeout(res, 500));
    }
};