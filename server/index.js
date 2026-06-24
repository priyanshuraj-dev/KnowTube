import express from "express";
import cors from "cors";
import { agent } from "./agent.js";
import { addYTVideoToVectorStore } from "./embeddings.js";
const port = process.env.PORT || 3000;

const app = express();

app.use(express.json({limit:'200mb'})); 
app.use(cors());

app.get('/',(req,res) => {
    res.send("Hello World");
})

// app.post('/generate', async (req,res) => {
//     const {query,thread_id} = req.body;
//     console.log(query);
//     if (!query || !thread_id) {
//         return res.status(400).send('query and thread_id are required');
//     }
//     try {
//         const result = await agent.invoke(
//             { messages: [{ role: "user", content: query }] },
//             { configurable: { thread_id } }
//         );
//         console.log(result.messages.at(-1).content)
//         res.send(result.messages.at(-1).content);
//     } catch (err) {
//         if (err.code === 'ECONNRESET' || err.message?.includes('Connection terminated')) {
//             // retry once with a fresh connection
//             console.log('DB connection dropped mid-query, retrying...');
//             try {
//                 const result = await agent.invoke(
//                     { messages: [{ role: "user", content: query }] },
//                     { configurable: { thread_id } }
//                 );
//                 console.log(result.messages.at(-1).content)
//                 res.send(result.messages.at(-1).content);
//             } catch (retryErr) {
//                 res.status(500).send('Something went wrong, please try again.');
//             }
//         } else {
//             res.status(500).send('Something went wrong, please try again.');
//         }
//     }
// });

app.post('/generate', async (req, res) => {
    const { query, thread_id } = req.body;
    console.log('Query:', query);

    if (!query || !thread_id) {
        return res.status(400).send('query and thread_id are required');
    }

    try {
        const result = await Promise.race([
            agent.invoke(
                { messages: [{ role: "user", content: query }] },
                { configurable: { thread_id } }
            ),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Agent timeout')), 60000)
            )
        ]);

        const lastMessage = result.messages.at(-1);
        const content = lastMessage.content;

        let responseText;
        if (typeof content === 'string') {
            responseText = content;
        } else if (Array.isArray(content)) {
            responseText = content
                .filter(block => block.type === 'text')
                .map(block => block.text)
                .join('\n');
        } else {
            responseText = JSON.stringify(content);
        }

        console.log('Response:', responseText);
        res.send(responseText);

    } catch (err) {
        console.error('Error:', err.message); // 👈 check this in terminal
        
        if (err.message === 'Agent timeout') {
            return res.status(504).send('Request timed out, please try again.');
        }

        if (err.code === 'ECONNRESET' || err.message?.includes('Connection terminated')) {
            console.log('DB connection dropped mid-query, retrying...');
            try {
                const result = await agent.invoke(
                    { messages: [{ role: "user", content: query }] },
                    { configurable: { thread_id } }
                );
                const content = result.messages.at(-1).content;
                res.send(typeof content === 'string' ? content : content.map(b => b.text).join('\n'));
            } catch (retryErr) {
                console.error('Retry error:', retryErr.message);
                res.status(500).send('Something went wrong, please try again.');
            }
        } else {
            res.status(500).send('Something went wrong, please try again.');
        }
    }
});

app.post('/webhook',async (req,res) => {
    console.log(req.body);
    await Promise.all(req.body.map(async(video) => addYTVideoToVectorStore(video)))
    res.send('OK')
})

app.listen(port,()=>{
    console.log(`Server is running on port ${port}`)
})