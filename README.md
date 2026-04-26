# Local System File RAG

A semantic file search system that lets you describe files in natural language and find them across your entire Windows system using RAG (Retrieval Augmented Generation).

## Features

- 🔍 Natural language file search across all drives
- 🤖 AI-powered chatbot interface
- 📄 Supports text, code, PDF, Word, Excel files
- 🎨 Modern dark-themed UI
- 🆓 Completely free using Groq + Voyage AI

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get free API keys:
   - Groq: https://console.groq.com (for LLM)
   - Voyage AI: https://www.voyageai.com (for embeddings)

3. Create `.env` file:
```
GROQ_API_KEY=your_groq_key_here
VOYAGE_API_KEY=your_voyage_key_here
```

4. Run the server:
```bash
python -m uvicorn main:app --reload
```

5. Open http://localhost:8000 in your browser

6. Click "Build Index" to crawl your system (takes 10-30 mins first time)

7. Start searching! Try:
   - "python script I wrote for web scraping"
   - "budget spreadsheet from last year"
   - "my resume PDF"

## How It Works

1. **Crawl**: Walks all drives, extracts file metadata + content
2. **Chunk**: Splits content into overlapping chunks
3. **Embed**: Converts chunks to vectors using Voyage AI
4. **Search**: Your query gets embedded and matched via cosine similarity
5. **Chat**: Results are passed to Groq's LLM for natural responses

## Tech Stack

- **Backend**: FastAPI + Python
- **LLM**: Groq (llama3-8b-8192)
- **Embeddings**: Voyage AI (voyage-3-lite)
- **Vector Store**: Simple JSON + numpy
- **UI**: Vanilla HTML/CSS/JS
