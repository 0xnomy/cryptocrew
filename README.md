# CryptoCrew

Multi-agent cryptocurrency chatbot using CrewAI and Groq LLMs.

![CryptoCrew](mp.png)

Demo: https://drive.google.com/file/d/1M1e4E2Dd2vxbW5lKz6Cahn891EGu33I6/view?usp=sharing

## Architecture

- **Intent Router**: Classifies user intent and extracts cryptocurrency mentions
- **Market Analyzer**: Fetches real-time price data and sentiment analysis
- **Memory Manager**: Manages conversation history with ChromaDB
- **Response Generator**: Generates personalized responses

## Quick Start

1. Create `.env` file:
```
GROQ_API_KEY=your_groq_api_key
```

2. Deploy with Docker:
```bash
docker-compose up -d
```

3. Access at http://localhost:3000

## Local Development

```bash
pip install -r requirements.txt
python start.py
```

## API Endpoints

- `POST /chat` - Main chat endpoint
- `GET /health` - Health check

## Environment Variables

- `GROQ_API_KEY` - Required Groq API key
- `CHROMA_HOST` - ChromaDB host (default: localhost)
- `CHROMA_PORT` - ChromaDB port (default: 8000)
- `CREWAI_URL` - Backend URL (default: http://localhost:8001)

## Ports

- Frontend: 3000
- Backend: 8001
- ChromaDB: 8000 
