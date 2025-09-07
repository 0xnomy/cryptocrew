# CryptoCrew

CryptoCrew is a multi-agent chatbot for cryptocurrency insights, powered by CrewAI and Groq LLMs. It provides real-time market data, sentiment analysis, and personalized responses about various cryptocurrencies.

![CryptoCrew](mp.png)

<<<<<<< HEAD
## How It Works
- The backend uses CrewAI agents for intent classification, market analysis, memory management, and response generation.
- ChromaDB stores conversation history for context-aware replies.
- The frontend (Flask) serves a chat UI and connects to the backend via REST API.
=======
Demo: https://drive.google.com/file/d/1M1e4E2Dd2vxbW5lKz6Cahn891EGu33I6/view?usp=sharing

## Architecture
>>>>>>> dec4e4c45ccc6d4298a53763bba3edb32b995a52

## APIs Used
- Groq LLMs (via langchain-groq)
- ChromaDB (for memory)
- CoinGecko (for price data)
- FastAPI (backend)
- Flask (frontend)

## Main Endpoints
- `POST /chat` — Chat with the bot
- `GET /health` — Service health check

## Quick Start
1. Add your Groq API key to `.env`.
2. Run `docker-compose up -d` or use `python start.py` for local development.
3. Access the app at http://localhost:3000

## License

<<<<<<< HEAD
This project is licensed under the MIT License. 
=======
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
>>>>>>> dec4e4c45ccc6d4298a53763bba3edb32b995a52
