import os
import json
import uuid
import requests
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import chromadb
from chromadb.config import Settings

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cryptocrew.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="CryptoCrew - CrewAI Version")

# Initialize Groq LLMs
logger.info("Initializing Groq LLMs...")
try:
    compound_beta_llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="compound-beta"
    )
    logger.info("[SUCCESS] Compound-Beta LLM initialized successfully")
    
    # Default LLaMA-4 LLM (will be overridden by settings)
    llama4_llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
    )
    logger.info("[SUCCESS] LLaMA-4 LLM initialized successfully")
except Exception as e:
    logger.error(f"[ERROR] Failed to initialize LLMs: {e}")
    raise

def create_llm_with_settings(settings: Dict[str, Any] = None) -> ChatGroq:
    if not settings:
        settings = {
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "temperature": 0.7,
            "maxTokens": 4096
        }
    
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=settings.get("model", "meta-llama/llama-4-maverick-17b-128e-instruct"),
        temperature=settings.get("temperature", 0.7),
        max_tokens=settings.get("maxTokens", 4096)
    )

# Initialize ChromaDB
logger.info("Initializing ChromaDB connection...")
chroma_client = None
memory_collection = None

try:
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=os.getenv("CHROMA_PORT", "8000"),
        settings=Settings(allow_reset=True)
    )
    logger.info("[SUCCESS] ChromaDB client initialized")
    
    # Test connection
    chroma_client.heartbeat()
    logger.info("[SUCCESS] ChromaDB heartbeat successful")
    
    # Create or get collection
    memory_collection = chroma_client.get_or_create_collection(
        name="cryptocrew_memory",
        metadata={"description": "CryptoCrew conversation memory"}
    )
    logger.info("[SUCCESS] ChromaDB collection created/retrieved successfully")
    
except Exception as e:
    logger.warning(f"[WARNING] ChromaDB connection failed: {e}")
    logger.info("System will continue without memory persistence")
    chroma_client = None
    memory_collection = None

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default"
    settings: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    intent: str
    crypto: str
    sentiment: str
    confidence: float
    memory_id: Optional[str] = None

# Utility functions
def fetch_crypto_price_data(crypto: str) -> Dict[str, Any]:
    logger.info(f"Fetching price data for {crypto}")
    try:
        crypto_ids = {
            "bitcoin": "bitcoin", "ethereum": "ethereum", "cardano": "cardano",
            "solana": "solana", "polkadot": "polkadot", "chainlink": "chainlink",
            "litecoin": "litecoin", "ripple": "ripple", "dogecoin": "dogecoin",
            "binance coin": "binancecoin", "polygon": "matic-network",
            "avalanche": "avalanche-2", "cosmos": "cosmos", "uniswap": "uniswap",
            "aave": "aave", "compound": "compound-governance-token",
            "sushi": "sushi", "yearn": "yearn-finance", "curve": "curve-dao-token",
            "balancer": "balancer"
        }
        
        crypto_id = crypto_ids.get(crypto.lower(), "bitcoin")
        logger.info(f"Mapped {crypto} to CoinGecko ID: {crypto_id}")
        
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": crypto_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true"
        }
        
        logger.info(f"Making request to CoinGecko API: {url}")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Received data from CoinGecko: {data}")
        
        if crypto_id in data:
            price_data = data[crypto_id]
            result = {
                "price": f"${price_data.get('usd', 0):,.2f}",
                "market_cap": f"${price_data.get('usd_market_cap', 0):,.0f}",
                "volume_24h": f"${price_data.get('usd_24h_vol', 0):,.0f}",
                "change_24h": f"{price_data.get('usd_24h_change', 0):.2f}%"
            }
            logger.info(f"✅ Price data fetched successfully: {result}")
            return result
        
        logger.warning(f"Crypto ID {crypto_id} not found in response")
        return {
            "price": "$0.00",
            "market_cap": "$0",
            "volume_24h": "$0",
            "change_24h": "0.00%"
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching price data: {e}")
        return {
            "price": "$0.00",
            "market_cap": "$0",
            "volume_24h": "$0", 
            "change_24h": "0.00%"
        }

def store_memory(user_id: str, crypto: str, summary: str, sentiment_score: float, 
                intent: str, message: str, response: str) -> Optional[str]:
    if not memory_collection:
        logger.warning("Memory collection not available, skipping memory storage")
        return None
    
    try:
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        logger.info(f"Storing memory for user {user_id}, crypto {crypto}")
        
        # Simple embedding (hash-based)
        import hashlib
        hash_obj = hashlib.md5(summary.encode())
        hash_hex = hash_obj.hexdigest()
        embedding = []
        for i in range(0, len(hash_hex), 2):
            if len(embedding) < 384:
                val = int(hash_hex[i:i+2], 16) / 255.0
                embedding.append(val)
        while len(embedding) < 384:
            embedding.append(0.0)
        
        metadata = {
            "user_id": user_id,
            "crypto": crypto,
            "sentiment_score": str(sentiment_score),
            "intent": intent,
            "message": message,
            "response": response,
            "timestamp": timestamp,
            "memory_id": memory_id
        }
        
        logger.info(f"Adding memory to ChromaDB: {memory_id}")
        memory_collection.add(
            embeddings=[embedding],
            documents=[summary],
            metadatas=[metadata],
            ids=[memory_id]
        )
        
        logger.info(f"✅ Memory stored successfully: {memory_id}")
        return memory_id
    except Exception as e:
        logger.error(f"❌ Error storing memory: {e}")
        return None

def get_user_memories(user_id: str, crypto: str, limit: int = 3) -> List[Dict[str, Any]]:
    if not memory_collection:
        logger.warning("Memory collection not available, returning empty memories")
        return []
    
    try:
        logger.info(f"Retrieving memories for user {user_id}, crypto {crypto}")
        
        # Fix: Use proper ChromaDB query syntax with $and operator
        where_clause = {
            "$and": [
                {"user_id": {"$eq": user_id}}
            ]
        }
        
        if crypto:
            where_clause["$and"].append({"crypto": {"$eq": crypto}})
        
        logger.info(f"Querying ChromaDB with where clause: {where_clause}")
        results = memory_collection.query(
            query_embeddings=[[0.0] * 384],  # Empty query
            n_results=limit,
            where=where_clause
        )
        
        memories = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                if metadata:
                    memory = {
                        "memory_id": metadata.get('memory_id', ''),
                        "summary": results['documents'][0][i] if results['documents'][0] and i < len(results['documents'][0]) else '',
                        "sentiment_score": float(metadata.get('sentiment_score', 0)),
                        "intent": metadata.get('intent'),
                        "timestamp": metadata.get('timestamp', '')
                    }
                    memories.append(memory)
        
        logger.info(f"✅ Retrieved {len(memories)} memories")
        return memories
    except Exception as e:
        logger.error(f"❌ Error retrieving memories: {e}")
        return []

# CrewAI Agents
def create_intent_router_agent() -> Agent:
    return Agent(
        role="Intent Router",
        goal="Classify user intent and extract cryptocurrency mentions from messages",
        backstory="""You are an expert at analyzing cryptocurrency-related messages and determining user intent. 
        You can identify what users want to know about crypto and extract the specific cryptocurrency they're asking about.""",
        verbose=True,
        allow_delegation=False,
        llm=llama4_llm,
        tools=[],
        memory=False
    )

def create_market_analyzer_agent() -> Agent:
    return Agent(
        role="Market Analyzer",
        goal="Analyze cryptocurrency market sentiment and fetch real-time price data using Compound-Beta",
        backstory="""You are a cryptocurrency market analyst with access to real-time data through Compound-Beta. 
        You can fetch current prices, market data, and provide sentiment analysis for any cryptocurrency.""",
        verbose=True,
        allow_delegation=False,
        llm=compound_beta_llm,
        tools=[],
        memory=False
    )

def create_memory_agent() -> Agent:
    return Agent(
        role="Memory Manager",
        goal="Manage and retrieve conversation history and user context",
        backstory="""You are responsible for maintaining conversation memory and providing historical context. 
        You can store and retrieve past interactions to provide personalized responses.""",
        verbose=True,
        allow_delegation=False,
        llm=llama4_llm,
        tools=[],
        memory=False
    )

def create_response_generator_agent() -> Agent:
    return Agent(
        role="Crypto Assistant",
        goal="Generate personalized, informative responses as the main cryptocurrency chatbot",
        backstory="""You are a friendly and knowledgeable cryptocurrency assistant. You provide clear, helpful, 
        and personalized advice about cryptocurrencies. You use all available context including market data, 
        sentiment analysis, and user history to give the most relevant and useful information.""",
        verbose=True,
        allow_delegation=False,
        llm=llama4_llm,
        tools=[],
        memory=False
    )

# CrewAI Tasks
def create_intent_routing_task(message: str, user_id: str) -> Task:
    return Task(
        description=f"""
        Analyze the following user message and classify the intent and extract the cryptocurrency mentioned.
        
        User Message: "{message}"
        User ID: {user_id}
        
        Intent categories:
        - sentiment_analysis: User wants to know market sentiment or emotional analysis
        - investment_advice: User is asking for investment recommendations or advice
        - news: User wants latest news or updates about crypto
        - price_prediction: User wants price forecasts or predictions
        - technical_analysis: User wants technical analysis or chart analysis
        - general_question: General questions about cryptocurrency
        
        Respond with a JSON object containing:
        - intent: the classified intent
        - crypto: the cryptocurrency mentioned
        - sentiment: initial sentiment guess (positive/negative/neutral)
        - confidence: confidence score (0-1)
        """,
        agent=create_intent_router_agent(),
        expected_output="JSON with intent, crypto, sentiment, and confidence"
    )

def create_market_analysis_task(message: str, crypto: str) -> Task:
    return Task(
        description=f"""
        You are a cryptocurrency market analyst using Compound-Beta. Analyze the market sentiment for {crypto}.
        
        User Query: "{message}"
        Cryptocurrency: {crypto}
        
        Your task is to:
        1. Use Compound-Beta to analyze current market sentiment for {crypto}
        2. Provide a sentiment score (0-1, where 1 is very positive)
        3. Give a sentiment label (positive/negative/neutral)
        4. Provide a brief market summary
        
        IMPORTANT: Use your Compound-Beta capabilities to provide real-time sentiment analysis.
        
        Respond with ONLY a JSON object like this:
        {{
            "sentiment_score": 0.75,
            "sentiment_label": "positive",
            "market_summary": "Bitcoin is showing positive momentum with strong buying pressure"
        }}
        """,
        agent=create_market_analyzer_agent(),
        expected_output="JSON object with sentiment_score, sentiment_label, and market_summary"
    )

def create_memory_retrieval_task(user_id: str, crypto: str) -> Task:
    return Task(
        description=f"""
        Retrieve relevant conversation history for user {user_id} regarding {crypto}.
        
        Look for:
        1. Past interactions about this cryptocurrency
        2. Previous sentiment analysis
        3. Investment advice given
        4. User preferences and patterns
        
        Use the get_user_memories function to retrieve historical data.
        """,
        agent=create_memory_agent(),
        expected_output="Summary of relevant historical interactions"
    )

def create_response_generation_task(message: str, crypto: str, intent: str,
                                  market_analysis: str, memories: List[Dict]) -> Task:
    return Task(
        description=f"""
        You are a friendly cryptocurrency chatbot. Generate a helpful response to the user's question.
        
        User Message: "{message}"
        Cryptocurrency: {crypto}
        Market Analysis: {market_analysis}
        
        Your task is to:
        1. Create a friendly, conversational response that answers the user's question
        2. Include the sentiment analysis from the market data
        3. Be helpful and informative
        4. Use natural, engaging language
        
        IMPORTANT: 
        - Be conversational and helpful
        - Include the sentiment analysis
        - Keep it simple and clear
        
        Respond with a natural, conversational message.
        """,
        agent=create_response_generator_agent(),
        expected_output="Natural, conversational response with sentiment analysis"
    )

# Main CrewAI workflow
def process_chat_with_crewai(message: str, user_id: str, settings: Dict[str, Any] = None) -> Dict[str, Any]:
    logger.info(f"[START] Starting CrewAI workflow for user {user_id}")
    logger.info(f"Message: {message}")
    
    # Smart intent and crypto extraction
    intent = "general_question"
    crypto = "Bitcoin"
    sentiment = "neutral"
    confidence = 0.5
    
    message_lower = message.lower()
    
    # Extract crypto from message
    crypto_keywords = {
        "bitcoin": ["bitcoin", "btc"],
        "ethereum": ["ethereum", "eth"],
        "cardano": ["cardano", "ada"],
        "solana": ["solana", "sol"],
        "banana": ["banana", "banana gun", "banana gun token"],
        "dogecoin": ["dogecoin", "doge"],
        "ripple": ["ripple", "xrp"],
        "litecoin": ["litecoin", "ltc"],
        "polkadot": ["polkadot", "dot"],
        "chainlink": ["chainlink", "link"]
    }
    
    # Find crypto in message
    for crypto_name, keywords in crypto_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            crypto = crypto_name.title()
            break
    
    # Extract intent from message
    intent_keywords = {
        "sentiment_analysis": ["sentiment", "feeling", "mood", "how is", "what's the sentiment"],
        "investment_advice": ["invest", "buy", "sell", "should i", "advice", "recommendation"],
        "price_prediction": ["prediction", "forecast", "price will", "going to", "future"],
        "technical_analysis": ["technical", "chart", "indicator", "support", "resistance", "trend"],
        "news": ["news", "update", "latest", "recent", "announcement"],
        "general_question": ["what is", "explain", "tell me about", "how does"]
    }
    
    # Find intent in message
    for intent_type, keywords in intent_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            intent = intent_type
            break
    
    logger.info(f"[SUCCESS] Extracted intent: {intent}, crypto: {crypto}")
    
    # Step 1: Direct LLM calls instead of CrewAI
    logger.info("[MARKET] Step 1: Market Analysis with Compound-Beta")
    try:
        # Test LLM connection first
        logger.info("Testing Compound-Beta LLM connection...")
        test_response = compound_beta_llm.invoke("Hello, test message")
        logger.info(f"[SUCCESS] Compound-Beta test successful: {test_response.content[:50]}...")
        
        # Direct call to Compound-Beta
        market_prompt = f"""
        Analyze the market sentiment for {crypto}. 
        User Query: "{message}"
        
        Provide a JSON response with:
        - sentiment_score: float between 0-1
        - sentiment_label: "positive", "negative", or "neutral"
        - market_summary: brief analysis
        
        Respond with ONLY valid JSON.
        """
        
        logger.info(f"Sending market analysis prompt to Compound-Beta...")
        market_response = compound_beta_llm.invoke(market_prompt)
        market_result = market_response.content
        logger.info(f"[SUCCESS] Market analysis completed: {market_result[:200]}...")
        
        # Try to parse the market analysis result
        try:
            market_data = json.loads(market_result)
            logger.info(f"[SUCCESS] Parsed market analysis: {market_data}")
            sentiment = market_data.get("sentiment_label", "neutral")
            confidence = float(market_data.get("sentiment_score", 0.5))
        except:
            logger.warning("Market analysis result is not valid JSON, using as-is")
            market_data = market_result
            
    except Exception as e:
        logger.error(f"[ERROR] Market analysis failed: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        market_result = json.dumps({
            "sentiment_score": 0.5,
            "sentiment_label": "neutral",
            "market_summary": "Unable to analyze market data at this time."
        })
    
    # Step 2: Memory Retrieval
    logger.info("[MEMORY] Step 2: Memory Retrieval")
    memories = get_user_memories(user_id, crypto, limit=3)
    logger.info(f"[SUCCESS] Retrieved {len(memories)} memories")
    
    # Step 3: Response Generation (LLaMA-4 as main chatbot)
    logger.info("[RESPONSE] Step 3: Response Generation with LLaMA-4")
    try:
        # Create LLM with custom settings if provided
        if settings:
            logger.info(f"[SETTINGS] Using custom settings: {settings}")
            custom_llm = create_llm_with_settings(settings)
            logger.info(f"[MODEL] Using custom model: {settings.get('model', 'meta-llama/llama-4-maverick-17b-128e-instruct')}")
        else:
            custom_llm = llama4_llm
            logger.info(f"[MODEL] Using default model: meta-llama/llama-4-maverick-17b-128e-instruct")
        
        # chatbot call
        response_prompt = f"""
        You are a knowledgeable cryptocurrency assistant. Provide a comprehensive response to the user's inquiry.

        User Message: "{message}"
        Cryptocurrency: {crypto}
        Market Analysis: {market_result}

        Generate a response that:
        1. Directly addresses the user's question with accurate information
        2. Incorporates relevant market sentiment and analysis data
        3. Provides actionable insights when appropriate
        4. Uses clear, professional yet approachable language
        5. Includes context about market conditions affecting {crypto}
        6. Include reference to the market analysis

        Guidelines:
        - Be informative and data-driven
        - Explain technical concepts in accessible terms
        - Include market sentiment analysis naturally in your response
        - Maintain a helpful, professional tone
        - Keep responses concise but thorough

        Focus on delivering value through accurate information and meaningful insights about {crypto} and the broader crypto market.
        """
        
        response_response = custom_llm.invoke(response_prompt)
        response_result = response_response.content
        logger.info(f"[SUCCESS] Response generation completed: {response_result[:200]}...")
        
        # Clean up the response formatting
        # Remove markdown formatting
        if response_result.startswith('```json') or response_result.startswith('```'):
            response_result = response_result.replace('```json', '').replace('```', '').strip()
        
        # Remove extra quotes at the beginning and end
        response_result = response_result.strip()
        
        # Remove quotes from the beginning
        while response_result.startswith('"'):
            response_result = response_result[1:]
        
        # Remove quotes from the end (but preserve punctuation)
        if response_result.endswith('"'):
            response_result = response_result[:-1]
        elif response_result.endswith('".'):
            response_result = response_result[:-2] + '.'
        elif response_result.endswith('"!'):
            response_result = response_result[:-2] + '!'
        elif response_result.endswith('"?'):
            response_result = response_result[:-2] + '?'
        
        # Final cleanup
        response_result = response_result.strip()
            
    except Exception as e:
        logger.error(f"[ERROR] Response generation failed: {e}")
        response_result = "I apologize, but I'm experiencing technical difficulties right now. Please try again in a moment."
    
    # Step 4: Store Memory
    logger.info("[STORE] Step 4: Store Memory")
    
    # Extract sentiment score from market analysis if available
    try:
        market_data = json.loads(market_result)
        sentiment_score = float(market_data.get("sentiment_score", confidence))
    except:
        sentiment_score = confidence
    
    memory_id = store_memory(
        user_id=user_id,
        crypto=crypto,
        summary=f"User asked about {crypto} with intent {intent}. Compound-Beta analysis: {market_result[:200]}... Response provided.",
        sentiment_score=sentiment_score,
        intent=intent,
        message=message,
        response=response_result
    )
    
    logger.info("[COMPLETE] CrewAI workflow completed successfully")
    return {
        "response": response_result,
        "intent": intent,
        "crypto": crypto,
        "sentiment": sentiment,
        "confidence": confidence,
        "memory_id": memory_id
    }

# FastAPI endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint using CrewAI"""
    logger.info(f"[RECEIVED] Received chat request from user {request.user_id}")
    logger.info(f"Message: {request.message}")
    
    try:
        logger.info("[CALL] About to call process_chat_with_crewai...")
        result = process_chat_with_crewai(request.message, request.user_id, request.settings)
        logger.info("[SUCCESS] process_chat_with_crewai completed successfully")
        
        logger.info(f"[SUCCESS] Chat request completed successfully")
        logger.info(f"Response: {result['response'][:100]}...")
        
        return ChatResponse(
            response=result["response"],
            intent=result["intent"],
            crypto=result["crypto"],
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            memory_id=result["memory_id"]
        )
    except Exception as e:
        logger.error(f"[ERROR] Error in chat processing: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ChatResponse(
            response="I apologize, but I'm experiencing technical difficulties right now. Please try again in a moment.",
            intent="general_question",
            crypto="Bitcoin",
            sentiment="neutral",
            confidence=0.0,
            memory_id=None
        )

@app.get("/health")
async def health_check():    return {
        "status": "healthy",
        "service": "cryptocrew-crewai",
        "chromadb_connected": memory_collection is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 