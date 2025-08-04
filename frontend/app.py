import os
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

CREWAI_URL = os.getenv("CREWAI_URL", "http://localhost:8001")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        user_id = data.get('user_id', 'default')
        settings = data.get('settings', None)
        
        # Call the CrewAI backend
        response = requests.post(
            f"{CREWAI_URL}/chat",
            json={"message": message, "user_id": user_id, "settings": settings},
            timeout=120  # Increased timeout for CrewAI processing
        )
        response.raise_for_status()
        
        return jsonify(response.json())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    try:
        response = requests.get(f"{CREWAI_URL}/health", timeout=5)
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True) 