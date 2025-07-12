from flask import Flask, request, jsonify, render_template_string
import asyncio
import sys
from pathlib import Path

# Import from the same directory
from rag_service import rag_service
from log_utility import setup_logger

# Set up logger
logger = setup_logger()

app = Flask(__name__)

# Enhanced HTML chat UI with only RAG functionality
CHAT_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Document RAG Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        #chat { width: 100%; max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .msg { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background-color: #e3f2fd; color: #1976d2; }
        .agent { background-color: #f3e5f5; color: #7b1fa2; }
        .system { background-color: #fff3e0; color: #f57c00; font-style: italic; }
        .error { background-color: #ffebee; color: #c62828; }
        #user-input { width: 80%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        #send-btn { padding: 10px 20px; background-color: #4caf50; color: white; border: none; border-radius: 5px; cursor: pointer; }
        #send-btn:hover { background-color: #45a049; }
        .loading { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div id="chat">
        <h2>Document RAG Assistant</h2>
        <div id="messages"></div>
        <form id="chat-form">
            <input type="text" id="user-input" placeholder="Ask about financial documents..." required />
            <button type="submit" id="send-btn">Send</button>
        </form>
    </div>
    <script>
        const form = document.getElementById('chat-form');
        const input = document.getElementById('user-input');
        const messages = document.getElementById('messages');
        
        // Initialize with system message
        messages.innerHTML = '<div class="msg system"><b>System:</b> Document RAG Mode - Ask me about earnings call transcripts and financial documents!</div>';
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            const userMsg = input.value;
            messages.innerHTML += `<div class='msg user'><b>You:</b> ${userMsg}</div>`;
            input.value = '';
            
            // Show loading message
            const loadingId = Date.now();
            messages.innerHTML += `<div id="loading-${loadingId}" class='msg loading'><b>Assistant:</b> Thinking...</div>`;
            
            try {
                const res = await fetch('/chat/rag', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: userMsg })
                });
                
                if (!res.ok) {
                    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
                }
                
                const data = await res.json();
                
                // Remove loading message and add response
                document.getElementById(`loading-${loadingId}`).remove();
                
                if (data.error) {
                    messages.innerHTML += `<div class='msg error'><b>Error:</b> ${data.error}</div>`;
                } else {
                    messages.innerHTML += `<div class='msg agent'><b>Assistant:</b> ${data.response}</div>`;
                }
            } catch (error) {
                document.getElementById(`loading-${loadingId}`).remove();
                messages.innerHTML += `<div class='msg error'><b>Error:</b> ${error.message}</div>`;
                console.error('Chat error:', error);
            }
            
            messages.scrollTop = messages.scrollHeight;
        };
    </script>
</body>
</html>
'''

@app.route("/")
def index():
    logger.info("RAG: User accessed the main page")
    return render_template_string(CHAT_HTML)

@app.route("/chat/rag", methods=["POST"])
def chat_rag():
    """Handle RAG assistant chat requests."""
    try:
        data = request.get_json()
        if not data:
            logger.error("RAG: No JSON data provided in request")
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_message = data.get("message", "")
        if not user_message:
            logger.error("RAG: No message provided in request")
            return jsonify({"error": "No message provided."}), 400
        
        logger.info(f"RAG: Processing user query: {user_message[:100]}...")
        
        # Run the async query
        response = asyncio.run(rag_service.query(user_message))
        
        logger.info(f"RAG: Generated response for user query")
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"RAG: Error processing chat request: {str(e)}")
        print(f"RAG chat error: {e}")
        return jsonify({"error": f"RAG service error: {str(e)}"}), 500

@app.route("/health")
def health():
    """Health check endpoint."""
    logger.info("RAG: Health check requested")
    return jsonify({"status": "healthy", "services": ["document_rag"]})

@app.route("/init-rag", methods=["POST"])
def init_rag():
    """Initialize the RAG service."""
    try:
        logger.info("RAG: Initializing RAG service")
        asyncio.run(rag_service.initialize())
        logger.info("RAG: RAG service initialized successfully")
        return jsonify({"status": "RAG service initialized successfully"})
    except Exception as e:
        logger.error(f"RAG: RAG initialization failed: {str(e)}")
        print(f"RAG initialization error: {e}")
        return jsonify({"error": f"RAG initialization failed: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info("RAG: Starting Flask application")
    # If Flask is not installed, run: pip install flask
    app.run(debug=True)