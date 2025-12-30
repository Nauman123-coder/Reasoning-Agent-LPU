# server.py
from __future__ import annotations

import os
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the agent logic we converted earlier
from agent import ReasoningAgent, AgentReply

# ---- 0. GROQ CONFIGURATION ----
load_dotenv()

# Redirect the underlying OpenAI SDK to Groq's endpoint
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ["OPENAI_API_KEY"] = groq_key
    os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"

# Disable background tracing to prevent the 401 errors you saw earlier
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
os.environ["OTEL_SDK_DISABLED"] = "true"

# ---- 1. Pydantic models (JSON-safe) ----
class ToolEventModel(BaseModel):
    name: str
    input: str
    ok: bool
    output: str

class AgentReplyModel(BaseModel):
    text: str
    tool_events: List[ToolEventModel]

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []

# ---- 2. App Initialization ----
app = FastAPI(
    title="AgentKit Groq Backend", 
    description="High-speed clinical reasoning API powered by Groq LPU",
    version="1.1.0"
)

# Allow Streamlit or other frontends on localhost to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Agent Instance
# This is initialized once; it will use the Groq env vars set above.
AGENT = ReasoningAgent()

# ---- 3. API Endpoints ----

@app.post("/chat", response_model=AgentReplyModel)
def chat(req: ChatRequest) -> AgentReplyModel:
    """
    Main chat endpoint. 
    Accepts user message and history, returns agent text and tool trace.
    """
    # The agent handles the tool routing and Groq API call
    reply: AgentReply = AGENT.chat(req.message, history=req.history or [])
    
    # Convert dataclass ToolEvent -> Pydantic model for JSON serialization
    events = [
        ToolEventModel(
            name=e.name, input=e.input, ok=e.ok, output=e.output
        ) for e in (reply.tool_events or [])
    ]
    
    return AgentReplyModel(text=reply.text, tool_events=events)

@app.get("/health")
def health():
    """Health check endpoint for monitoring."""
    status = "Groq-Connected" if os.environ.get("OPENAI_BASE_URL") else "Local-Only"
    return {"ok": True, "mode": status}

if __name__ == "__main__":
    import uvicorn
    # To run: python server.py
    uvicorn.run(app, host="0.0.0.0", port=8000)