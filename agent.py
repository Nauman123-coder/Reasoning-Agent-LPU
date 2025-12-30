from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# Assuming these are available in your local directory
from tools import ToolResult, calculator, clock, FactsStore, ToolEvent  

load_dotenv()

@dataclass
class AgentReply:
    text: str
    tool_events: List[ToolEvent]

def _is_greeting(text: str) -> bool:
    return re.match(r"^\s*(hi|hello|hey|hola|namaste|good\s*(morning|afternoon|evening)?)\b", text, re.I) is not None

def _is_followup(text: str) -> bool:
    return bool(re.search(r"\b(explain (this|that)|in brief|briefly|why|how so)\b", text, re.I))

def _last_assistant_text(history: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(history or []):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return None

def _last_user_text(history: List[Dict[str, Any]]) -> Optional[str]:
    for msg in reversed(history or []):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg["content"])
    return None

def _local_brief_explanation(prev_answer: str, prev_user: Optional[str]) -> str:
    if not prev_answer:
        return "I can explain – could you repeat the part you want clarified?"
    if len(prev_answer) <= 160 and prev_answer.count(".") <= 1:
        return f"In short: {prev_answer}"
    sent = prev_answer.split(".")[0].strip()
    if not sent:
        return "Briefly: it follows from the previous result."
    return f"Briefly: {sent}." if not prev_user else f"Briefly: {sent}. (Related to your earlier message: \"{prev_user}\")"

class ReasoningAgent:
    """
    - Natural greetings via Groq
    - Tool routing (math/time/memory)
    - Conversational memory for follow-ups
    - Groq-based reasoning fallback
    """
    
    def __init__(self) -> None:
        self.facts = FactsStore()
        self._client: Optional[OpenAI] = None
        
        # PRIORITIZE GROQ_API_KEY
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if api_key:
            # Pointing the standard OpenAI client to Groq's LPU
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            # Disable non-fatal tracing errors
            os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
    
    def _openai_reply(self, user: str, history: List[Dict[str, Any]]) -> Optional[str]:
        """Actually a Groq reply using the OpenAI client alias."""
        if not self._client:
            return None
        try:
            msgs: List[Dict[str, str]] = [{
                "role": "system",
                "content": "You are a warm, concise assistant inside a teaching demo. "
                           "Use chat history for context. Be brief (1-3 sentences). "
                           "Prioritize accuracy and medical professionalism if relevant."
            }]
            compact = [{"role": m["role"], "content": m["content"]}
                      for m in (history or []) if m.get("role") in ("user", "assistant") and m.get("content")]
            msgs.extend(compact[-6:])
            msgs.append({"role": "user", "content": user})
            
            resp = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile", # Groq-compatible model
                messages=msgs,
                temperature=0.6,
                max_tokens=160,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # Useful for debugging terminal output during your demo
            print(f"[Debug] API Error: {e}")
            return None
    
    def chat(self, user: str, history: Optional[List[Dict[str, Any]]] = None) -> AgentReply:
        events: List[ToolEvent] = []
        history = history or []
        
        # 0) Greeting
        if _is_greeting(user):
            text = self._openai_reply(user, history) or \
                   "Hello! I'm your assistant. Ask me anything, or say 'help' for my tools."
            return AgentReply(text=text, tool_events=events)
        
        # 1) Memory: remember key = value
        m = re.match(r"^\s*(remember|save)\s*(.+?)\s*=\s*(.+?)\s*$", user, flags=re.I)
        if m:
            _, key, val = m.groups()
            r: ToolResult = self.facts.remember(key, val)
            events.append(r.event)
            return AgentReply(text=f"Understood. I've saved {key} as {val}.", tool_events=events)
        
        # 2) Memory: recall key
        m = re.match(r"^\s*(recall|what did i save for)\s*(.+?)\s*$", user, flags=re.I)
        if m:
            _, key = m.groups()
            r = self.facts.recall(key)
            events.append(r.event)
            return AgentReply(text=r.content if r.ok else f"I don't have a record for '{key}' yet.", tool_events=events)
        
        # 3) Time
        if re.search(r"\b(time|date|now)\b", user, flags=re.I):
            r = clock()
            events.append(r.event)
            return AgentReply(text=f"The current time is {r.content}.", tool_events=events)
        
        # 4) Calculator
        calc_match = re.match(r"^\s*calculate\s*(.+?)$", user, flags=re.I)
        expr = calc_match.group(1) if calc_match else None
        if expr is None and re.match(r"^[\d\.\s\-\+\*\/\(\)\%]+$", user):
            expr = user.strip()
        if expr:
            r = calculator(expr)
            events.append(r.event)
            return AgentReply(text=f"Result: {r.content}" if r.ok else "Sorry, I couldn't process that math expression.", tool_events=events)
        
        # 5) Follow-up clarification
        if _is_followup(user):
            prev_answer = _last_assistant_text(history) or ""
            prev_user = _last_user_text(history)
            ai = self._openai_reply(f"{user}\n\nContext:\nPrevious answer: {prev_answer}", history)
            if ai:
                return AgentReply(text=ai, tool_events=events)
            return AgentReply(text=_local_brief_explanation(prev_answer, prev_user), tool_events=events)
        
        # 6) General fallback
        ai = self._openai_reply(user, history)
        if ai:
            return AgentReply(text=ai, tool_events=events)
        
        return AgentReply(
            text=("I can do math (e.g., 'calculate 25*4'), tell the time, or remember facts "
                  "(e.g., 'save weight = 70kg'). I also handle follow-ups like 'explain this?'. "
                  "Please ensure GROQ_API_KEY is set in your .env for full chat features."),
            tool_events=events,
        )