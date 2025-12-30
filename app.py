import os
import streamlit as st
from typing import Dict, Any, List
from agent import ReasoningAgent, AgentReply

# --- 1. CONFIGURATION ---
APP_TITLE = "Reasoning Agent • Groq UI Demo"

def init_state():
    if "agent" not in st.session_state:
        st.session_state.agent = ReasoningAgent()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role": "user"|"assistant", "content": str, "trace": [ToolEvent]}]

# --- 2. SIDEBAR (CONTROLS) ---
def sidebar():
    st.sidebar.title("⚙️ Agent Controls")
    
    # Check for API status
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    status_color = "green" if (has_groq or has_openai) else "orange"
    status_text = "Groq LPU Active" if has_groq else ("OpenAI Active" if has_openai else "Local Mode")
    
    st.sidebar.markdown(f"**Status:** :{status_color}[{status_text}]")
    st.sidebar.caption("Llama-3.3-70b via Groq handles reasoning and small-talk.")
    
    st.sidebar.divider()
    
    st.sidebar.subheader("UI Settings")
    st.sidebar.checkbox("Show Tool Traces", value=True, key="show_trace", help="View the internal logic of math, time, and memory tools.")
    
    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 3. UX COMPONENTS (CENTERED HEADER) ---
def render_header():
    # Centering the title using column offsets
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(
            f"""
            <h1 style="text-align: center; color: #1e3a8a; margin-bottom: 0; font-size: 2.5rem;">
                🧠 {APP_TITLE}
            </h1>
            <p style="text-align: center; color: #64748b; font-size: 1.1rem; margin-top: 0; margin-bottom: 1.5rem;">
                High-Performance Agentic Intelligence for Clinical Reasoning
            </p>
            """, 
            unsafe_allow_html=True
        )

    # Subtle informational bar replacing the heavy blue block
    # st.info(
    #     "**System Status:** Clinical-grade reasoning active via Groq LPUs. "
    #     "I can process Math, retrieve Session Memory, and provide Real-time Clock data.",
    #     icon="🏥"
    #)

def display_trace(trace):
    """Helper to render the tool execution steps with a cleaner layout."""
    if trace and st.session_state.get("show_trace"):
        with st.expander("🛠️ Tool Execution Trace", expanded=False):
            for i, ev in enumerate(trace, start=1):
                t_col1, t_col2 = st.columns([1, 5])
                with t_col1:
                    st.write(f"**Step {i}**")
                with t_col2:
                    st.markdown(f"**Tool:** `{ev.name}`")
                    st.markdown(f"**Input:** `{ev.input}`")
                    status_icon = "✅" if ev.ok else "❌"
                    st.markdown(f"**Result:** {status_icon} `{ev.output}`")
                st.divider()

# --- 4. MAIN CHAT LOGIC ---
def main():
    # Set page config for wide layout and medical icon
    st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
    
    # Custom CSS to improve message bubbles and chat input placement
    st.markdown("""
        <style>
            .stChatMessage {
                border-radius: 15px;
                padding: 15px;
                margin-bottom: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .stChatInputContainer {
                padding-bottom: 2rem;
            }
            /* Style the titles */
            h1 { font-family: 'Inter', sans-serif; font-weight: 700; }
        </style>
    """, unsafe_allow_html=True)

    init_state()
    sidebar()
    render_header()
    st.divider()

    # Chat history container
    chat_container = st.container()

    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant":
                    display_trace(m.get("trace"))

    # Chat Input
    user_query = st.chat_input("Ask: 'What is 45*12?', 'remember name=Ali', or 'What time is it?'")

    if user_query:
        # Add user message to state and UI
        st.session_state.messages.append({"role": "user", "content": user_query})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)

        # Generate Assistant Reply with spinner
        with st.chat_message("assistant"):
            with st.spinner("Agent analyzing request..."):
                reply: AgentReply = st.session_state.agent.chat(
                    user_query,
                    history=st.session_state.messages
                )
                st.markdown(reply.text)
                display_trace(reply.tool_events)
        
        # Save response to state and refresh page
        st.session_state.messages.append({
            "role": "assistant", 
            "content": reply.text, 
            "trace": reply.tool_events
        })
        st.rerun()

if __name__ == "__main__":
    main()