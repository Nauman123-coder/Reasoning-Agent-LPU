# 🚀 Reasoning-Agent-LPU

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nauman123-coder-reasoning-agent-lpu-app-bgrcor.streamlit.app/)

A high-performance, autonomous reasoning agent system built with a **Client-Server architecture**. This project leverages **Groq's LPU (Language Processing Unit)** technology to deliver near-instantaneous AI reasoning, tool-calling, and stateful conversation management.

---

## 🌟 Live Demo
Experience the speed of the LPU-powered agent here:
👉 **[Live Deployment on Streamlit Cloud](https://nauman123-coder-reasoning-agent-lpu-app-bgrcor.streamlit.app/)**

---

## 🛠️ Tech Stack
- **Inference Engine:** [Groq Cloud](https://groq.com/) (Llama-3.3-70B-Versatile)
- **Backend Framework:** FastAPI (Asynchronous Python)
- **Frontend UI:** Streamlit
- **Logic Handling:** Python AST (for safe mathematical evaluations)
- **Communication:** RESTful API with CORS support

## 🧠 Key Features
- **Ultra-Fast Reasoning:** Powered by Groq's LPU for sub-second response latencies.
- **Multi-Agent Collaboration:** Specialized internal logic for extraction, calculation, and final synthesis.
- **Autonomous Tool-Calling:** The agent can decide when to use a Calculator, check the current time, or store "facts" into long-term memory.
- **Session-Based Memory:** Maintains context across interactions using a stateless API design where history is managed by the client.
- **Clinical-Grade Accuracy:** Configured with low-temperature settings ($0.1$) to prioritize factual correctness.

## 1. Clone the repository

- git clone [https://github.com/Nauman123-coder/Reasoning-Agent-LPU.git](https://github.com/Nauman123-coder/Reasoning-Agent-LPU.git)
- cd Reasoning-Agent-LPU
