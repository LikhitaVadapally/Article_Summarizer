# Web Article Summarizer 🧠
## A simple LLM agent that:

Takes a user’s topic from a Streamlit chat interface

Fetches 3 relevant articles via the Tavily API

Summarizes them into a concise, grounded answer with citations

Runs on either OpenAI models or local Ollama models (swap with one click)

Built with LangGraph, LangChain, Streamlit, and Tavily.



# 📂 Project Structure
llm_agent/
├─ app.py                  # Streamlit entry point (UI + workflow)

├─ graph/
│  └─ agent_graph.py       # LangGraph definition (fetch → summarize)

├─ llm/
│  └─ model_factory.py     # Model provider factory (OpenAI vs Ollama)

├─ retrieval/
│  └─ web_search.py        # Tavily API search wrapper

├─ prompts/
│  └─ synthesis_prompt.txt # Instructions for summarization

├─ requirements.txt        # Python dependencies

├─ .env.example            # Example env vars (keys + defaults)

└─ README.md               

# ⚙️ Installation

## 1. Clone the repo:

git clone https://github.com/your-username/llm_agent.git
cd llm_agent


## 2. Create and activate a virtual environment:

python -m venv .venv
.venv\Scripts\activate   # On Windows (PowerShell)
source .venv/bin/activate # On Mac/Linux


## 3. Install dependencies:

pip install -r requirements.txt


## 4. Set up .env:

cp .env.example .env

### Fill in:

OPENAI_API_KEY=sk-... (if using OpenAI)

TAVILY_API_KEY=tvly-...

Defaults like DEFAULT_PROVIDER=ollama

## 5. If using Ollama locally:

Install Ollama

Pull a model:

ollama pull llama3.2:3b

# ▶️ Running the App

## Start the Streamlit app:

streamlit run app.py


## You’ll see:

Local URL: http://localhost:8501
