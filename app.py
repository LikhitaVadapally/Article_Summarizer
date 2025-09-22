import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

from llm.model_factory import get_chat_model
from graph.agent_graph import build_graph

load_dotenv()

# Load prompt
with open("prompts/synthesis_prompt.txt", "r", encoding="utf-8") as f:
    SYNTHESIS_PROMPT = f.read()

st.set_page_config(page_title="LangGraph Summarizer", page_icon="🧠", layout="wide")
st.title("🧠 Simple LangGraph Web Summarizer")

with st.sidebar:
    st.header("Model Settings")
    provider = st.selectbox("Provider", ["ollama", "openai"], index=0 if os.getenv("DEFAULT_PROVIDER","ollama")=="ollama" else 1)
    if provider == "openai":
        model_name = st.text_input("OpenAI Model", os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini"))
    else:
        model_name = st.text_input("Ollama Model", os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:3b"))
    temperature = st.slider("Temperature", 0.0, 1.0, float(os.getenv("DEFAULT_TEMPERATURE","0")), 0.1)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_topic = st.chat_input("Give me a topic (e.g., 'LLM agents vs. traditional RAG')")

def run_graph(question: str, provider: str, model_name: str, temperature: float):
    chat_model = get_chat_model(provider, model_name, temperature)
    app = build_graph(chat_model, SYNTHESIS_PROMPT)
    init_state = {"question": question, "articles": [], "final_answer": ""}
    return asyncio.run(app.ainvoke(init_state))

if user_topic:
    st.session_state.messages.append(("user", user_topic))
    with st.chat_message("assistant"):
        with st.spinner("Fetching 3 articles & summarizing..."):
            try:
                final_state = run_graph(user_topic, provider, model_name, temperature)
                answer = final_state["final_answer"]
                st.markdown(answer)

                # Show sources found by the graph (optional: the prompt already cites them)
                articles = final_state.get("articles", [])
                if articles:
                    with st.expander("Sources (the 3 fetched)"):
                        for i, a in enumerate(articles, 1):
                            st.markdown(f"**{i}. {a['title']}**  \n{a['url']}")
                st.session_state.messages.append(("assistant", answer))
            except Exception as e:
                st.error(f"Error: {e}")
