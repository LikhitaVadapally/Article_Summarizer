import os
import asyncio
import time
import streamlit as st
from dotenv import load_dotenv
from graph.agent_graph import build_agent  # Adjust path if needed (e.g., from agent_graph)

load_dotenv()

# Robust path to the prompt file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "synthesis_prompt.txt")  # Assumes 'prompts/' dir; adjust if flat
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYNTHESIS_PROMPT = f.read()

st.set_page_config(page_title="LangGraph Summarizer", page_icon="🧠", layout="wide")
st.title("Web Article Summarizer")

with st.sidebar:
    st.header("Model Settings")
    provider = st.selectbox(
        "Provider",
        ["openai", "ollama"],
        index=0 if os.getenv("DEFAULT_PROVIDER", "ollama") == "ollama" else 1,
    )
    if provider == "openai":
        model_name = st.text_input("OpenAI Model", os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini"))
    else:
        model_name = st.text_input("Ollama Model", os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:1b"))
    temperature = st.slider("Randomness", 0.0, 1.0, float(os.getenv("DEFAULT_TEMPERATURE", "0")), 0.1)
    k = st.slider("Number of sources (k)", 1, 10, int(os.getenv("DEFAULT_K", "3")))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
elif st.session_state.messages and isinstance(st.session_state.messages[0], tuple):
    migrated = []
    for role, content in st.session_state.messages:
        migrated.append({"role": role, "content": content, "sources": [], "errors": []})
    st.session_state.messages = migrated

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Article Summaries & Sources"):
                for i, summary in enumerate(m["sources"], 1):
                    with st.container():
                        st.markdown(f"### {i}. {summary.get('title', f'Source {i}')}")
                        if summary.get('url'):
                            st.markdown(f"[Source URL]({summary['url']})")
                        # Relevance Score removed
                        st.markdown(f"**Summary:** {summary.get('summary', 'No summary available')}")
                        st.markdown("---")

user_topic = st.chat_input("Give me a topic (e.g., 'LLM agents vs. traditional RAG')")

def run_agent(question: str, provider: str, model_name: str, temperature: float, k: int):
    agent = build_agent(provider, model_name, temperature, SYNTHESIS_PROMPT)
    return agent(question, k)

if user_topic:
    st.session_state.messages.append({"role": "user", "content": user_topic, "sources": [], "errors": []})

    with st.chat_message("assistant"):
        try:
            t0 = time.perf_counter()
            with st.status("Starting…", expanded=True) as status:
                status.write("🎯 Agent prepping tools…")
                status.update(label=f"Searching and summarizing {k} sources…")
                response = run_agent(user_topic, provider, model_name, temperature, k)

                status.update(label="Processing articles…")
                status.update(label="Done!", state="complete")
                status.write(f"⏱️ Total time: {time.perf_counter() - t0:0.1f}s")
                status.write(f"📄 Found {len(response.summaries)} articles")

            # Display all individual article summaries
            if response.summaries:
                content_parts = []
                for i, summary in enumerate(response.summaries, 1):
                    title = summary.title.strip()
                    url = summary.url.strip()
                    summary_text = summary.summary.strip()
                    
                    part = f"### {i}. {title}\n\n"
                    if url:
                        part += f"[Source URL]({url})\n\n"
                    # Relevance Score removed
                    part += f"**Summary:** {summary_text}\n\n"
                    part += "---\n\n"
                    content_parts.append(part)
                
                full_content = "".join(content_parts)
                st.markdown(full_content)

                # Store summaries for chat history (no relevance_score)
                source_data = [
                    {
                        "title": summary.title,
                        "url": summary.url,
                        "summary": summary.summary
                    }
                    for summary in response.summaries
                ]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_content,
                    "sources": source_data,
                    "errors": []
                })

            else:
                st.warning("No articles found for this topic.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "No relevant articles were found for this topic.",
                    "sources": [],
                    "errors": ["No search results available"]
                })

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "An error occurred while processing your request.",
                "sources": [],
                "errors": [error_msg]
            })