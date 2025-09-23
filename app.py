import os
import asyncio
import time
import streamlit as st
from dotenv import load_dotenv

from llm.model_factory import get_chat_model
from graph.agent_graph import build_graph
from retrieval.web_search import fetch_three_articles  # <-- added

load_dotenv()

# Loads and reads the summary instructions
with open("prompts/synthesis_prompt.txt", "r", encoding="utf-8") as f:
    SYNTHESIS_PROMPT = f.read()

#streamlit page setup
st.set_page_config(page_title="LangGraph Summarizer", page_icon="🧠", layout="wide")
st.title("Web Article Summarizer")

#sidebar settings
with st.sidebar:
    st.header("Model Settings")
    provider = st.selectbox(
        "Provider",
        ["ollama", "openai"],
        index=0 if os.getenv("DEFAULT_PROVIDER", "ollama") == "ollama" else 1,
    )
    if provider == "openai":
        model_name = st.text_input("OpenAI Model", os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini"))
    else:
        model_name = st.text_input("Ollama Model", os.getenv("DEFAULT_OLLAMA_MODEL", "llama3.2:3b"))
    temperature = st.slider("Temperature", 0.0, 1.0, float(os.getenv("DEFAULT_TEMPERATURE", "0")), 0.1)

#chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

#chat input
user_topic = st.chat_input("Give me a topic (e.g., 'LLM agents vs. traditional RAG')")

#Run graph func
def run_graph_with_articles(question: str, provider: str, model_name: str, temperature: float, articles: list):
    """
    Runs the LangGraph using pre-fetched articles.
    If your graph's fetch node ignores pre-filled articles, it may fetch again,
    but passing them lets you also display sources immediately in the UI.
    """
    chat_model = get_chat_model(provider, model_name, temperature)
    app = build_graph(chat_model, SYNTHESIS_PROMPT)
    init_state = {"question": question, "articles": articles, "final_answer": ""}
    return asyncio.run(app.ainvoke(init_state))

#main workflow
if user_topic:
    st.session_state.messages.append(("user", user_topic))
    with st.chat_message("assistant"):
        try:
            t0 = time.perf_counter()
            with st.status("Starting…", expanded=True) as status:
                # Step 1 — Tavily searches (3 articles). Shows article count
                status.write("🎯 Getting things lined up…")
                status.update(label="Step 1/3 • Fetching 3 articles from Tavily…")
                articles = fetch_three_articles(user_topic)
                status.write(f"🔎 Tavily returned {len(articles)} article(s)")
                if not articles:
                    raise RuntimeError("No results from Tavily. Check TAVILY_API_KEY or connectivity.")

                # Step 2 — Builds graph + runs summarization with selected LLM.
                status.update(label="Step 2/3 • Summarizing with Ollama/OpenAI…")
                final_state = run_graph_with_articles(user_topic, provider, model_name, temperature, articles)
                answer = final_state["final_answer"]

                # Step 3 — Shows time taken + mark complete
                status.update(label="Step 3/3 • Done!", state="complete")
                status.write(f"⏱️ Total time: {time.perf_counter() - t0:0.1f}s")

            # Shows final answer
            st.markdown(answer)

            # Show sources we fetched (prompt also cites them)
            if articles:
                with st.expander("Sources (the 3 fetched)"):
                    for i, a in enumerate(articles, 1):
                        st.markdown(f"**{i}. {a['title']}**  \n{a['url']}")
#saves the answer in chat history
            st.session_state.messages.append(("assistant", answer))

        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
