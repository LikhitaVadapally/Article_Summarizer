import os
import asyncio
import time
import streamlit as st
from dotenv import load_dotenv

from llm.model_factory import get_chat_model
from graph.agent_graph import build_graph

load_dotenv()

with open("prompts/synthesis_prompt.txt", "r", encoding="utf-8") as f:
    SYNTHESIS_PROMPT = f.read()

st.set_page_config(page_title="LangGraph Summarizer", page_icon="🧠", layout="wide")
st.title("Web Article Summarizer")

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
    k = st.slider("Number of sources (k)", 1, 10, int(os.getenv("DEFAULT_K", "3")))

if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role": "user"|"assistant", "content": str, "sources": list, "errors": list}
elif st.session_state.messages and isinstance(st.session_state.messages[0], tuple):
    migrated = []
    for role, content in st.session_state.messages:
        migrated.append({"role": role, "content": content, "sources": [], "errors": []})
    st.session_state.messages = migrated

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources for this answer"):
                for i, a in enumerate(m["sources"], 1):
                    st.markdown(f"**{i}. {a.get('title','')}**  \n{a.get('url','')}")
        if m.get("errors"):
            with st.expander("Notices / Errors"):
                for e in m["errors"]:
                    st.markdown(f"- {e}")

user_topic = st.chat_input("Give me a topic (e.g., 'LLM agents vs. traditional RAG')")

def run_graph(question: str, provider: str, model_name: str, temperature: float, k: int):
    chat_model = get_chat_model(provider, model_name, temperature)
    app = build_graph(chat_model, SYNTHESIS_PROMPT)
    init_state = {"question": question, "k": k}
    return asyncio.run(app.ainvoke(init_state))

if user_topic:
    st.session_state.messages.append({"role": "user", "content": user_topic, "sources": [], "errors": []})

    with st.chat_message("assistant"):
        try:
            t0 = time.perf_counter()
            with st.status("Starting…", expanded=True) as status:
                status.write("🎯 Getting things lined up…")
                status.update(label=f"Step 1/3 • Searching {k} sources…")
                final_state = run_graph(user_topic, provider, model_name, temperature, k)

                answer = (final_state.get("final_answer") or "").strip()
                articles = final_state.get("articles", []) or []
                errors = final_state.get("errors", []) or []

                status.update(label="Step 2/3 • Summarizing sources…")
                status.update(label="Step 3/3 • Done!", state="complete")
                status.write(f"⏱️ Total time: {time.perf_counter() - t0:0.1f}s")

            if answer:
                st.markdown(answer)

            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": articles,   
                "errors": errors
            })

        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
