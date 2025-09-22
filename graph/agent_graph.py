from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage

from retrieval.web_search import fetch_three_articles

class Article(TypedDict):
    title: str
    url: str
    content: str

class AgentState(TypedDict):
    question: str
    articles: List[Article]
    final_answer: str

def node_fetch_3(state: AgentState) -> AgentState:
    q = state["question"]
    state["articles"] = fetch_three_articles(q)
    return state

def _build_context_block(articles: List[Article]) -> str:
    blocks = []
    for i, a in enumerate(articles, 1):
        content = (a.get("content") or "")[:6000]
        blocks.append(f"[{i}] Title: {a.get('title','')}\nURL: {a.get('url','')}\n\n{content}\n")
    return "\n\n".join(blocks)

def node_summarize(chat_model, synthesis_prompt: str):
    def _inner(state: AgentState) -> AgentState:
        ctx = _build_context_block(state.get("articles", []))
        prompt = (
            f"{synthesis_prompt}\n\n"
            f"User topic:\n{state['question']}\n\n"
            f"Articles:\n{ctx}\n"
        )
        msg = chat_model.invoke([HumanMessage(content=prompt)])
        state["final_answer"] = msg.content
        return state
    return _inner

def build_graph(chat_model, synthesis_prompt: str):
    graph = StateGraph(AgentState)

    graph.add_node("fetch_3", node_fetch_3)
    graph.add_node("summarize", node_summarize(chat_model, synthesis_prompt))

    graph.set_entry_point("fetch_3")
    graph.add_edge("fetch_3", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
