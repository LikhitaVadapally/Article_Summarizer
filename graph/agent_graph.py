from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage
from retrieval.web_search import fetch_articles

class Article(TypedDict):
    title: str
    url: str
    content: str

class AgentState(TypedDict, total=False):
    question: str
    k: int
    articles: List[Article]
    per_article_summaries: List[Dict]  # [{title,url,summary}]
    final_answer: str
    errors: List[str]

# Node 1 — search
def node_fetch(state: AgentState) -> AgentState:
    q = state["question"]
    k = state.get("k", 3)
    items = fetch_articles(q, k=k)
    if not items:
        errs = state.get("errors", [])
        errs.append("No results from Tavily (or retries exhausted).")
        state["errors"] = errs
    state["articles"] = items
    return state

# summarizer agent
def article_summary_prompt(article: Article) -> str:
    return (
        "You are a careful research assistant.\n"
        "Summarize the article below in 6–8 sentences focusing on key facts, numbers, and claims.\n"
        "Avoid opinions, avoid fluff. If facts are uncertain, say so.\n\n"
        f"Title: {article.get('title','')}\n"
        f"URL: {article.get('url','')}\n\n"
        f"Content:\n{article.get('content','')}\n"
    )

# Node 2 — summarize each article with a dedicated agent
def node_summarize_each(chat_model):
    def _inner(state: AgentState) -> AgentState:
        summaries: List[Dict] = []
        for a in state.get("articles", []):
            prompt = article_summary_prompt(a)
            msg = chat_model.invoke([HumanMessage(content=prompt)])
            summaries.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "summary": msg.content.strip(),
            })
        state["per_article_summaries"] = summaries
        return state
    return _inner

def build_summaries_block(items: List[Dict]) -> str:
    blocks = []
    for i, s in enumerate(items, 1):
        blocks.append(
            f"[{i}] {s.get('title','')}\nURL: {s.get('url','')}\n\n{s.get('summary','')}\n"
        )
    return "\n\n".join(blocks)

# Node 3 — synthesize a final answer from the per-article summaries
def node_synthesize(chat_model, synthesis_prompt: str):
    def _inner(state: AgentState) -> AgentState:
        summaries = state.get("per_article_summaries", [])
        prompt = (
            f"{synthesis_prompt}\n\n"
            f"User topic:\n{state['question']}\n\n"
            f"Source summaries:\n{build_summaries_block(summaries)}\n"
        )
        msg = chat_model.invoke([HumanMessage(content=prompt)])
        state["final_answer"] = msg.content
        return state
    return _inner

# Build the graph
def build_graph(chat_model, synthesis_prompt: str):
    graph = StateGraph(AgentState)
    graph.add_node("fetch", node_fetch)
    graph.add_node("summarize_each", node_summarize_each(chat_model))
    graph.add_node("synthesize", node_synthesize(chat_model, synthesis_prompt))
    graph.set_entry_point("fetch")
    #defines the flow
    graph.add_edge("fetch", "summarize_each")
    graph.add_edge("summarize_each", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()
