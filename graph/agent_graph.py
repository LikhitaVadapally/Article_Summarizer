from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage

from retrieval.web_search import fetch_three_articles

#shape of the article
class Article(TypedDict):
    title: str
    url: str
    content: str

#global states passed betn graph nodes
class AgentState(TypedDict):
    question: str
    articles: List[Article]
    final_answer: str

#Node 1 - calls tavily and fetches 3 articles
def node_fetch_3(state: AgentState) -> AgentState:
    q = state["question"]
    state["articles"] = fetch_three_articles(q)
    return state

#takes articles and formats nicely
def _build_context_block(articles: List[Article]) -> str:
    blocks = []
    for i, a in enumerate(articles, 1):
        content = (a.get("content") or "")[:4000]
        blocks.append(f"[{i}] Title: {a.get('title','')}\nURL: {a.get('url','')}\n\n{content}\n")
    return "\n\n".join(blocks) #returns single str for llm to read

#Node 2 - Summarize
def node_summarize(chat_model, synthesis_prompt: str):
    def _inner(state: AgentState) -> AgentState:
        #creates prompt with instrctions, topic and 3 articles
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

#build the graph
def build_graph(chat_model, synthesis_prompt: str):
    graph = StateGraph(AgentState)

#adds 2 nodes
    graph.add_node("fetch_3", node_fetch_3)
    graph.add_node("summarize", node_summarize(chat_model, synthesis_prompt))

#defines the flow
    graph.set_entry_point("fetch_3")
    graph.add_edge("fetch_3", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
