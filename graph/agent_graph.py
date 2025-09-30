import re
import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from retrieval.web_search import fetch_articles  
from llm.model_factory import get_chat_model  

# Setup basic logging if not already (shared with web_search.py)
logging.basicConfig(level=logging.INFO)

class ArticleSummary(BaseModel):
    """Structured schema for individual article summary with source."""
    title: str = Field(description="The title of the article")
    url: str = Field(description="The URL of the article")
    summary: str = Field(description="The summary of the article content")

class ResponseFormatter(BaseModel):
    """Structured output schema for the agent's response with multiple article summaries."""
    summaries: List[ArticleSummary] = Field(description="List of k article summaries with their sources")

def build_agent(provider: str, model_name: str, temperature: float, synthesis_prompt: str):
    """
    Builds an agent that uses tool calling to process a question and return k article summaries with sources.
    
    Args:
        provider: The model provider ("openai" or "ollama").
        model_name: The specific model name to use.
        temperature: The temperature setting for the model.
        synthesis_prompt: The prompt template for summarization guidance.
    
    Returns:
        A function that takes a question and k (number of sources) and returns a ResponseFormatter object.
    """
    chat_model = get_chat_model(provider, model_name, temperature)
    
    # Define tools
    @tool
    def web_search(query: str, k: int = 3) -> List[Dict[str, str]]:
        """Search the web and return up to k articles with title, url, and content."""
        items = fetch_articles(query, k=k) or []
        return [
            {"title": i.get("title", ""), "url": i.get("url", ""), "content": i.get("content", "")}
            for i in items if i.get("url")
        ]

    def summarize_article_direct(article: Dict[str, str], max_words: int = 200, synthesis_prompt: str = "") -> Dict[str, Any]:
        """Direct summarization (bypasses tool-calling for reliability)."""
        prompt = (
            f"{synthesis_prompt}\n\n"  # Uses the passed synthesis_prompt
            "Summarize this article in 6–8 sentences with key facts/numbers; avoid fluff.\n\n"
            f"Title: {article.get('title', '')}\nURL: {article.get('url', '')}\n\n"
            f"Content:\n{article.get('content', '')}\n\n"
            f"Limit: <= {max_words} words. Focus on factual content relevant to the topic."
        )
        
        response = chat_model.invoke([HumanMessage(content=prompt)])
        return {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "summary": response.content.strip()
        }

    # Keep tool for potential LLM use, but we won't rely on it
    @tool
    def summarize_article(article: Dict[str, str], max_words: int = 200) -> Dict[str, Any]:
        """Summarize an article in 6-8 sentences with key facts."""
        return summarize_article_direct(article, max_words, "")  # Fallback to direct

    TOOLS = [web_search, summarize_article]
    llm = chat_model.bind_tools(TOOLS)

    def agent(question: str, k: int = 3) -> ResponseFormatter:
        """
        Processes the input question using tool calling to return k article summaries with sources.
    
        Args:
            question: The user's question.
            k: Number of web search results to summarize.
    
        Returns:
            ResponseFormatter object with k ArticleSummary objects.
        """
        messages = [SystemMessage(content="You are a helpful assistant. Use tools to search and summarize articles.")]
        summaries = []
        articles = []
        iteration = 0
        max_iterations = 2  # Keep light retry if search fails entirely

        while len(summaries) < k and iteration < max_iterations:
            iteration += 1
            # Initial or retry search
            if len(articles) == 0 or len(summaries) < k:
                search_query = f"{question} additional sources" if iteration > 1 else question
                response = llm.invoke(messages + [HumanMessage(content=f"Search for articles on: {search_query} (fetch up to {k}). Use web_search tool.")])
                messages.append(AIMessage(content=response.content, tool_calls=response.tool_calls))

                if not response.tool_calls or response.tool_calls[0]["name"] != "web_search":
                    messages.append(AIMessage(content="No tool calls made. Please use the provided tools."))
                    continue

                for tool_call in response.tool_calls:
                    if tool_call["name"] == "web_search":
                        tool_args = tool_call.get("args", {})
                        search_results = web_search.invoke(tool_args)
                        new_articles = [a for a in search_results if a.get("url") and not any(s.url == a["url"] for s in summaries)]
                        articles.extend(new_articles)
                        logging.info(f"Fetched {len(new_articles)} new articles. Total articles: {len(articles)}")
                        messages.append(ToolMessage(
                            content=str(new_articles)[:8000],
                            tool_call_id=tool_call.get("id"),
                            name="web_search"
                        ))

            # Summarize fetched articles (up to remaining needed) - Direct call to bypass tool arg issues
            for article in articles[:k - len(summaries)]:
                if isinstance(article, dict) and article.get("url") and not any(s.url == article["url"] for s in summaries):
                    try:
                        # Direct invocation: No LLM tool call needed
                        summary_dict = summarize_article_direct(article, synthesis_prompt=synthesis_prompt)
                        summaries.append(ArticleSummary(
                            title=summary_dict["title"],
                            url=summary_dict["url"],
                            summary=summary_dict["summary"]
                        ))
                        logging.info(f"Summarized article: {summary_dict['title'][:50]}...")
                        
                        # Optional: Mock ToolMessage for consistency (no error if skipped)
                        messages.append(ToolMessage(
                            content=str(summary_dict)[:8000],
                            tool_call_id="direct_summary_mock",
                            name="summarize_article"
                        ))
                    except Exception as sum_err:
                        logging.error(f"Summarization failed for {article.get('title', 'unknown')}: {sum_err}")
                        # Fallback placeholder
                        summaries.append(ArticleSummary(
                            title=article.get("title", "Error"),
                            url=article.get("url", ""),
                            summary=f"Failed to summarize: {str(sum_err)[:100]}"
                        ))

        # Pad to exactly k if needed (rare now)
        while len(summaries) < k:
            summaries.append(ArticleSummary(
                title="No additional relevant article found",
                url="",
                summary="Unable to find a relevant article after multiple searches."
            ))

        logging.info(f"Final: {len(summaries)} summaries generated for k={k}")
        return ResponseFormatter(summaries=summaries[:k])
    
    return agent