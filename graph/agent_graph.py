from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from retrieval.web_search import fetch_articles  
from llm.model_factory import get_chat_model  

class ArticleSummary(BaseModel):
    """Structured schema for individual article summary with source."""
    title: str = Field(description="The title of the article")
    url: str = Field(description="The URL of the article")
    summary: str = Field(description="The summary of the article content")
    relevance_score: float = Field(description="Relevance score of the article to the query (0-1)")

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

    @tool
    def evaluate_relevance(article: Dict[str, str], question: str) -> float:
        """Evaluate the relevance of an article to the query (0-1 score)."""
        prompt = (
            "Evaluate how relevant the following article is to the query: '{}'\n\n"
            "Title: {}\nContent: {}\n\n"
            "Return a relevance score between 0 (irrelevant) and 1 (highly relevant) as a float."
        ).format(question, article.get("title", ""), article.get("content", "")[:500])
        
        response = chat_model.invoke([HumanMessage(content=prompt)])
        try:
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5  # Default score if parsing fails

    @tool
    def summarize_article(article: Dict[str, str], max_words: int = 200) -> Dict[str, Any]:
        """Summarize an article in 6-8 sentences with key facts."""
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

    TOOLS = [web_search, evaluate_relevance, summarize_article]
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
        messages = [
            SystemMessage(content=(
                "You are a web research agent. Use the provided tools to fetch, evaluate, and summarize articles. "
                "Goal: Return exactly k article summaries with relevance scores. "
                "Steps: 1) Use web_search to fetch articles. 2) Use evaluate_relevance to score each article. "
                "3) Select the top k articles by relevance. 4) Use summarize_article for each selected article. "
                "If fewer than k relevant articles are found, search again with a modified query."
            )),
            HumanMessage(content=f"Topic: {question}\nFetch and summarize {k} relevant articles.")
        ]
        
        summaries = []
        articles = []
        max_iterations = 3
        iteration = 0

        while len(summaries) < k and iteration < max_iterations:
            iteration += 1
            response = llm.invoke(messages)
            messages.append(AIMessage(content=response.content, tool_calls=response.tool_calls))

            if not response.tool_calls:
                messages.append(AIMessage(content="No tool calls made. Please use the provided tools."))
                continue

            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                call_id = tool_call.get("id", f"{tool_name}-{iteration}")

                try:
                    if tool_name == "web_search":
                        search_results = web_search.invoke(tool_args)
                        articles.extend(search_results)
                        messages.append(ToolMessage(
                            content=str(search_results)[:8000],
                            tool_call_id=call_id,
                            name=tool_name
                        ))

                    elif tool_name == "evaluate_relevance":
                        score = evaluate_relevance.invoke(tool_args)
                        article = tool_args.get("article", {})
                        article["relevance_score"] = score
                        messages.append(ToolMessage(
                            content=str(score),
                            tool_call_id=call_id,
                            name=tool_name
                        ))

                    elif tool_name == "summarize_article":
                        summary = summarize_article.invoke(tool_args)
                        summaries.append(ArticleSummary(
                            title=summary["title"],
                            url=summary["url"],
                            summary=summary["summary"],
                            relevance_score=tool_args.get("article", {}).get("relevance_score", 0.5)
                        ))
                        messages.append(ToolMessage(
                            content=str(summary)[:8000],
                            tool_call_id=call_id,
                            name=tool_name
                        ))

                except Exception as e:
                    messages.append(ToolMessage(
                        content=f"Error in {tool_name}: {str(e)}",
                        tool_call_id=call_id,
                        name=tool_name
                    ))

            # Filter and sort articles by relevance
            articles = sorted(
                [a for a in articles if a.get("relevance_score", 0) > 0.3],
                key=lambda x: x.get("relevance_score", 0),
                reverse=True
            )[:k]

            # If needed, summarize remaining articles
            for article in articles[:k - len(summaries)]:
                if article.get("url") and not any(s.url == article["url"] for s in summaries):
                    response = llm.invoke(messages + [HumanMessage(content=(
                        f"Summarize this article:\nTitle: {article['title']}\nURL: {article['url']}\n"
                        f"Content: {article['content'][:500]}"
                    ))])
                    if response.tool_calls and response.tool_calls[0]["name"] == "summarize_article":
                        messages.append(AIMessage(content=response.content, tool_calls=response.tool_calls))

            # If still short, trigger another search
            if len(summaries) < k:
                messages.append(HumanMessage(content=(
                    f"Found only {len(summaries)} summaries. "
                    f"Search again for {k - len(summaries)} more articles with query: '{question} additional information'."
                )))

        # Ensure exactly k summaries (fill with placeholders if necessary)
        while len(summaries) < k:
            summaries.append(ArticleSummary(
                title="No additional relevant article found",
                url="",
                summary="Unable to find a relevant article after multiple searches.",
                relevance_score=0.0
            ))

        return ResponseFormatter(summaries=summaries[:k])
    
    return agent