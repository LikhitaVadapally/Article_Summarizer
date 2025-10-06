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
    """Structured schema for individual article summary with source and relevance score."""
    title: str = Field(description="The title of the article")
    url: str = Field(description="The URL of the article")
    summary: str = Field(description="The summary of the article content")
    # article_relevance_score: float = Field(default=-1.0, description="Relevance of article to query (0.0–1.0)")  # Commented out: Removed article relevance scoring
    summary_relevance_score: float = Field(default=-1.0, description="Faithfulness of summary to article (0.0–1.0)")

class ResponseFormatter(BaseModel):
    """Structured output schema for the agent's response with multiple article summaries."""
    summaries: List[ArticleSummary] = Field(description="List of k article summaries with their sources and faithfulness scores")

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
    def evaluate_relevance(mode: str, text1: str, text2: str) -> Dict[str, Any]:
        """Evaluate faithfulness of summary to article (mode: 'summary_to_article').
        Returns {'score': 0.0–1.0, 'rationale': str}."""
        if mode != "summary_to_article":
            raise ValueError(f"Invalid mode: {mode}")
        prompt = (
            f"You are a JSON expert. Output ONLY valid JSON: {{\"score\": <float 0.0-1.0>, \"rationale\": \"<brief explanation>\"}}\n"
            f"Mode: summary_to_article\n"
            f"Text1 (source content): {text1[:2000]}\n"  # Increase to 2000; add "source" not "article"
            f"Text2 (summary): {text2[:2000]}\n"
            "Score how well the summary captures key facts from the source (even if it's a video description, snippet, or short text). "
            "High score (0.8+) if facts/numbers match without hallucination. Default to 0.6+ for partial matches."
            "Ensure keys/values are double-quoted. No extra text."
            )
        
# In fallback/validate: Change default to 0.6 if rationale mentions "video" or "description"
        if "video" in rationale.lower() or "description" in rationale.lower():
            score = max(score, 0.6)  # Leniency for media
            
        try:
            response = chat_model.invoke([HumanMessage(content=prompt)])
            raw_output = response.content.strip()
            logging.info(f"Raw eval output for '{text1[:50]}...': {raw_output[:200]}...")  # Debug log
            # Try json.loads
            result = json.loads(raw_output)
            score = float(result.get('score', 0.0))
            rationale = result.get('rationale', 'No rationale')
            # Validate score range
            if not 0.0 <= score <= 1.0:
                logging.warning(f"Invalid score {score}; defaulting to 0.5")
                score = 0.5
            return {"score": score, "rationale": rationale}
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode failed: {e}. Raw: {raw_output[:500]}")
            # Fallback: Regex extract
            try:
                score_match = re.search(r'"score"\s*:\s*([0-9.]+)', raw_output)
                score = float(score_match.group(1)) if score_match and 0.0 <= float(score_match.group(1)) <= 1.0 else 0.5
                rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', raw_output)
                rationale = rationale_match.group(1) if rationale_match else f"Parse error: {str(e)[:100]}"
                logging.warning(f"JSON fallback used: score={score}, rationale={rationale}")
                return {"score": score, "rationale": rationale}
            except:
                return {"score": 0.5, "rationale": f"Regex fallback failed: {str(e)[:100]}"}
        except Exception as e:
            logging.error(f"Relevance evaluation failed: {e}")
            return {"score": 0.5, "rationale": f"Unexpected error: {str(e)[:100]}"}

    def summarize_article_direct(article: Dict[str, str], max_words: int = 200, synthesis_prompt: str = "") -> Dict[str, Any]:
        """Direct summarization (bypasses tool-calling for reliability)."""
        prompt = (
            f"{synthesis_prompt}\n\n"
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

    @tool
    def summarize_article(article: Dict[str, str], max_words: int = 200) -> Dict[str, Any]:
        """Summarize an article in 6-8 sentences with key facts."""
        return summarize_article_direct(article, max_words, synthesis_prompt)

    TOOLS = [web_search, evaluate_relevance, summarize_article]
    llm = chat_model.bind_tools(TOOLS)

    def agent(question: str, k: int = 3) -> ResponseFormatter:
        """
        Processes the input question using tool calling to return k article summaries with sources and faithfulness scores.
    
        Args:
            question: The user's question.
            k: Number of web search results to summarize.
    
        Returns:
            ResponseFormatter object with k ArticleSummary objects.
        """
        messages = [SystemMessage(content=(
            "You are a helpful assistant. Use tools to search and summarize articles.\n"
            "- web_search(query,k) -> [{title,url,content}]\n"
            "- evaluate_relevance(mode,text1,text2) -> {score,rationale} (mode: 'summary_to_article')\n"
            "- summarize_article(article,max_words)\n"
            "After summarize_article, evaluate summary faithfulness to article (threshold: 0.7). Retry if score <0.7."
        ))]
        summaries = []
        articles = []
        iteration = 0
        max_iterations = 3
        total_evals = 0
        max_evals = 10

        while len(summaries) < k and iteration < max_iterations and total_evals < max_evals:
            iteration += 1
            # Fetch articles if needed
            if len(articles) < k - len(summaries):
                search_query = f"{question} additional sources" if iteration > 1 else question
                response = llm.invoke(messages + [HumanMessage(content=f"Search for articles on: {search_query} (fetch up to {k}). Use web_search tool.")])
                messages.append(AIMessage(content=response.content, tool_calls=response.tool_calls))

                if not response.tool_calls or response.tool_calls[0]["name"] != "web_search":
                    messages.append(AIMessage(content="No tool calls made. Please use the web_search tool."))
                    continue

                for tool_call in response.tool_calls:
                    if tool_call["name"] == "web_search":
                        tool_args = tool_call.get("args", {})
                        search_results = web_search.invoke(tool_args)
                        # article relevance evaluation
                        # scored_articles = []
                        # for article in search_results:
                        #     if not article.get("url") or any(s.url == article["url"] for s in summaries):
                        #         continue
                        #     eval_result = evaluate_relevance.invoke({
                        #         "mode": "article_to_query",
                        #         "text1": question,
                        #         "text2": article["content"]
                        #     })
                        #     article_relevance = eval_result["score"]
                        #     logging.info(f"Article '{article['title'][:50]}...' relevance: {article_relevance:.2f}")
                        #     if article_relevance >= 0.5:  # Threshold
                        #         article["relevance_score"] = article_relevance
                        #         scored_articles.append(article)
                        #     else:
                        #         logging.warning(f"Discarded article '{article['title'][:50]}...' (relevance: {article_relevance:.2f})")
                        # new_articles = [a for a in scored_articles if a.get("url") and not any(s.url == a["url"] for s in summaries)]
                        new_articles = [a for a in search_results if a.get("url") and not any(s.url == article["url"] for s in summaries)]
                        articles.extend(new_articles)
                        logging.info(f"Fetched {len(new_articles)} new articles. Total articles: {len(articles)}")
                        messages.append(ToolMessage(
                            content=str(new_articles)[:8000],
                            tool_call_id=tool_call.get("id"),
                            name="web_search"
                        ))

            # Summarize fetched articles
            for article in articles[:k - len(summaries)]:
                if isinstance(article, dict) and article.get("url") and not any(s.url == article["url"] for s in summaries):
                    try:
                        summary_dict = summarize_article_direct(article, synthesis_prompt=synthesis_prompt)
                        total_evals += 1
                        eval_result = evaluate_relevance.invoke({
                            "mode": "summary_to_article",
                            "text1": article["content"],
                            "text2": summary_dict["summary"]
                        })
                        summary_relevance = eval_result["score"]
                        logging.info(f"Summary for '{summary_dict['title'][:50]}...' faithfulness: {summary_relevance:.2f}")
                        if summary_relevance < 0.7:
                            logging.warning(f"Low-faithfulness summary for '{summary_dict['title'][:50]}...'; retrying...")
                            total_evals += 1
                            summary_dict = summarize_article_direct(article, synthesis_prompt=synthesis_prompt)
                            eval_result = evaluate_relevance.invoke({
                                "mode": "summary_to_article",
                                "text1": article["content"],
                                "text2": summary_dict["summary"]
                            })
                            summary_relevance = eval_result["score"]
                            logging.info(f"Retry summary faithfulness: {summary_relevance:.2f}")
                        if summary_relevance >= 0.7:
                            summaries.append(ArticleSummary(
                                title=summary_dict["title"],
                                url=summary_dict["url"],
                                summary=summary_dict["summary"],
                                # article_relevance_score=article.get("relevance_score", -1.0),  # Commented out
                                summary_relevance_score=summary_relevance
                            ))
                            logging.info(f"Summarized article: {summary_dict['title'][:50]}...")
                            messages.append(ToolMessage(
                                content=str(summary_dict)[:8000],
                                tool_call_id="direct_summary_mock",
                                name="summarize_article"
                            ))
                        else:
                            logging.warning(f"Discarded low-faithfulness summary for '{summary_dict['title'][:50]}...' (score: {summary_relevance:.2f})")
                    except Exception as sum_err:
                        logging.error(f"Summarization failed for {article.get('title', 'unknown')}: {sum_err}")
                        summaries.append(ArticleSummary(
                            title=article.get("title", "Error"),
                            url=article.get("url", ""),
                            summary=f"Failed to summarize: {str(sum_err)[:100]}",
                            # article_relevance_score=article.get("relevance_score", -1.0),  # Commented out
                            summary_relevance_score=0.0
                        ))

        # Pad to exactly k if needed
        while len(summaries) < k:
            summaries.append(ArticleSummary(
                title="No additional article found",
                url="",
                summary="Unable to find a relevant article after multiple searches.",
                # article_relevance_score=0.0,  # Commented out
                summary_relevance_score=0.0
            ))

        logging.info(f"Final: {len(summaries)} summaries generated for k={k}. Total evals: {total_evals}")
        return ResponseFormatter(summaries=summaries[:k])
    
    return agent