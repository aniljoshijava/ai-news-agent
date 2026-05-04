"""
AI News Intelligence Agent - LangGraph Version
================================================
A real multi-step agentic pipeline using LangGraph.

Install dependencies:
    pip install langgraph langchain langchain-google-genai requests flask flask-cors

Run:
    python agent.py
"""

import os
import json
import requests
from typing import TypedDict, List, Annotated
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ─────────────────────────────────────────────
# 1. AGENT STATE
# This is the memory/state shared across all nodes
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    query: str                      # User's original query
    search_queries: List[str]       # List of search queries agent decides to run
    search_results: List[dict]      # Raw results from Serper
    articles: List[dict]            # Cleaned articles
    needs_more_search: bool         # Agent decides if more searching needed
    additional_query: str           # Agent's follow-up search query
    report: str                     # Final generated report
    steps_log: List[str]            # Log of what agent did
    serper_key: str                 # API keys passed in
    gemini_key: str


# ─────────────────────────────────────────────
# 2. NODE FUNCTIONS
# Each node is one step the agent takes
# ─────────────────────────────────────────────

def plan_searches(state: AgentState) -> AgentState:
    """
    NODE 1: Agent THINKS and plans what to search for.
    This is what makes it a real agent - it decides its own search strategy.
    """
    log = state.get("steps_log", [])
    log.append("🧠 Agent is planning search strategy...")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=state["gemini_key"],
        temperature=0.3
    )

    response = llm.invoke([
        SystemMessage(content="You are a research planning agent. Given a topic, generate 3 targeted search queries to find comprehensive, latest news. Return ONLY a JSON array of 3 strings. No explanation. Example: [\"query1\", \"query2\", \"query3\"]"),
        HumanMessage(content=f"Topic: {state['query']}\n\nGenerate 3 different search queries to find the most complete and recent news about this topic.")
    ])

    try:
        raw = response.content.strip()
        # Clean up markdown if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        queries = json.loads(raw)
        if not isinstance(queries, list):
            queries = [state["query"]]
    except:
        queries = [state["query"], f"latest {state['query']} 2025", f"{state['query']} news update"]

    log.append(f"📋 Planned {len(queries)} search queries:")
    for i, q in enumerate(queries, 1):
        log.append(f"   {i}. {q}")

    return {**state, "search_queries": queries, "steps_log": log}


def execute_searches(state: AgentState) -> AgentState:
    """
    NODE 2: Agent SEARCHES the internet using all planned queries.
    """
    log = state.get("steps_log", [])
    log.append("🔍 Executing searches with Serper API...")

    all_results = []
    seen_urls = set()

    for query in state["search_queries"]:
        log.append(f"   Searching: '{query}'")
        try:
            res = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": state["serper_key"],
                    "Content-Type": "application/json"
                },
                json={"q": query, "num": 5, "gl": "us", "hl": "en"},
                timeout=10
            )
            data = res.json()
            for item in data.get("organic", []):
                url = item.get("link", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "url": url,
                        "source": item.get("source", ""),
                        "query_used": query
                    })
        except Exception as e:
            log.append(f"   ⚠️ Search error: {str(e)}")

    log.append(f"✅ Found {len(all_results)} unique articles total")

    return {**state, "search_results": all_results, "articles": all_results, "steps_log": log}


def evaluate_results(state: AgentState) -> AgentState:
    """
    NODE 3: Agent EVALUATES if it has enough information.
    This is the decision-making part of the agent.
    """
    log = state.get("steps_log", [])
    log.append("🤔 Agent evaluating search results quality...")

    articles = state.get("articles", [])

    if len(articles) < 4:
        log.append("⚠️ Not enough results. Agent deciding to search more...")
        return {**state, "needs_more_search": True,
                "additional_query": f"{state['query']} recent developments breaking news",
                "steps_log": log}

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=state["gemini_key"],
        temperature=0.1
    )

    snippets = "\n".join([f"- {a['title']}: {a['snippet'][:100]}" for a in articles[:8]])

    response = llm.invoke([
        SystemMessage(content="You are evaluating search results. Answer with ONLY valid JSON: {\"sufficient\": true/false, \"reason\": \"brief reason\", \"additional_query\": \"query if needed\"}"),
        HumanMessage(content=f"Topic: {state['query']}\n\nArticles found:\n{snippets}\n\nAre these results sufficient for a comprehensive report?")
    ])

    try:
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        evaluation = json.loads(raw)
        needs_more = not evaluation.get("sufficient", True)
        additional = evaluation.get("additional_query", "")
        reason = evaluation.get("reason", "")
        log.append(f"📊 Evaluation: {'Need more data' if needs_more else 'Sufficient data'} — {reason}")
    except:
        needs_more = False
        additional = ""

    return {**state, "needs_more_search": needs_more,
            "additional_query": additional, "steps_log": log}


def search_more(state: AgentState) -> AgentState:
    """
    NODE 4: Agent searches AGAIN if it decided it needs more info.
    This is the self-correcting loop.
    """
    log = state.get("steps_log", [])
    query = state.get("additional_query", state["query"] + " latest news")
    log.append(f"🔄 Agent doing additional search: '{query}'")

    try:
        res = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": state["serper_key"],
                "Content-Type": "application/json"
            },
            json={"q": query, "num": 5},
            timeout=10
        )
        data = res.json()
        new_articles = []
        existing_urls = {a["url"] for a in state.get("articles", [])}

        for item in data.get("organic", []):
            url = item.get("link", "")
            if url not in existing_urls:
                new_articles.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "url": url,
                    "source": item.get("source", ""),
                    "query_used": query
                })

        all_articles = state.get("articles", []) + new_articles
        log.append(f"✅ Found {len(new_articles)} additional articles. Total: {len(all_articles)}")
        return {**state, "articles": all_articles, "steps_log": log}

    except Exception as e:
        log.append(f"⚠️ Additional search failed: {str(e)}")
        return {**state, "steps_log": log}


def generate_report(state: AgentState) -> AgentState:
    """
    NODE 5: Agent WRITES the final comprehensive report.
    """
    log = state.get("steps_log", [])
    log.append("✍️ Agent generating intelligence report with Gemini...")

    articles = state.get("articles", [])
    context = "\n\n".join([
        f"[{i+1}] **{a['title']}**\nSource: {a.get('source','')}\nURL: {a['url']}\nSummary: {a['snippet']}"
        for i, a in enumerate(articles[:12])
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=state["gemini_key"],
        temperature=0.7
    )

    response = llm.invoke([
        SystemMessage(content="""You are an expert AI research journalist writing intelligence reports. 
Write comprehensive, analytical reports with clear structure. Use markdown formatting.
Be specific, cite article numbers like [1], [2], etc. Minimum 700 words."""),
        HumanMessage(content=f"""Write a comprehensive intelligence report about: "{state['query']}"

Source Articles:
{context}

Structure your report with these sections:
## Executive Summary
## Key Developments  
## Emerging Trends
## Notable Players
## Analysis & Implications
## What to Watch Next

Be analytical, specific, and reference articles by number [1], [2] etc.""")
    ])

    report = response.content
    log.append(f"✅ Report generated — {len(report)} characters")
    log.append("🎉 Agent pipeline complete!")

    return {**state, "report": report, "steps_log": log}


# ─────────────────────────────────────────────
# 3. BUILD THE LANGGRAPH
# Connect all nodes into a graph
# ─────────────────────────────────────────────

def should_search_more(state: AgentState) -> str:
    """Router function — decides which node to go to next"""
    if state.get("needs_more_search", False):
        return "search_more"
    return "generate_report"


def build_agent_graph():
    """Build and compile the LangGraph agent"""
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("plan_searches", plan_searches)
    graph.add_node("execute_searches", execute_searches)
    graph.add_node("evaluate_results", evaluate_results)
    graph.add_node("search_more", search_more)
    graph.add_node("generate_report", generate_report)

    # Connect nodes with edges
    graph.set_entry_point("plan_searches")
    graph.add_edge("plan_searches", "execute_searches")
    graph.add_edge("execute_searches", "evaluate_results")

    # Conditional edge — agent decides whether to search more
    graph.add_conditional_edges(
        "evaluate_results",
        should_search_more,
        {
            "search_more": "search_more",
            "generate_report": "generate_report"
        }
    )

    graph.add_edge("search_more", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 4. FLASK API SERVER
# Serves the agent to the frontend HTML
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

@app.route("/run-agent", methods=["POST"])
def run_agent():
    data = request.json
    query = data.get("query", "")
    serper_key = data.get("serper_key", "")
    gemini_key = data.get("gemini_key", "")

    if not all([query, serper_key, gemini_key]):
        return jsonify({"error": "Missing query, serper_key, or gemini_key"}), 400

    try:
        agent = build_agent_graph()

        initial_state = AgentState(
            query=query,
            search_queries=[],
            search_results=[],
            articles=[],
            needs_more_search=False,
            additional_query="",
            report="",
            steps_log=[],
            serper_key=serper_key,
            gemini_key=gemini_key
        )

        final_state = agent.invoke(initial_state)

        return jsonify({
            "report": final_state["report"],
            "articles": final_state["articles"][:12],
            "steps_log": final_state["steps_log"],
            "search_queries": final_state["search_queries"],
            "total_articles": len(final_state["articles"])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running", "agent": "LangGraph AI News Agent"})


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  LangGraph AI News Agent")
    print("  Server running at http://localhost:5000")
    print("  Open index.html in your browser")
    print("="*50 + "\n")
    port = os.environ.get("PORT")
    app.run(host="0.0.0.0",port=port)
