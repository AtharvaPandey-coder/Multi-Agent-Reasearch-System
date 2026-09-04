Multi-Tool Research Assistant

An agentic research assistant built with LangChain that answers questions by intelligently routing across multiple tools — searching provided documents first, and falling back to the web or Wikipedia only when needed.

Overview

Instead of relying on a single LLM call, this project uses an agent architecture that decides, step by step, which tool to invoke to answer a query. The agent is document-first: it checks uploaded/indexed documents before reaching out to external sources, making answers more grounded and reducing unnecessary web calls.

How It Works

Agent Core — Built using LangChain's create_agent, which lets the LLM reason about which tool to call, call it, observe the result, and decide the next step (or produce a final answer).
Tool Routing (Groq) — Groq is used as the fast, low-latency LLM for the agent's tool-routing/decision step — i.e., figuring out which tool to call and how to call it.
Final Answer Generation (Gemini) — Once the agent has gathered enough information from its tools, Gemini is used to synthesize the final, coherent answer for the user.
Tools available to the agent:
document_search — searches over the user's provided/indexed documents (primary source of truth)
web_search — general web search, used as a fallback when documents don't have the answer
wikipedia — quick factual/encyclopedic lookups
Document-first, web-fallback logic — The agent always attempts document_search before reaching for web_search, so it prioritizes grounded, user-provided context over generic web results.

Architecture Flow

User Query → Agent (routing via Groq) → document_search (found? → Final Answer) → not found → web_search / wikipedia as needed → Final Answer synthesis (Gemini) → Response to User

Tech Stack

LangChain — agent framework (create_agent)
Groq (e.g. llama-3.3-70b-versatile) — tool-routing / decision-making LLM
Google Gemini — final answer generation
Python

Example Usage

Query: "What does the uploaded report say about Q3 revenue?"
→ Agent checks document_search first → Answer grounded in the uploaded document.

Query: "Who is the current CEO of Company X?"
→ document_search finds nothing relevant → falls back to web_search → Answer from web.

Key Learnings / Highlights

Implemented multi-tool agent routing rather than a single-shot LLM call
Designed a document-first, web-fallback strategy to keep answers grounded
Combined two LLM providers for different stages of the pipeline (fast routing vs. quality final synthesis)

Future Improvements

Add conversational memory across turns
Expand toolset (e.g., code execution, calculator)
Add source citations in the final answer
