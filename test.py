from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from typing import Annotated, List, TypedDict, Optional, Literal
from langgraph.types import Send
#from __future__ import annotations
import operator
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path

from langchain_tavily import TavilySearch
load_dotenv()

tool = TavilySearch(max_results=2,tavily_api_key="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")
results = tool.invoke({"query": 'Latest updates on ChatGPT'})

print(results)
# print(results['query'])
# print(results['results'][0]['content'])

# normalized: List[dict] = []
# for r in results:
#     normalized.append(
#             {
#                 "title": r.get("title") or "",
#                 "url": r.get("url") or "",
#                 "snippet": r.get("content") or r.get("snippet") or "",
#                 "published_at": r.get("published_date") or r.get("published_at"),
#                 "source": r.get("source"),
#             }
#         )

# print(normalized)

