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

# -------------------------
# MODELS
# -------------------------

model = ChatOpenAI(model = 'gpt-4o-mini')

class ResearchDecision(BaseModel):
    need_research: bool
    queries: List[str] = []

class Task(BaseModel):
    id: str
    title: str
    brief: str = Field(description="What to cover")

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]

class State(TypedDict, total=False):
    topic: str
    plan: Plan
    research: str
    need_research: bool
    queries: List[str]
    sections: Annotated[List[str], operator.add]
    final: str

model = ChatOpenAI(model="gpt-4.1-mini")

# -------------------------
# ROUTER
# -------------------------

def router(state: State) -> dict:
    topic = state["topic"]

    decision = model.with_structured_output(ResearchDecision).invoke(
        [
            SystemMessage(
                content=(
                    "Decide if research is needed. "
                    "Return need_research=true only if factual data or recent updates are required. "
                    "If true, return 3–10 specific search queries."
                )
            ),
            HumanMessage(content=f"Topic: {topic}")
        ]
    )

    return {
        "need_research": decision.need_research,
        "queries": decision.queries
    }

# -------------------------
# RESEARCH
# -------------------------

def research(state: State):
    queries = state.get("queries", [])
    if not queries:
        return {"research": ""}

    tavily = TavilySearch(max_results=3)

    chunks = []
    for q in queries:
        results = tavily.invoke({"query": q})
        texts = [r["content"] for r in results.get("results", [])]
        combined = "\n".join(texts)
        chunks.append(f"### Research for {q}\n{combined}")

    return {"research": "\n\n".join(chunks)}

# -------------------------
# ORCHESTRATOR
# -------------------------

def orchestrator(state: State):
    topic = state["topic"]

    plan = model.with_structured_output(Plan).invoke(
        [
            SystemMessage(content="Create a blog plan with 5–7 sections."),
            HumanMessage(content=f"Topic: {topic}"),
            HumanMessage(content=f"Research:\n{state.get('research','')}")
        ]
    )

    return {"plan": plan}

# -------------------------
# FANOUT
# -------------------------

def fanout(state: State):
    return [
        Send("worker", {
            "task": task,
            "topic": state["topic"],
            "plan": state["plan"],
            "research": state.get("research", "")
        })
        for task in state["plan"].tasks
    ]

# -------------------------
# WORKER
# -------------------------

def worker(payload: dict) -> dict:
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]
    research = payload.get("research", "")

    section_md = model.invoke(
        [
            SystemMessage(content="Write one clean Markdown section."),
            HumanMessage(
                content=(
                    f"Blog: {plan.blog_title}\n"
                    f"Topic: {topic}\n"
                    f"Section: {task.title}\n"
                    f"Brief: {task.brief}\n\n"
                    f"Research:\n{research}\n\n"
                    "Return only Markdown."
                )
            )
        ]
    ).content.strip()

    return {"sections": [section_md]}

# -------------------------
# REDUCER
# -------------------------

def reducer(state: State) -> dict:
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"])

    final_md = f"# {title}\n\n{body}"

    Path(f"{title.lower().replace(' ','_')}.md").write_text(final_md)
    return {"final": final_md}

# -------------------------
# GRAPH BUILD
# -------------------------

g = StateGraph(State)
g.add_node("router", router)
g.add_node("research", research)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

def route_next(state: State) -> str:
    return "research" if state.get("need_research") else "orchestrator"

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, ["research", "orchestrator"])
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

print(app.invoke({"topic": "Self attention"}))