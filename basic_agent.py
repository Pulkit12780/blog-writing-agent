from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel, Field
from typing import Annotated, List, TypedDict
from langgraph.types import Send
#from __future__ import annotations
import operator
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path

load_dotenv()


class Task(BaseModel):
    id: str
    title: str
    brief: str = Field(description = 'What to cover')

class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]


class State(TypedDict):
    topic: str
    plan: Plan
    #reducer -> 
    sections : Annotated[List[str], operator.add]
    final: str


#model
model = ChatOpenAI(model = 'gpt-4.1-mini')

#orchestrator node
def orchestrator(state: State):
    topic = state['topic']
    plan = model.with_structured_output(Plan).invoke(
        [
            SystemMessage(content="Create a blog plan with 5-7 sections on the following topic."),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )
    return {"plan": plan}

#fanout: The function returns a list of Send() instructions. Each Send() means: “Send this message to the worker node in the graph.”
def fanout(state: State):
    return [ Send("worker", {'task': task, 'topic': state['topic'], 'plan': state['plan']}) for task in state['plan'].tasks]


#worker node
def worker(payload: dict) -> dict:

    # payload contains what we sent
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    blog_title = plan.blog_title

    section_md = model.invoke(
        [
            SystemMessage(content="Write one clean Markdown section."),
            HumanMessage(
                content=(
                    f"Blog: {blog_title}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Brief: {task.brief}\n\n"
                    "Return only the section content in Markdown."
                )
            ),
        ]
    ).content.strip()

    return {"sections": [section_md]}


#collating content
def reducer(state: State) -> dict:
    
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}\n"

    # ---- save to file ----
    filename = title.lower().replace(" ", "_") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")

    return {"final": final_md}

#graph

g = StateGraph(State)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker)
g.add_node("reducer", reducer)

g.add_edge(START, "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

print(app.invoke({'topic': 'Self attention' }))


#start
#orchestration
#plan object -> title, list of tasks
#task object -> title, id, description about one section of the blog


#worker node: orchestrator - worker flow. dynaically n worker nodes will be created which will run in parallel
#reducer -> sticth and create md file








