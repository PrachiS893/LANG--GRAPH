from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

class BlogState(TypedDict):
    topic:str
    outline:str
    content:str

def create_outline(state:BlogState) -> BlogState:
    topic=state['topic']
    prompt=f"generate a detailed outline for a blog on the topic -{topic}"
    outline=model.invoke(prompt).content
    state['outline']=outline
    return state

def create_blog(state:BlogState) -> BlogState:
    topic=state['topic']
    outline=state['outline']
    prompt=f"write a detailed blog on the topic - {topic} using the following outline \n {outline}"
    content=model.invoke(prompt).content
    state['content']=content
    return state

graph = StateGraph(BlogState)

graph.add_node('create_outline',create_outline)
graph.add_node('create_blog',create_blog)

graph.add_edge(START,'create_outline')
graph.add_edge('create_outline','create_blog')
graph.add_edge('create_blog',END)

workflow=graph.compile()

topic=input("Enter your topic here: ")

initial_state={'topic':topic}
final_state=workflow.invoke(initial_state)
print(final_state['content'])