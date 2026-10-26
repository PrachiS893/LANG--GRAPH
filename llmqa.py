from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

class LLMState(TypedDict):
    question:str
    answer:str

def llm_qa(state:LLMState) -> LLMState:
    question=state['question']
    prompt=f"answer the following questions- {question}"
    answer=model.invoke(prompt).content
    state['answer']=answer
    return state

graph=StateGraph(LLMState)

graph.add_node('llm_qa',llm_qa)

graph.add_edge(START,'llm_qa')
graph.add_edge('llm_qa',END)

workflow=graph.compile()

question=input("Enter your question here : ")

initial_state={'question': question}
final_state= workflow.invoke(initial_state)
print("answer : ",final_state['answer'])