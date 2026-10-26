from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
import operator 

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

class EvaluationSchema(BaseModel):
    feedback:str = Field(description='detailed feedback for the essay')
    score:int = Field(description="score out of 10", ge=0, le=10)

parser=PydanticOutputParser(pydantic_object=EvaluationSchema)

class UPSCState(TypedDict):
    essay:str
    language_feedback:str
    analysis_feedback:str
    clarity_feedback:str
    overall_feedback:str
    individual_scores: Annotated[list[int],operator.add]
    avg_score:float

def evaluate_language(state:UPSCState):
    prompt=f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output=model.invoke(prompt)
    final_output=parser.parse(output.content)
    return {'language_feedback':final_output.feedback, 'individual_scores':[final_output.score]}

def evaluate_analysis(state:UPSCState):
    prompt=f"Evaluate the depth of analysis of the following essay and provide a feedback and asssign a score out of 10 \n {state['essay']}"
    output =model.invoke(prompt)
    final_output=parser.parse(output.content)
    return{'analysis_feedback':final_output.feedback,'individual_scores':[final_output.score]}

def evaluate_thought(state:UPSCState):
    prompt=f"Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {state['essay']}"
    output=model.invoke(prompt)
    final_output=parser.parse(output.content)
    return{'clarity_feedback':final_output.feedback,'individual_scores':[final_output.score]}

def final_evaluation(state:UPSCState):
    prompt=f"""Based on the following feedbacks create a summarized feedback \n language feedback - {state['language_feedback']}\n
    depth of analysis feedback - {state['analysis_feedback']} \n clarity of thought feedback- {state['clarity_feedback']}"""
    output=model.invoke(prompt)
    avg_score=sum(state['individual_scores'])/(len(state['individual_scores']))
    return{'overall_feedback':output , 'avg_score':avg_score}

graph = StateGraph(UPSCState)

graph.add_node('evaluate_language',evaluate_language)
graph.add_node('evaluate_analysis',evaluate_analysis)
graph.add_node('evaluate_thought',evaluate_thought)
graph.add_node('final_evaluation',final_evaluation)

graph.add_edge(START,'evaluate_language')
graph.add_edge(START,'evaluate_analysis')
graph.add_edge(START,'evaluate_thought')
graph.add_edge('evaluate_language','final_evaluation')
graph.add_edge('evaluate_analysis','final_evaluation')
graph.add_edge('evaluate_thought','final_evaluation')
graph.add_edge('final_evaluation',END)

workflow=graph.compile()

essay=input("write your essay here : ")

initial_state={
    'essay':essay
}
final_state=workflow.invoke(initial_state)

print(final_state)