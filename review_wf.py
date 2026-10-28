from langgraph.graph import StateGraph,START, END
from langchain_groq import ChatGroq
from typing import TypedDict,Literal
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

class SentimentSchema(BaseModel):
    sentiment:Literal['positive','negative']=Field(description="sentiment of the review")

class DiagnosisSchema(BaseModel):
    issue_type:Literal["UX","performance","bug","support","others"]=Field(description="the category of the issue mentioned in the review")
    tone:Literal["angry","frustrated","disappointed","calm"]=Field(description="the emotional tone expressed by the user")
    urgency:Literal["low","medium","high"]=Field(description="how urgent or critical the issue appear to be")

parser1= PydanticOutputParser(pydantic_object=SentimentSchema)
parser2=PydanticOutputParser(pydantic_object=DiagnosisSchema)

class ReviewState(TypedDict):
    review:str
    sentiment:Literal["positive", "negative"]
    diagonsis:dict
    response:str

def find_sentiment(state:ReviewState):
    prompt=f'for the following review find out the sentiment \n {state['review']}'
    sentiment=model.invoke(prompt)
    final_sentiment=parser1.parse(sentiment.content)
    return{'sentiment':final_sentiment}

def check_sentiment(state:ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state['sentiment']=='positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'

def positive_response(state:ReviewState):
    prompt=f"""
write a warm thank-you message in response to this review :\n\n "{state['review']}"\n 
also, kindly ask the user to leave feedback on our website.
"""
    response=model.invoke(prompt).content
    return{'response':response}

def run_diagonsis(state:ReviewState):
    prompt=f"""
diagnose this negative review : \n\n{state['review']} \n"
"return issue_type,tone,and urgency
"""
    response=model.invoke(prompt)
    final_response=parser2.parse(response.content)
    return{'diagonsis':final_response.model_dump()}

def negative_response(state:ReviewState):
    diagonsis=state['diagnosis']
    prompt=f"""
you are a support assistant. The user had a '{diagonsis['issue_type']}' issue, sounded '{diagonsis['tone']}' and marked urgency as '{diagonsis['urgency']}'
weite an empathetic ,helpful resolution message"""
    response=model.invoke(prompt).content
    return{'response':response}

graph = StateGraph(ReviewState)

graph.add_node('find_sentiment',find_sentiment)
graph.add_node('positive_response',positive_response)
graph.add_node('run_diagonsis',run_diagonsis)
graph.add_node('negative_response',negative_response)

graph.add_edge(START,'find_sentiment')
graph.add_conditional_edges('find_sentiment',check_sentiment)
graph.add_edge('positive_response',END)
graph.add_edge('run_diagonsis','negative_response')
graph.add_edge('negative_response',END)

workflow=graph.compile()

review=input("Enter your review here : ")

initial_state={'review':review}

final_state=workflow.invoke(initial_state)

print(final_state['response'])

