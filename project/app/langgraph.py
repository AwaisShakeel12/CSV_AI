
import os 
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import  CompiledStateGraph 
from langgraph.graph import  StateGraph, START, END
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import  MessagesState
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_experimental.agents import create_pandas_dataframe_agent


from .models import Document



load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

api_key

genai.configure(api_key=api_key)

document = Document.objects.last()
file = document.file.path


llm =  ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=api_key)
@tool
def show_null():
    """
    Show all containing null values from the DataFrame.
    Returns a success message.
    """
    global df  
    df.isnull().sum()  
    return "✅ Successfully Showed all Null values"


@tool
def remove_duplicated():
    """
    Remove all duplicated values from the DataFrame.
    Returns a success message.
    """
    global df  
    df.drop_duplicates(inplace=True) 
    df.to_csv("new.csv", index=False)
    return "✅ Successfully Remove duplivcated"


@tool
def remove_null_values():
    """
    Removes all rows containing null values from the DataFrame.
    Returns a success message.
    """
    global df 
    df.dropna(inplace=True)  
    df.to_csv("new.csv", index=False)
    return "✅ Successfully removed all null values."

# df = pd.read_csv("placement.csv")
df = pd.read_csv(file)


action_agent = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type="tool-calling",
    verbose=True,
    extra_tools=[remove_null_values, show_null, remove_duplicated],
    allow_dangerous_code=True
)

info_agent = create_csv_agent(llm,file,allow_dangerous_code=True)

prompt = SystemMessage(
    content= """ 
    
      "Categorize the following user query into one of these categories: "
      "if user want query is to just get analysis of file like inquery so categorie it to Inquery"
      "if user want to perform some action like clean data, remove null  so categorize it as Clean"
      answer in only one word  either "Inquery" or "Clean"
      "Inquery, Clean, Query: {query}"
      """
  )

def categorize(state: MessagesState) -> MessagesState:

  return {'messages':[llm.invoke([prompt] + state['messages'])]}

def categorize(state: MessagesState) -> MessagesState:
    response = llm.invoke([prompt] + state['messages'])
    return {'messages': state['messages'] + [response]}  

def Inquery_agent(state: MessagesState) -> MessagesState:
    response = info_agent.run(state['messages'])
    return {'messages': state['messages'] + [response]}  

def Action_agent(state: MessagesState) -> MessagesState:
    response = action_agent.run(state['messages'])
    return {'messages': state['messages'] + [response]}  
def route_query(state: MessagesState) -> str:
    last_message = state["messages"][-1]  
    
    if isinstance(last_message, AIMessage) and last_message.content.strip() == "Inquery":
        return "Inquery_agent"
    else:
        return "Action_agent"






builder: StateGraph= StateGraph(MessagesState)

builder.add_node('categorize',categorize)
builder.add_node('Inquery_agent',Inquery_agent)
builder.add_node('Action_agent',Action_agent)

builder.add_edge(START, 'categorize')

builder.add_conditional_edges(
    "categorize",
    route_query,{
        "Inquery_agent" : "Inquery_agent",
        "Action_agent" :  "Action_agent",
    }
)


builder.add_edge('Inquery_agent', END)
builder.add_edge('Action_agent', END)



memory: MemorySaver = MemorySaver()
graph: CompiledStateGraph= builder.compile( checkpointer=memory)
graph: CompiledStateGraph= builder.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))


