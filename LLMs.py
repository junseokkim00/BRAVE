from dotenv import find_dotenv, load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
def set_model(name:str):
    load_dotenv(find_dotenv())
    groq_models=['llama3-8b-8192']
    openai_models=['gpt-4o-mini', 'gpt-4o','gpt-3.5-turbo']
    if name in groq_models:
        os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
        llm = ChatGroq(model=name)
    elif name in openai_models:
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        llm = ChatOpenAI(model=name)
    elif name == 'None':
        llm = ""
    else:
        raise Exception(f"{name} is currently not supported.")
    
    return llm
    

