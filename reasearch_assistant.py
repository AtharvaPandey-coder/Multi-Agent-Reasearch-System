import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import Tool
from langchain_core.tools import tool

## First step is to load,split,embed,and store the uploaded documnents
def build_retriever(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    docs =loader.load()

    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.split_documents(docs)

    embedding_model=HuggingFaceEndpointEmbeddings(
        model='sentence-transformer/all-MiniLM-L6-v2',
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
    )

    vector_store=FAISS.from_documents(chunks,embedding_model)
    retriever=vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})
    return retriever

def make_rag_tool(retriever):
    @tool
    def rag_search(query:str)->str:
        """"Use this to first answer question about the uploaded docs .
            Input should be the user's question or qa short searc query."""
        retrieved_docs=retriever.invoke(query)
        content_text="\n\n".join(doc.page_content for doc in retrieved_docs)
        return content_text if content_text else 'No relevant information Present'
    return rag_search

#3 make the web search tool 
@tool
def make_web_tool(query:str)->str:

    """Use this tool to current events ,recent actions,or anything not.
        Covered (or not covered) by document search .Input should be a user query"""
    try:

        search=DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return "Web search failed :{e}"

# make the wikipedia tool
@tool
def make_wiki_tool(query:str)->str:
    """"Use this to cover the background information of definition.
        That is not covered by the document search.Input should be a user's query"""
    try:

        wiki=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
        return wiki.run(query)
    except Exception as e:
        return f"Wikipedia searchfailed:{e}"

# Building the Agent
def build_agent(retriever=None):
    tools=[make_web_tool,make_wiki_tool]
    if retriever is not None:
        tools.append(make_rag_tool(retriever))

    agent_llm=ChatGroq(
        model='openai/gpt-oss-120b',
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    if retriever is not None:
        system_prompt =(
            "You are a helpful reasearch assistant .A document has been uploaded"
            "Always trey document search first for the user's question"
            "Check Whether the returned context has actually answers the question fully"
            "if it is missing then try web_search and wikipedia search "
            "to fill in what is missing .and returns the combined final answer"
            "Do not skip the document search always try this first"
            "Give accurate results and not hallucinate"
            
        )
    else:
        system_prompt=(
            "You are a research assistant.No document is uploaded"
            "Use the web search search for current informations and for background use wikipedia search "
            'Try to cover most of the things you found include everything but the information should becorrect'
            "Try to search twice before coming to the solution"
        )

    agent=create_agent(model=agent_llm,tools=tools,system_prompt=system_prompt)
    return agent


## Using gemini for the final answer
def summarising_final_answer(question:str,agent_output:str) -> str:
    model=ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        google_api_key=os.getenv('GEMINI_API_KEY')
    )

    prompt=(
        f'This is the question {question}'
        f'These are the raw finding and information gathered by the {agent_output}'
        'Now you have one task is to combine all these things and summarise in a good manner'
        'try to give the answer in a good manner so that users get satisfied with your answer'
    )

    result=model.invoke(prompt)
    return result.content












