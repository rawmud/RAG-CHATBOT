
#from huggingface_hub import login
#from huggingface_hub import InferenceClient
#from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
#from langchain_huggingface import HuggingFacePipeline
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEndpointEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.document_loaders import WebBaseLoader,TextLoader,RecursiveUrlLoader

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
import bs4
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
import json
from langchain_core.prompts import MessagesPlaceholder
from langchain.chat_models import init_chat_model
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import ssl
import argparse
from langchain_core.messages import AIMessage, HumanMessage
import sys
import os
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
import warnings
import flask
from flask import Flask, request, jsonify
from flask_socketio import SocketIO

#import flask-socketio
#socketio = SocketIO(app)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


warnings.filterwarnings("ignore")
#hfApiKey = Enter API Key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"]  =  #Enter API Key
print("python script started");

def get_sitemap(url):
    req = Request(
        url=url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    response = urlopen(req)
    xml = BeautifulSoup(
        response,
        "lxml-xml",
        from_encoding=response.info().get_param("charset")
    )
    return xml

def get_urls(xml, name=None, data=None, verbose=False):
    urls = []
    for url in xml.find_all("url"):
        if xml.find("loc"):
            loc = url.findNext("loc").text
            urls.append(loc)
    return urls


def createVectorStore():
    print("create Vectore Store Started")
    #login(hfApiKey,False,False)
    print("loggedin")
    loader = PyPDFLoader("./RSS-RAG-DOC 2.pdf") # or enter in pdf document name
    docs = loader.load()
    print("pdf loaded")



    #from langchain_community.vectorstores import FAISS

    #model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    print("model set")
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    print("split set")
    splits = text_splitter.split_documents(docs)
    print("embedding set")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = OpenAIEmbeddings()
    print("vectorstre set ")
    vectorstore = Chroma.from_documents(documents=splits,
                                        embedding=embeddings)
    print("retriever loading")
    retriever = vectorstore.as_retriever()
    #print(f"retriever within python file: {retriever}")
    print("VectorStore Script Finished.  Retriever created.")


    """ 
    ssl._create_default_https_context = ssl._create_stdlib_context
    

    #url ="https://www.rawstockstats.com/assets/sitemap.xml"
    #xml = get_sitemap(url)
    #urls = get_urls(xml, verbose=False)
    #urls = ["https://www.rawstockstats.com"]
  
    docs = []
    
    for i, url in enumerate(urls):
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=embeddings)
        retriever = vectorstore.as_retriever()
        #print(f"retriever within python file: {retriever}")
        print("VectorStore Script Finished.  Retriever created.")
  loader = PyPDFLoader("/content/RSS-RAG-DOC 2.pdf")
  """
    return retriever


def convert_chat_history(node_history):
    """
    Converts a chat history from Node.js format to Python Langchain format.

    Args:
        node_history (list): A list of message objects from Node.js.
                              Each object should have 'type' (either 'human' or 'ai') and 'text' keys.

    Returns:
        list: A list of message objects in Python Langchain format (HumanMessage or AIMessage).
    """
    python_history = []
    print("converting History")
    for message in node_history:
        #print(message)
        #print(message['kwargs'])
        #print(message['kwargs']['content'])
        if message['id'][2] == 'HumanMessage':
            if message['kwargs']['content'] !='':
                python_history.append(HumanMessage(content=message['kwargs']['content']))
        elif message['id'][2] == 'AIMessage':
            python_history.append(AIMessage(content=message['kwargs']['content']))
    return python_history
    
def initializePrompt(retriever):
    print("initializePrompt Started")
    #login(hfApiKey,False,False)
    """  
    llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3.5-mini-instruct",
    # repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    stop_sequences=['Human:'],
    )
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125");
    #gpt-4.1-nano-2025-04-14
    #gpt-4o-mini-search-preview-2025-03-11
    #babbage-002
    #chat = ChatHuggingFace(llm=llm, verbose=True)

    system_prompt = (
    "You are a sales assitant for Raw Stock Stats (RSS), for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "Answer in a persuasive tone that gets the client interested in purchasing a "
    "subscription for RSS SP500 Report."
    "If you cannot provide an"
    "answer, say that you "
    "don't know and ask the client to send an email rawootllc@gmail.com for "
    "further questioning. Use three sentences maximum and keep the "
    "answer concise.  "
      "\n\n"
    "{context}"
    )
    """  
    "If the question is not clear ask follow up questions in the format 'Did you mean to ask:', followed by follow up"
    "questions.  Do not include the answers" 
    "for these follow up questions.  Just include suggested follow up questions."
    "\n\n"
    "{context}"
    )
   
"""
  
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do not answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )

    
    

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,
                                   question_answer_chain)
    print("initializePrompt Script Finished.  rag_chain created.")
    return rag_chain



   
#@app.route("/askQuestion",methods=["POST"])
#def askQuestion(rag_chain, question, chat_history):
#print("post request hit")
@socketio.on('askQuestion')
def askQuestion():
    print("start askQuestion in pythonScript")
    data = request.get_json()
    #print(data[0])
   
    #print(data['HumanQuestion'])
    """
      const message = {
        
      HumanQuestion:question,
      chatHistory:this.chatHistory
    }
    """
    
    question = data['HumanQuestion']
    chatHistory = data['chatHistory']
    print("send convert MEssage History")
    python_chat_history = convert_chat_history(chatHistory)
    print(question)
    print(python_chat_history)
    #login(hfApiKey,False,False)
    print("SEnding Invoke")
    #ai_msg =  rag_chain.invoke({"input": question, "chat_history": python_chat_history})
    print("finsihed invoke")
    """  
    chatHistory.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg["answer"]),
    ]
)
"""
    #print("ai_message: " + ai_msg)
    #print(f"python answer: {ai_msg['answer']}" )
    #return ai_msg["answer"]
    return chatHistory


 

def initiateLLM():
    retreiver = createVectorStore()
    chatBot = initializePrompt(retreiver)
    return chatBot

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('message')
def handle_message(data):
    #print('Received message:', data)
    #print("start askQuestion in pythonScript")
    #data = data.get_json()
    #print(data[0])
   
    #print(data['HumanQuestion'])
    """
      const message = {
        
      HumanQuestion:question,
      chatHistory:this.chatHistory
    }
    """
    
    question = data['HumanQuestion']
    chatHistory = data['chatHistory']
    python_chat_history = convert_chat_history(chatHistory)
    rag_chain = initializePrompt(retriever1)
    ai_msg =  rag_chain.invoke({"input": question, "chat_history": python_chat_history})

    answer = ai_msg["answer"]

    socketio.emit('response', {'answer':  answer,'question': question})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == "__main__":
    #def start():
    print("main script started")
    a = 1
    print("call createVectoreStore() script")
    global retriever1
    retriever1 = createVectorStore()
    #print("finshed createVectoreStore() script")
    #print("call initializeprompt() started")
    global rag_chain
    #rag_chain = initializePrompt(retriever1)
    #print("finished intializeprompt() script")
    socketio.run(app, host="0.0.0.0", debug=True, port=5000)

    #app.run(debug=True)



