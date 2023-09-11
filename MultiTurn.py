import gradio as gr
import os
import time

from langchain.document_loaders import OnlinePDFLoader

from langchain.text_splitter import CharacterTextSplitter


from langchain.llms import OpenAI
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory

from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient


def loading_pdf():
    return "Loading..."

def pdf_changes(pdf_doc, open_ai_key):
    if openai_key is not None:
        os.environ['OPENAI_API_KEY'] = open_ai_key
        text = ""
        for pdf in pdf_doc:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(text)
        mongo_client = MongoClient("mongodb+srv://Cluster1:2612Nani@cluster0.6uuxrae.mongodb.net/")
        db_name = "data1"
        collection_name = "col2"
        collection = mongo_client[db_name][collection_name]
        index_name = "langchain1"
        embeddings = OpenAIEmbeddings()
    
        Vectorstore = MongoDBAtlasVectorSearch(embedding=embeddings,collection=collection,index_name=index_name)
        Vectorstore.add_texts(texts)
        retriever = Vectorstore.as_retriever()
        global qa
        message_history = RedisChatMessageHistory(
        url="redis://default:HQswZNLFCK6KEVDuja1Cinzr47pN8wex@redis-19498.c9.us-east-1-2.ec2.cloud.redislabs.com:19498", ttl=600, session_id="session-1"
        )
        memory= ConversationSummaryMemory( llm=OpenAI(temperature=0.5,model_name="gpt-3.5-turbo"),chat_memory=message_history,memory_key="chat_history") 
        qa = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(temperature=0.5,model_name="gpt-3.5-turbo"), 
            retriever=retriever, 
            memory=memory,
            get_chat_history=get_chat_history,
            return_source_documents=False)
        return "Ready.. please ask questions regarding doc"
    else:
        return "You forgot OpenAI API key"
def get_chat_history(inputs) -> str:
    res = []
    for chat_history in inputs:
        res.append(f"chat_history:{chat_history}")
    return "\n".join(res)

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history)
    history[-1][1] = ""
    
    for character in response:     
        history[-1][1] += character
        time.sleep(0.05)
        yield history
    

def infer(question, history):
    
    res = []
    for human, ai in history[:-1]:
        pair = (human, ai)
        res.append(pair)
    
    chat_history = res
    #print(chat_history)
    query = question
    result = qa({"question": query, "chat_history": chat_history})
    #print(result)
    return result["answer"]

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with PDF • OpenAI</h1>
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the pdf ;) <br />
    This version is set to store chat history, and uses OpenAI as LLM, don't forget to copy/paste your OpenAI API key</p>
</div>
"""


with gr.Blocks(css=css) as demo:
    gr.HTML(
        """<h1>Welcome to AI PDF Assistant</h1>"""
    )
    gr.Markdown(
        "AI Assistant for PDF documents. Upload your pdf document, click 'Process PDF docs' and wait for success confirmation message.<br>"
        "After success confirmation, click on the 'AI Assistant' tab to interact with your document.<br>"
        "Type your query, and  hit enter. Click on 'Clear Chat History' to delete all previous conversations."
    )
    with gr.Column():
            openai_key = gr.Textbox(label="You OpenAI API key", type="password")
            pdf_doc = gr.File(label="Load a pdf", file_types=['.pdf'], type="file")
            with gr.Row():
                langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Load pdf to langchain")

    

    with gr.Column():
        chatbot = gr.Chatbot()
        question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
        submit_btn = gr.Button("Send Message")
    load_pdf.click(loading_pdf, None, langchain_status, queue=False)    
    load_pdf.click(pdf_changes, inputs=[pdf_doc, openai_key], outputs=[langchain_status], queue=False)
    question.submit(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
        bot, chatbot, chatbot)

    
    
    
    
    
demo.queue().launch(share=True)
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
