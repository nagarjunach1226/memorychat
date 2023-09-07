from dotenv import load_dotenv
import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate



def main():
    load_dotenv()
    st.set_page_config(page_title="Ask yout PDF")
    st.header("Ask your PDF ")
    
    pdf = st.file_uploader("upload your PDF ",type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
    
        chunk_size = 100,
        chunk_overlap  = 20,
        length_function = len,
    
        )
        
        chunks=text_splitter.split_text(text)
    
    
        embeddings=OpenAIEmbeddings()
        Vectorstore = FAISS.from_texts(chunks,embedding=embeddings)
        store_name = pdf.name[:4]
        if os.path.exists(f"{store_name}.pkl"):
        
            with open(f"{store_name}.pkl", "rb") as f:
                pickle.load(f)
        
        else:
        
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(Vectorstore,f)
         
        llm = OpenAI(temperature=0)
        
        
        
        message_history = RedisChatMessageHistory(
        url="redis://default:HQswZNLFCK6KEVDuja1Cinzr47pN8wex@redis-19498.c9.us-east-1-2.ec2.cloud.redislabs.com:19498", ttl=600, session_id=get_script_run_ctx().session_id
        )
        memory=ConversationBufferMemory( memory_key='chat_history', return_messages=True, output_key='answer')
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), Vectorstore.as_retriever(), memory=memory)
        
        
        
        
       

        
        
        count=0
        while True:
            query=st.text_input("Ask question about your file: ",key=count)
            if query.lower()=="exit":
                break
            
            result = qa({"question": query})
            res=result["answer"]
            st.write(res)
            count=count+1
if __name__=='__main__':
    main()
