from dotenv import load_dotenv
import os

import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import MongoDBChatMessageHistory
from pydantic import BaseModel
from langchain.chains import RetrievalQA




def main():
    load_dotenv()
    pdf=open("DP1Merrill_Manual_en.pdf","rb")
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
        def user1(query1):
            
            mongo_history1 = MongoDBChatMessageHistory(
            connection_string="mongodb://localhost:27017", 
            session_id="session-1"
            )
            
            memory1 = ConversationBufferMemory( memory_key='chat_history', return_messages=True, output_key='answer',chat_memory=mongo_history1)
            qachat1 = ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever, memory=memory1)
            response=qachat1({"question":query1})
            result=response['answer']
            print(result)
            
        retriever = Vectorstore.as_retriever()
            
        def user2(query2):   
            mongo_history2 = MongoDBChatMessageHistory(
            connection_string="mongodb://localhost:27017", 
            session_id="session-2"
            )
        
            memory2 = ConversationBufferMemory( memory_key='chat_history', return_messages=True, output_key='answer',chat_memory=mongo_history2)
            qachat2 = ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,memory=memory2)
            response=qachat2({"question":query2})
            result=response['answer']
            print(result)
        


        
        
        while True:
            
            query1=input("User 1 : ")
            user1(query1=query1)
            
            query2=input("User 2 : ")
            user2(query2=query2)
            
            
            
            
            
        
        
        
if __name__ == "__main__":
    main()