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
from langchain.chains.question_answering import load_qa_chain




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

        memory = ConversationBufferMemory( memory_key='chat_history', return_messages=True, output_key='answer')
        retriever = Vectorstore.as_retriever()


        qachat = ConversationalRetrievalChain.from_llm(
                llm=llm,
                memory=memory,
                retriever=retriever
                
            )
        while True:
            query=input("enter the input: ")
            response=qachat({"question":query})
            result=response['answer']
            print(result)
        
        
        
if __name__ == "__main__":
    main()