import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading FAISS database: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question", "chat_history"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"max_length": 512}
    )

def main():
    st.title("AI AyurDost")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the provided context and chat history to answer the user's question.
                If you don't know the answer, say you don't know, and don't make up an answer.
                Provide responses based only on the given context.
                
                Chat History: {chat_history}
                Context: {context}
                Question: {question}
                
                Start the answer directly, without small talk.
                """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)
            
            chat_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                memory=st.session_state.chat_memory,
                combine_docs_chain_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = chat_chain.invoke({'question': prompt})
            result = response["AyurDost"]
            
            st.chat_message('AI AyurDost').markdown(result)
            st.session_state.messages.append({'role': 'AI AyurDost', 'content': result})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()