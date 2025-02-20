import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define Paths
DB_FAISS_PATH = "vectorstore/db_faiss"
LOCAL_MODEL_PATH = "path_to_your_downloaded_model"  # Update this path

# Load Vector Store (FAISS)
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Load Local LLM Model
@st.cache_resource
def load_local_llm(model_path):
    """Loads the Mistral-7B model locally."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=llm_pipeline)

# Set Up Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the provided context to answer the user's question. 
If the answer is not available in the context, say "I don't know." Do NOT make up an answer.

Context: {context}
Question: {question}

Start the answer directly, no small talk.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Main Streamlit App
def main():
    st.title("AI AyurDost Chatbot")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Type your question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Error loading vector store.")
                return

            # Load LLM
            local_llm = load_local_llm(LOCAL_MODEL_PATH)

            # Create QA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=local_llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            result_to_show = f"**Answer:** {result}\n\n **Source Docs:**\n{str(source_documents)}"

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
