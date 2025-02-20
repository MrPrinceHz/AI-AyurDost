import os
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
#from dotenv import load_dotenv, find_dotenv

# Load environment variables (if needed)
#load_dotenv(find_dotenv())

#Step 1: Load Local LLM Model
model_path = "/home/prince-hazarika/Documents/Projects/AI AyurDost/llm"  

def load_local_llm(model_path):
    """Loads a locally stored LLM model for inference."""
    print("Loading local model...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")  # Uses GPU if available

    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)

    return HuggingFacePipeline(pipeline=llm_pipeline)

#Step 2: Set Up Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the provided context to answer the user's question. 
If the answer is not available in the context, say "I don't know." Do NOT make up an answer.

Context: {context}
Question: {question}

Start the answer directly, no small talk.
"""

def set_custom_prompt(custom_prompt_template):
    """Creates a custom LangChain prompt."""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

#Step 3: Load FAISS Vector Database
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS database
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

#Step 4: Create QA Chain with Local LLM
local_llm = load_local_llm(model_path)

qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("\n RESULT: ", response["result"])
print("\n SOURCE DOCUMENTS: ", response["source_documents"])
