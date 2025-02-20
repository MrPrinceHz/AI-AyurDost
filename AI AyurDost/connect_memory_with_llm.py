import os
import logging
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(find_dotenv())

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "tiiuae/falcon-7b-instruct"

def load_llm(huggingface_repo_id):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            huggingfacehub_api_token=HF_TOKEN,
            model_kwargs={"max_length": 512}
        )
        logger.info("LLM loaded successfully.")
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        raise

# Step 2: Connect LLM with FAISS and Create Chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    try:
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
        logger.info("Custom prompt template set successfully.")
        return prompt
    except Exception as e:
        logger.error(f"Error setting custom prompt: {e}")
        raise

# Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    logger.info("FAISS database loaded successfully.")
except Exception as e:
    logger.error(f"Error loading FAISS database: {e}")
    exit(1)

# Create QA Chain
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    logger.info("QA chain created successfully.")
except Exception as e:
    logger.error(f"Error creating QA chain: {e}")
    exit(1)

# Step 3: Invoke the QA Chain
try:
    user_query = input("Write Query Here: ")
    response = qa_chain.invoke({'query': user_query})
    print("RESULT: ", response["result"])
    print("SOURCE DOCUMENTS: ", response["source_documents"])
except Exception as e:
    logger.error(f"Error during QA chain invocation: {e}")