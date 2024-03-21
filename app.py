import os
import chainlit as cl
from langchain import HuggingFaceHub,PromptTemplate, LLMChain
#from langchain.schema import StrOutputParser
#from langchain.prompts import ChatPromptTemplate

os.environ['API_KEY']="your_api_key"

model_id = 'google/gemma-7b-it'
falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['API_KEY'],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":300})

template = """

    human:{question}
    Answer :
    
    """



#prompt = PromptTemplate(template=template, input_variables=['question'])



@cl.on_chat_start
def on_chat_start():
    
    prompt = PromptTemplate(template=template, input_variables=['question'])

    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm,verbose=True) 
    
    cl.user_session.set("llm_chain",llm_chain)
    
@cl.on_message
async def on_message(message:cl.Message):
    llm_chain=cl.user_session.get("llm_chain")
    
    res = await llm_chain.arun(question=message.content,callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    await cl.Message(content=res).send()
