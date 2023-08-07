from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import messages_from_dict, messages_to_dict
from utils.resources import retrieve_learning_topics, retrieve_url
import numpy as np
import json
import os

import dotenv

with open("data/dummy-lp.json") as f:
    learning_paths = json.load(f)

dotenv.load_dotenv("ops/.env")

DB_FAISS_PATH = os.getenv('DB_FAISS_PATH')

MAIN_PROMPT_PATH = os.environ.get('MAIN_PROMPT_PATH')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

with open(MAIN_PROMPT_PATH) as f:
    system = "\n".join(f.readlines())

custom_prompt_template = system

memory = ConversationBufferMemory(
    memory_key='chat_history', 
    return_messages=True, 
    output_key='answer'
)

def set_custom_prompt() -> PromptTemplate:
    """
    This function creates a custom prompt
    """
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm() -> OpenAI:
    """
    This function loads the OpenAI LLM
    """
    llm = OpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.1,
    )
    return llm

def retrival_qa_chain(
        llm: OpenAI, 
        prompt: str, 
        db: FAISS, 
        memory: ConversationBufferMemory
) -> ConversationalRetrievalChain:
    """
    This function creates a ConversationalRetrievalChain
    """
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(
            search_kwargs={
                "k": 2
            },
        ),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return qa_chain

emebddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
db = FAISS.load_local(DB_FAISS_PATH, emebddings)
llm = load_llm()

qa_prompt = set_custom_prompt()

def query_result(query: str, messages: list) -> tuple:
    """
    This function returns the answer and messages
    """
    messages = messages_from_dict(messages)
    chat_history = ChatMessageHistory(
        messages=messages
    )
    memory = ConversationBufferMemory(
            chat_memory=chat_history,
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
    )
    qa = retrival_qa_chain(
        llm, qa_prompt, db, memory
    )
    result = qa(
        {'question': query}
    )
    answer = result.get("answer")
    messages = messages_to_dict(
        qa.memory.chat_memory.messages
    )
    return answer, messages

def classify(query, topics):
    embed_topics = np.load("data/embed_topics.npy")
    embed = emebddings.embed_documents([query])
    embed = np.array(embed[0]).reshape(1, -1)
    scores = cosine_similarity(embed, embed_topics)
    arg = np.argmax(scores)
    return topics[arg], scores[0][arg]


def learning_path_query(query):
    data = learning_paths
    topics = retrieve_learning_topics(data)
    topic, score = classify(query, topics)
    if score > 0.5:
        topic = topic.split("learn ")[1]
        url = retrieve_url(topic, data)
        context = f"This is the maker station learning path for {topic} by TinkerHub. {url}"
        llm_chain = LLMChain(llm=llm, prompt=qa_prompt)
        response = llm_chain(inputs={"context": context, "question": query})
        return response.get("text")
    return None
    



