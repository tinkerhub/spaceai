from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
import os
import dotenv
dotenv.load_dotenv("ops/.env")

DB_FAISS_PATH = "vectorstores/db_faiss"

with open("prompts/main.txt") as f:
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
        custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm() -> OpenAI:
    """
    This function loads the OpenAI LLM
    """
    llm = OpenAI(
        os.getenv('OPENAI_API_KEY'),
        temperature=0.5,
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
        llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_kwargs={
                "k": 2
            },
        ),
        return_messages=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot(memory: ConversationBufferMemory) -> ConversationalRetrievalChain:
    """
    This function creates a QA bot
    """
    emebddings = OpenAIEmbeddings(
        os.getenv('OPENAI_API_KEY'),
    )
    db = FAISS.load_local(DB_FAISS_PATH, emebddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrival_qa_chain(
        llm, qa_prompt, db, memory
    )
    return qa

def query_result(query: str, messages: dict) -> tuple:
    """
    This function returns the answer and messages
    """
    chat_history = ChatMessageHistory(
        messages=messages
    )
    memory = ConversationBufferMemory(
            chat_memory=chat_history,
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
    )
    qa = qa_bot(memory=memory)
    result = qa(
        {'question': query}
    )
    answer = result.get("answer")
    messages = messages_to_dict(
        qa.memory.chat_memory.messages
    )
    return answer, messages



