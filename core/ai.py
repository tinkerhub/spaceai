import weaviate
import os
from openai.embeddings_utils import get_embedding
from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import messages_from_dict, messages_to_dict
import weaviate
import openai
import redis

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


openai.api_key = os.environ.get("OPENAI_API_KEY")
vector_class = os.environ.get("VECTOR_CLASS")


auth_config = weaviate.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY"))

client = weaviate.Client(
    os.environ.get("WEAVIATE_URL"), 
    auth_config,
    additional_headers={
        "X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")
    }
)

vectorstore = Weaviate(
    client,
    vector_class,
    "content"
)

memory = ConversationBufferMemory(
    memory_key='chat_history', 
    return_messages=True, 
    output_key='answer'
)

MyOpenAI = OpenAI(temperature=0.2,
    openai_api_key=os.environ.get("OPENAI_API_KEY")),


class ChatBot:
    def __init__(self, model=MyOpenAI, vectorstore=vectorstore, memory=memory):
        self.model = model
        self.prompt = "You are SpaceAI, TinkerSpace's front desk chatbot created by GKS and team. You use the following pieces of context to answer the question about Tinkerspace given at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. \n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        self.vectorstore = vectorstore
        self.memory = memory
        self.qa = ConversationalRetrievalChain.from_llm(
            self.model, 
            self.vectorstore.as_retriever(),
            memory=self.memory
        )
        self.redis_client = redis_client
    
    def get_response(self, chat_id, question):
        memory = self.get_memory(chat_id)
        if memory:
            self.qa = ConversationalRetrievalChain.from_llm(
                self.model, 
                self.vectorstore.as_retriever(),
                memory=memory
            )
        result = self.qa({"question": question})
        self.serialize_messages(
            chat_id=chat_id
        )
        return result["answer"]
    
    def serialize_messages(self, chat_id):
        extracted_messages = messages_to_dict(self.qa.memory.chat_history.messages)
        if self.redis_client.exists(f"chat_history-{chat_id}"):
            self.redis_client.set(f"chat_history-{chat_id}", extracted_messages)
        else:
            self.redis_client.setex(
                f"chat_history-{chat_id}",
                extracted_messages,
                1800
            )

    def get_memory(self, chat_id):
        memory = None
        if self.redis_client.exists(f"chat_history-{chat_id}"):
            messages = self.redis_client.get(f"chat_history-{chat_id}")
            messages = messages_from_dict(messages)
            chat_history = ChatMessageHistory(
                messages=messages
            )
            memory = ConversationBufferMemory(
                chat_memory=chat_history,
            )
        return memory