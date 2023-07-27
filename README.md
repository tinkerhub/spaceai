# SpaceAI - Front desk AI for TinkerSpace

## What is SpaceAI?

SpaceAI is a front desk AI for TinkerSpace. You can ask questions about Tinkerhub and TinkerSpace to this chatbot

## How to use SpaceAI?

You can ask questions like:
- What is Tinkerhub?
- What is TinkerSpace?

## What is the tech stack?

SpaceAI is built in python using

- FastAPI - for the API
- LangChain - Language model
- telegram api - for the chatbot
- weaviate - for the vector database


## Setup instructions

- [Create openai account and create an api key](https://tfthacker.medium.com/how-to-get-your-own-api-key-for-using-openai-chatgpt-in-obsidian-41b7dd71f8d3)
- [Create a vercel account and create a team](https://vercel.com/docs/accounts/create-an-account)
- [Create a weaviate account and create a cluster](https://weaviate.io/developers/academy/zero_to_mvp/hello_weaviate/set_up)
- [Create a telegram bot](https://www.freecodecamp.org/news/how-to-create-a-telegram-bot-using-python/)
- Clone the repo
- Install the dependencies using `pip install -r requirements.txt`
- Create .env and update the variables specified in .env.example
```
cp ops/.env.example ops/.env
```
- Make changes to the code for your application (refer notebooks/test_weaviate.ipynb for learning how to create vector database with your data)
- Run `python main.py` for testing
- [Deploy to vercel](https://blog.logrocket.com/deploying-fastapi-applications-to-vercel/)
