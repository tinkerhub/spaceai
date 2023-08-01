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
- FAISS - for the vector database


## Setup instructions

- [Create openai account and create an api key](https://tfthacker.medium.com/how-to-get-your-own-api-key-for-using-openai-chatgpt-in-obsidian-41b7dd71f8d3)
- clone the repo
- Create `ops/.env` and update the variables specified in `ops/.env.example`
```
cp ops/.env.example ops/.env
```
- Create a telegram bot and put the api key in ops/.env file
- Install the dependencies using `pip install -r requirements.txt`
- Upload pdf files to `/data` folder
- Run 
```python
python core/ingest.py
```
- Make changes to the code for your application
- Run `python tele_bot.py` for testing the bot
