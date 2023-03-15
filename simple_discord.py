from llama_index import download_loader
import streamlit as st
from llama_index import (
    download_loader,
    GPTSimpleVectorIndex,
    LLMPredictor,
    QuestionAnswerPrompt,
)
from langchain import OpenAI
import os
from dotenv import load_dotenv
import nest_asyncio
import asyncio


load_dotenv()
discord_token = os.getenv("DISCORD_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")


async def main(query, channel_ids_raw):
  nest_asyncio.apply()

  DiscordReader = download_loader('DiscordReader')
  print(channel_ids_raw)

  channel_ids = [int(x) for x in [channel_ids_raw]]  # Replace with your channel_id
  print("channel_ids", channel_ids)
  reader = DiscordReader(discord_token=discord_token)
  documents = await reader.load_data(channel_ids=[983083956230574080])
  print("documents", documents)
  QA_PROMPT_TMPL = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question in bullet point format with e new line after each point: {query_str}\n"
    )
  QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

  llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0, model_name="text-davinci-003", openai_api_key=openai_api_key
        )
    )
  index = GPTSimpleVectorIndex(documents)
  response = index.query(query, text_qa_template=QA_PROMPT)
  print(response)
  return response


def discord(query, channel_ids_raw):
  nest_asyncio.apply()

  DiscordReader = download_loader('DiscordReader')
  print(channel_ids_raw)

  channel_ids = [int(x) for x in [channel_ids_raw]]  # Replace with your channel_id
  print("channel_ids", channel_ids)
  reader = DiscordReader(discord_token=discord_token)
  documents = reader.load_data(channel_ids=[983083956230574080])
  print("documents", documents)
  QA_PROMPT_TMPL = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question in bullet point format with e new line after each point: {query_str}\n"
    )
  QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

  llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0, model_name="text-davinci-003", openai_api_key=openai_api_key
        )
    )
  index = GPTSimpleVectorIndex(documents)
  response = index.query(query, text_qa_template=QA_PROMPT)
  print(response)
  return response


st.header("Dsicord Qs")

channel_ids = st.text_input("channel ids")
user_input = st.text_input("Ask a question about the channel ids")

if st.button("Find out"):

    st.write(discord(query=user_input, channel_ids_raw=channel_ids))

# if st.button("Find out"):
#     loop = asyncio.get_event_loop()
#     result = loop.run_until_complete(main(query=user_input, channel_ids_raw=channel_ids))

#     st.markdown(result)

# if st.button("Find out"):
#     if not asyncio.get_event_loop().is_running():
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

#     result = loop.run_until_complete(main(query=user_input, channel_ids_raw=channel_ids))
#     st.markdown(result)
