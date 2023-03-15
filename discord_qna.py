from llama_index import download_loader

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


  reader = DiscordReader(discord_token=discord_token)
  documents = reader.load_data(channel_ids=[channel_ids_raw])
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

r = discord(query="what has been said?", channel_ids_raw=[123456]) #replace with your channel id
print(r)