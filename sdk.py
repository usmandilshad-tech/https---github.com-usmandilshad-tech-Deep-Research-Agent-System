# sdk.py
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import set_default_openai_client, OpenAIResponsesModel

load_dotenv()

# single async client for the whole app (Agents SDK awaits this)
external_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
set_default_openai_client(external_client)

def model_cheap():
    return OpenAIResponsesModel(
        model=os.getenv("MODEL_CHEAP", "gpt-4o-mini"),
        openai_client=external_client,
    )

def model_smart():
    return OpenAIResponsesModel(
        model=os.getenv("MODEL_SMART", "gpt-4o"),
        openai_client=external_client,
    )

def model_reasoning():
    return OpenAIResponsesModel(
        model=os.getenv("MODEL_REASONING", "o4-mini"),
        openai_client=external_client,
    )
