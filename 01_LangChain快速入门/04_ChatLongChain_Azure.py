import os

from langchain_openai import AzureChatOpenAI;
chat = AzureChatOpenAI(
    azure_endpoint=os.getenv("BASE_URL"),
    openai_api_version=os.getenv("API_VERSION"),
    deployment_name="gpt-4-turbo",
    openai_api_key=os.getenv("API_KEY"),
    temperature=0.8,
    max_tokens=200
);
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
messages = [
    SystemMessage(content="你是一个很棒的智能助手"),
    HumanMessage(content="请给我的花店起个名")
]
response = chat.invoke(messages)
print(response)