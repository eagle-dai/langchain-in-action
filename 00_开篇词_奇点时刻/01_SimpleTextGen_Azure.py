from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os
from langchain_openai import AzureChatOpenAI;

llm = AzureChatOpenAI(
    temperature=0.7,
    azure_endpoint=os.getenv("BASE_URL"),
    openai_api_version=os.getenv("API_VERSION"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("API_KEY")
);

text = llm.invoke("请给我写一句情人节红玫瑰的中文宣传语")
print(text)
