
samples = [
  {
    "flower_type": "玫瑰",
    "occasion": "爱情",
    "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
  },
  {
    "flower_type": "康乃馨",
    "occasion": "母亲节",
    "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
  },
  {
    "flower_type": "百合",
    "occasion": "庆祝",
    "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
  },
  {
    "flower_type": "向日葵",
    "occasion": "鼓励",
    "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
  }
]

# 2. 创建一个提示模板
from langchain.prompts.prompt import PromptTemplate
prompt_sample = PromptTemplate(input_variables=["flower_type", "occasion", "ad_copy"], 
                               template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}")
print(prompt_sample.format(**samples[0]))

# 3. 创建一个FewShotPromptTemplate对象
from langchain.prompts.few_shot import FewShotPromptTemplate
prompt = FewShotPromptTemplate(
    examples=samples,
    example_prompt=prompt_sample,
    suffix="鲜花类型: {flower_type}\n场合: {occasion}", # suffix is put after examples
    input_variables=["flower_type", "occasion"]
)
print(prompt.format(flower_type="野玫瑰", occasion="爱情"))

# 4. 把提示传递给大模型
import os
from langchain_openai import AzureChatOpenAI
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("BASE_URL"),
    openai_api_version=os.getenv("API_VERSION"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("API_KEY"),
)
result = model.invoke(prompt.format(flower_type="野玫瑰", occasion="爱情"))
print(result.content)

# 5. 使用示例选择器
#
# 如果我们的示例很多，那么一次性把所有示例发送给模型是不现实而且低效的。
# 另外，每次都包含太多的 Token 也会浪费流量（OpenAI 是按照 Token 数来收取费用）。
# LangChain 给我们提供了示例选择器，来选择最合适的样本。
# （注意，因为示例选择器使用向量相似度比较的功能，此处需要安装向量数据库）

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Qdrant
from langchain_openai import AzureOpenAIEmbeddings

# 初始化示例选择器
print("-- Initializing example selector...")

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("BASE_URL"),
    openai_api_key=os.getenv("API_KEY"),
    azure_deployment="text-embedding-ada-002",
    openai_api_version=os.getenv("API_VERSION")
)
example_selector = SemanticSimilarityExampleSelector.from_examples(
    samples,
    embeddings,
    Qdrant,
    k=1,
    location=":memory:",  # in-memory 存储
)

# 创建一个使用示例选择器的FewShotPromptTemplate对象
prompt = FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=prompt_sample, 
    suffix="鲜花类型: {flower_type}\n场合: {occasion}", 
    input_variables=["flower_type", "occasion"]
)
print(prompt.format(flower_type="红玫瑰", occasion="爱情"))
