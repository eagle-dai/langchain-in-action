# 设置环境变量和API密钥
import os
# os.environ["OPENAI_API_KEY"] = '你的OpenAI API Key'

# 创建聊天模型
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("BASE_URL"),
    openai_api_version=os.getenv("API_VERSION"),
    deployment_name=os.getenv("DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("API_KEY"),
    temperature=0
)

# 设定 AI 的角色和目标
role_template = "你是一个为花店电商公司工作的AI助手, 你的目标是帮助客户根据他们的喜好做出明智的决定"

# CoT 的关键部分，AI 解释推理过程，并加入一些先前的对话示例（Few-Shot Learning）
cot_template = """
作为一个为花店电商公司工作的AI助手，我的目标是帮助客户根据他们的喜好做出明智的决定。 

我会按部就班的思考，先理解客户的需求，然后考虑各种鲜花的涵义，最后根据这个需求，给出我的推荐。
同时，我也会向客户解释我这样推荐的原因。

示例 1:
  人类：我想找一种象征爱情的花。
  AI：首先，我理解你正在寻找一种可以象征爱情的花。在许多文化中，红玫瑰被视为爱情的象征，这是因为它们的红色通常与热情和浓烈的感情联系在一起。因此，考虑到这一点，我会推荐红玫瑰。红玫瑰不仅能够象征爱情，同时也可以传达出强烈的感情，这是你在寻找的。

示例 2:
  人类：我想要一些独特和奇特的花。
  AI：从你的需求中，我理解你想要的是独一无二和引人注目的花朵。兰花是一种非常独特并且颜色鲜艳的花，它们在世界上的许多地方都被视为奢侈品和美的象征。因此，我建议你考虑兰花。选择兰花可以满足你对独特和奇特的要求，而且，兰花的美丽和它们所代表的力量和奢侈也可能会吸引你。
"""
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
system_prompt_role = SystemMessagePromptTemplate.from_template(role_template)
system_prompt_cot = SystemMessagePromptTemplate.from_template(cot_template)

# 用户的询问
human_template = "{human_input}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 将以上所有信息结合为一个聊天提示
chat_prompt = ChatPromptTemplate.from_messages([system_prompt_role, system_prompt_cot, human_prompt])

prompt = chat_prompt.format_prompt(human_input="我想为我的女朋友购买一些花。她喜欢粉色和紫色。你有什么建议吗?").to_messages()
print(prompt)

# 接收用户的询问，返回回答结果
response = llm.invoke(prompt)
print(response.content)

############################################################################################################
# output:
# 当然，根据你女朋友对粉色和紫色的偏好，我可以推荐几种花朵：
#
# 1. 粉色玫瑰：粉色玫瑰代表着温柔、欣赏和幸福，是表达爱意和感激之情的理想选择。
#
# 2. 紫色鸢尾花：紫色鸢尾花象征着智慧、尊敬和皇家。它们的独特形状和鲜艳的紫色可以为任何花束增添一抹优雅。
#
# 3. 粉色百合：粉色百合通常与崇高和纯洁的爱情相关联，它们的大型花朵和优雅的姿态可以制作成非常吸引人的花束。
#
# 4. 紫色风信子：紫色风信子带有浓郁的香气，代表着和平、承诺和宽恕。它们是春天的象征，也是一个温馨的礼物。
#
# 5. 粉色郁金香：粉色郁金香是完美的爱情宣言，它们代表着关怀和附加的情感。
#
# 6. 紫色罗勒：紫色罗勒不仅颜色独特，而且还有着愉悦的香气，可以作为花束中的一个有趣的补充。
#
# 你可以选择以上的某一种花，或者将几种不同的粉色和紫色花朵组合成一个花束，这样可以创造出一个既符合她的颜色偏好，又具有多样化美感的礼物。记得在选择时考虑花朵的季节性，以确保你能够 获得最新鲜、最美丽的花朵。
############################################################################################################
