import os
from dotenv import load_dotenv
from ai71 import AI71

# 加载环境变量
load_dotenv()
api_key = os.getenv("AI71_API_KEY")

# 初始化 AI71 客户端
client = AI71(api_key=api_key)

# 初始化对话消息
messages = [{"role": "system", "content": "You are a helpful assistant."}]

# 添加用户提问
user_input = "你对 AI 技术怎么看？"
messages.append({"role": "user", "content": user_input})

# 打印用户提问
print(f"User: {user_input}")
print("Falcon:", end=" ", flush=True)

# 初始化助手回复内容
assistant_reply = ""

# 调用模型并流式输出回复
for chunk in client.chat.completions.create(
    messages=messages,
    model="tiiuae/falcon3-10b-instruct",
    stream=True,
):
    delta_content = chunk.choices[0].delta.content
    if delta_content:
        print(delta_content, end="", flush=True)
        assistant_reply += delta_content

# 添加助手回复到消息列表
messages.append({"role": "assistant", "content": assistant_reply})
print("\n")
