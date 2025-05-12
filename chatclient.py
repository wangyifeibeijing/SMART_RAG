import os
from ai71 import AI71
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class FalconChatClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "tiiuae/falcon3-10b-instruct"):
        """
        初始化 FalconChatClient 实例。

        :param api_key: AI71 API 密钥。如果未提供，将从环境变量 'AI71_API_KEY' 中读取。
        :param model_name: 使用的模型名称，默认为 'tiiuae/falcon3-10b-instruct'。
        """
        self.api_key = api_key# or os.getenv("AI71_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Please set 'AI71_API_KEY' environment variable or pass it explicitly.")
        self.model_name = model_name
        self.client = AI71(api_key=self.api_key)
        self.default_system_prompt = {"role": "system", "content": "You are a helpful assistant."}

    def chat(self, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> str:
        """
        与模型进行对话。

        :param messages: 消息列表，每个消息是一个字典，包含 'role' 和 'content'。
        :param stream: 是否以流式方式返回响应。
        :param kwargs: 其他可选参数，如 temperature, top_p, max_tokens 等。
        :return: 模型的回复文本。
        """
        if not messages or messages[0].get("role") != "system":
            messages = [self.default_system_prompt] + messages

        if stream:
            response_text = ""
            for chunk in self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                stream=True,
                **kwargs
            ):
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    response_text += delta
            print()
            return response_text
        else:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                **kwargs
            )
            return response.choices[0].message.content

    def single_turn(self, user_input: str, stream: bool = False, **kwargs) -> str:
        """
        进行单轮对话。

        :param user_input: 用户输入的文本。
        :param stream: 是否以流式方式返回响应。
        :param kwargs: 其他可选参数。
        :return: 模型的回复文本。
        """
        messages = [{"role": "user", "content": user_input}]
        return self.chat(messages=messages, stream=stream, **kwargs)

    def multi_turn(self, history: List[Dict[str, str]], user_input: str, stream: bool = False, **kwargs) -> List[Dict[str, str]]:
        """
        进行多轮对话。

        :param history: 历史对话记录。
        :param user_input: 当前用户输入的文本。
        :param stream: 是否以流式方式返回响应。
        :param kwargs: 其他可选参数。
        :return: 更新后的对话历史记录。
        """
        messages = history + [{"role": "user", "content": user_input}]
        assistant_reply = self.chat(messages=messages, stream=stream, **kwargs)
        messages.append({"role": "assistant", "content": assistant_reply})
        return messages



    def batch_single_turn(self, prompts, max_workers=8):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {executor.submit(self.single_turn, prompt): prompt for prompt in prompts}
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append((prompt, result))
                except Exception as e:
                    results.append((prompt, f"Error: {e}"))
        return results
        
    def batch_multi_turn(self, histories: List[List[Dict[str, str]]], user_inputs: List[str], max_workers: int = 8, stream: bool = False, **kwargs) -> List[List[Dict[str, str]]]:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.multi_turn, history, user_input, stream, **kwargs)
                for history, user_input in zip(histories, user_inputs)
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append([{"role": "assistant", "content": f"Error: {e}"}])
        return results
