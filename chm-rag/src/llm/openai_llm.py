"""OpenAI LLM 实现"""

from typing import List, Optional

from openai import OpenAI

from .base import BaseLLM, Message
from ..config import get_openai_api_key, get_openai_base_url


class OpenAILLM(BaseLLM):
    """OpenAI LLM 实现"""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        初始化 OpenAI LLM

        Args:
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            api_key: API Key（默认从配置文件读取）
            base_url: API Base URL（默认从配置文件读取）
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or get_openai_api_key()
        base_url = base_url or get_openai_base_url()
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成回答

        Args:
            prompt: 用户提示
            system_prompt: 系统提示

        Returns:
            生成的回答
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content or ""

    def chat(self, messages: List[Message]) -> str:
        """
        多轮对话

        Args:
            messages: 消息列表

        Returns:
            生成的回答
        """
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content or ""
