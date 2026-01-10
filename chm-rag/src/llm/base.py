"""LLM 模块 - 抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Message:
    """聊天消息"""

    role: str  # system, user, assistant
    content: str


class BaseLLM(ABC):
    """LLM 抽象基类"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成回答

        Args:
            prompt: 用户提示
            system_prompt: 系统提示

        Returns:
            生成的回答
        """
        pass

    @abstractmethod
    def chat(self, messages: List[Message]) -> str:
        """
        多轮对话

        Args:
            messages: 消息列表

        Returns:
            生成的回答
        """
        pass
