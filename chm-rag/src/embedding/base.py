"""Embedding 模块 - 抽象基类"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """Embedding 抽象基类"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表转换为向量列表

        Args:
            texts: 文本列表

        Returns:
            向量列表，每个向量是一个 float 列表
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转换为向量

        Args:
            text: 查询文本

        Returns:
            向量
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        pass
