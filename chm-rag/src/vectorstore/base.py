"""向量存储模块 - 抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..chunker import Chunk


@dataclass
class SearchResult:
    """搜索结果"""

    chunk: Chunk
    score: float  # 相似度分数
    rank: int  # 排名


class BaseVectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """
        添加向量和对应的 chunks

        Args:
            chunks: 文档片段列表
            vectors: 向量列表
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        搜索相似向量

        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filters: 元数据过滤条件

        Returns:
            搜索结果列表
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        保存索引到文件

        Args:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        从文件加载索引

        Args:
            path: 文件路径
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """返回存储的向量数量"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空索引"""
        pass
