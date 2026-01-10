"""FAISS 向量存储实现"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from .base import BaseVectorStore, SearchResult
from ..chunker import Chunk


class FAISSVectorStore(BaseVectorStore):
    """FAISS 向量存储实现"""

    def __init__(self, dimension: int, use_gpu: bool = False):
        """
        初始化 FAISS 向量存储

        Args:
            dimension: 向量维度
            use_gpu: 是否使用 GPU（需要 faiss-gpu）
        """
        self.dimension = dimension
        self.use_gpu = use_gpu

        # 创建索引 - 使用内积（余弦相似度需要归一化）
        self._index = faiss.IndexFlatIP(dimension)

        # 存储 chunks 的元数据
        self._chunks: List[Chunk] = []

        # GPU 支持
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            except Exception:
                pass  # GPU 不可用，继续使用 CPU

    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """
        添加向量和对应的 chunks

        Args:
            chunks: 文档片段列表
            vectors: 向量列表
        """
        if not chunks or not vectors:
            return

        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量不匹配")

        # 转换为 numpy 数组
        vectors_array = np.array(vectors, dtype=np.float32)

        # L2 归一化（使内积等价于余弦相似度）
        faiss.normalize_L2(vectors_array)

        # 添加到索引
        self._index.add(vectors_array)

        # 存储 chunks
        self._chunks.extend(chunks)

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
        if self._index.ntotal == 0:
            return []

        # 转换为 numpy 数组并归一化
        query_array = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_array)

        # 如果有过滤条件，需要搜索更多结果然后过滤
        search_k = top_k * 3 if filters else top_k

        # 搜索
        scores, indices = self._index.search(query_array, min(search_k, self._index.ntotal))

        # 构建结果
        results: List[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS 返回 -1 表示无效结果
                continue

            chunk = self._chunks[idx]

            # 应用过滤条件
            if filters and not self._match_filters(chunk, filters):
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                    rank=len(results) + 1,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def _match_filters(self, chunk: Chunk, filters: Dict[str, Any]) -> bool:
        """检查 chunk 是否匹配过滤条件"""
        for key, value in filters.items():
            chunk_value = chunk.metadata.get(key)

            if isinstance(value, list):
                # 列表匹配（任一匹配即可）
                if chunk_value not in value:
                    return False
            elif chunk_value != value:
                return False

        return True

    def save(self, path: Path) -> None:
        """
        保存索引到文件

        Args:
            path: 保存路径（目录）
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        index_path = path / "index.faiss"
        if self.use_gpu:
            # GPU 索引需要先转回 CPU
            cpu_index = faiss.index_gpu_to_cpu(self._index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self._index, str(index_path))

        # 保存 chunks 元数据
        chunks_path = path / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)

        # 保存配置信息
        config_path = path / "config.json"
        config = {
            "dimension": self.dimension,
            "count": len(self._chunks),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def load(self, path: Path) -> None:
        """
        从文件加载索引

        Args:
            path: 文件路径（目录）
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"索引目录不存在: {path}")

        # 加载 FAISS 索引
        index_path = path / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        self._index = faiss.read_index(str(index_path))

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            except Exception:
                pass

        # 加载 chunks 元数据
        chunks_path = path / "chunks.pkl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks 文件不存在: {chunks_path}")

        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)

    def count(self) -> int:
        """返回存储的向量数量"""
        return self._index.ntotal

    def clear(self) -> None:
        """清空索引"""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._chunks = []

    def get_all_chunks(self) -> List[Chunk]:
        """获取所有 chunks"""
        return self._chunks.copy()
