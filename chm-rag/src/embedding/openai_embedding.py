"""OpenAI Embedding 实现"""

import time
from typing import List, Optional

from openai import OpenAI

from .base import BaseEmbedding
from ..config import get_openai_api_key, get_openai_base_url


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding 实现"""

    # 模型维度映射
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        初始化 OpenAI Embedding

        Args:
            model: 模型名称
            batch_size: 批处理大小
            api_key: API Key（默认从配置文件读取）
            base_url: API Base URL（默认从配置文件读取）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        api_key = api_key or get_openai_api_key()
        base_url = base_url or get_openai_base_url()
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)

    @property
    def dimension(self) -> int:
        """向量维度"""
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """处理单个批次"""
        # 清理文本（移除换行符过多的情况）
        cleaned_texts = [self._clean_text(t) for t in texts]

        for attempt in range(self.max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=cleaned_texts,
                )

                # 兼容不同 API 响应格式
                if hasattr(response, 'data'):
                    # 标准 OpenAI 响应格式
                    embeddings = sorted(response.data, key=lambda x: x.index)
                    return [e.embedding for e in embeddings]
                elif isinstance(response, dict) and 'data' in response:
                    # 字典格式响应
                    embeddings = sorted(response['data'], key=lambda x: x.get('index', 0))
                    return [e['embedding'] for e in embeddings]
                else:
                    raise RuntimeError(
                        f"API 响应格式不支持。响应类型: {type(response).__name__}, "
                        f"内容预览: {str(response)[:500]}"
                    )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Embedding 请求失败: {e}")

        return []

    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转换为向量

        Args:
            text: 查询文本

        Returns:
            向量
        """
        result = self.embed([text])
        if not result:
            raise RuntimeError("查询 Embedding 失败")
        return result[0]

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除过多的空白
        text = " ".join(text.split())
        # 截断过长的文本（text-embedding-3 限制 8192 tokens）
        # 按 2 字符/token 估算，安全上限约 15000 字符
        if len(text) > 15000:
            text = text[:15000]
        return text
