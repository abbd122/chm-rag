"""配置管理模块"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """配置管理类"""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self.load()

    def load(self, config_path: Optional[Path] = None) -> None:
        """加载配置文件"""
        if config_path is None:
            # 默认在项目根目录查找 config.yaml
            config_path = Path(__file__).parent.parent / "config.yaml"

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "openai": {
                "api_key": "",
                "base_url": "",
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-large",
                "batch_size": 100,
            },
            "vectorstore": {
                "provider": "faiss",
                "index_path": "./data/index",
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            "rag": {
                "top_k": 5,
                "max_chunk_tokens": 800,
            },
            "paths": {
                "chm_output": "./data/chm_extracted",
                "chunks_cache": "./data/chunks.json",
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    @property
    def embedding(self) -> Dict[str, Any]:
        """Embedding 配置"""
        return self._config.get("embedding", {})

    @property
    def vectorstore(self) -> Dict[str, Any]:
        """向量存储配置"""
        return self._config.get("vectorstore", {})

    @property
    def llm(self) -> Dict[str, Any]:
        """LLM 配置"""
        return self._config.get("llm", {})

    @property
    def rag(self) -> Dict[str, Any]:
        """RAG 配置"""
        return self._config.get("rag", {})

    @property
    def paths(self) -> Dict[str, Any]:
        """路径配置"""
        return self._config.get("paths", {})


def get_config() -> Config:
    """获取配置实例"""
    return Config()


def get_openai_api_key() -> str:
    """获取 OpenAI API Key（优先从配置文件读取，其次从环境变量）"""
    config = get_config()

    # 优先从配置文件读取
    key = config.get("openai.api_key", "")
    if key and key != "your-api-key-here":
        return key

    # 其次从环境变量读取
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key

    raise ValueError(
        "OpenAI API Key 未配置。请在 config.yaml 中设置 openai.api_key，"
        "或设置环境变量 OPENAI_API_KEY"
    )


def get_openai_base_url() -> Optional[str]:
    """获取 OpenAI API Base URL（代理地址）"""
    config = get_config()
    base_url = config.get("openai.base_url", "")
    if base_url:
        return base_url.rstrip("/")
    return None
