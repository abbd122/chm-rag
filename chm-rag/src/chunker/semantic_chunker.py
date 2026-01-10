"""语义切块模块 - 核心模块"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import tiktoken

from ..cleaner import CleanedDocument


@dataclass
class Chunk:
    """文档片段"""

    content: str  # 片段内容
    metadata: Dict[str, Any] = field(default_factory=dict)

    # metadata 包含:
    # - module: 模块名称（从目录路径推断）
    # - symbol: 函数名/类名/接口名
    # - chapter: 原 CHM 章节路径
    # - source_file: HTML 文件名
    # - title: 文档标题

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典创建"""
        return cls(content=data["content"], metadata=data.get("metadata", {}))

    def __len__(self) -> int:
        return len(self.content)


class SemanticChunker:
    """
    语义切块器

    按接口/类/函数级别进行切块，而非固定 token 数量切割。
    """

    # API 签名模式
    API_PATTERNS = [
        # C/C++ 函数
        r"^\s*(?:(?:static|virtual|inline|const|extern)\s+)*"
        r"(?:[\w:*&<>]+\s+)+(\w+)\s*\([^)]*\)",
        # Python 函数/方法
        r"^\s*def\s+(\w+)\s*\(",
        # Python 类
        r"^\s*class\s+(\w+)\s*[:\(]",
        # JavaScript/TypeScript 函数
        r"^\s*(?:async\s+)?function\s+(\w+)\s*\(",
        r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(",
        # 方法定义
        r"^\s*(\w+)\s*:\s*function\s*\(",
    ]

    # 标题模式（Markdown）
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(
        self,
        max_tokens: int = 800,
        min_tokens: int = 50,
        encoding_name: str = "cl100k_base",
    ):
        """
        初始化语义切块器

        Args:
            max_tokens: 单个 chunk 的最大 token 数
            min_tokens: 单个 chunk 的最小 token 数
            encoding_name: tiktoken 编码名称
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self._encoder = tiktoken.get_encoding(encoding_name)
        self._api_patterns = [re.compile(p, re.MULTILINE) for p in self.API_PATTERNS]

    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        return len(self._encoder.encode(text))

    def chunk_document(self, doc: CleanedDocument) -> List[Chunk]:
        """
        对清洗后的文档进行切块

        Args:
            doc: 清洗后的文档

        Returns:
            切块列表
        """
        # 基础元数据
        base_metadata = {
            "source_file": doc.source_file,
            "title": doc.title,
            "module": self._extract_module(doc.source_file),
            "chapter": self._extract_chapter(doc.source_file),
        }

        # 按标题分割
        sections = self._split_by_headings(doc.content)

        chunks: List[Chunk] = []

        for section_title, section_content in sections:
            # 提取 API 符号
            symbol = self._extract_symbol(section_title, section_content)

            # 创建元数据
            metadata = {
                **base_metadata,
                "symbol": symbol,
                "section_title": section_title,
            }

            # 检查 token 数量
            token_count = self.count_tokens(section_content)

            if token_count <= self.max_tokens:
                # 内容不超过限制，直接作为一个 chunk
                if token_count >= self.min_tokens or symbol:
                    chunks.append(Chunk(content=section_content, metadata=metadata))
            else:
                # 内容过长，需要智能分割
                sub_chunks = self._split_large_section(
                    section_content, section_title, metadata
                )
                chunks.extend(sub_chunks)

        # 合并过小的相邻 chunks
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _split_by_headings(self, content: str) -> List[tuple[str, str]]:
        """
        按标题分割文档

        Returns:
            [(标题, 内容), ...]
        """
        matches = list(self.HEADING_PATTERN.finditer(content))

        if not matches:
            # 没有标题，整个文档作为一个部分
            return [("", content)]

        sections: List[tuple[str, str]] = []

        # 处理第一个标题之前的内容
        if matches[0].start() > 0:
            pre_content = content[: matches[0].start()].strip()
            if pre_content:
                sections.append(("", pre_content))

        # 处理每个标题及其内容
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

            section_content = content[start:end].strip()
            if section_content:
                sections.append((title, section_content))

        return sections

    def _split_large_section(
        self, content: str, title: str, metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        分割过大的章节

        策略：
        1. 优先在代码块边界分割
        2. 其次在段落边界分割
        3. 确保示例代码与前置说明在一起
        """
        chunks: List[Chunk] = []

        # 识别代码块
        code_block_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        parts: List[str] = []
        last_end = 0

        for match in code_block_pattern.finditer(content):
            # 代码块前的内容
            if match.start() > last_end:
                pre_text = content[last_end : match.start()]
                parts.extend(self._split_by_paragraphs(pre_text))

            # 代码块本身（尝试与前一个非代码部分合并）
            code_block = match.group()
            if parts and self.count_tokens(parts[-1] + "\n" + code_block) <= self.max_tokens:
                parts[-1] = parts[-1] + "\n" + code_block
            else:
                parts.append(code_block)

            last_end = match.end()

        # 剩余内容
        if last_end < len(content):
            remaining = content[last_end:]
            parts.extend(self._split_by_paragraphs(remaining))

        # 合并小片段，确保不超过 max_tokens
        current_chunk = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if not current_chunk:
                current_chunk = part
            elif self.count_tokens(current_chunk + "\n\n" + part) <= self.max_tokens:
                current_chunk = current_chunk + "\n\n" + part
            else:
                if current_chunk:
                    chunks.append(
                        Chunk(content=current_chunk, metadata=metadata.copy())
                    )
                current_chunk = part

        if current_chunk:
            chunks.append(Chunk(content=current_chunk, metadata=metadata.copy()))

        return chunks

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        paragraphs = re.split(r"\n\n+", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _extract_symbol(self, title: str, content: str) -> str:
        """提取 API 符号（函数名/类名）"""
        # 优先从标题提取
        if title:
            # 常见的标题格式: "FunctionName", "ClassName.MethodName", "function_name()"
            symbol_match = re.search(r"[\w.]+(?:\s*\([^)]*\))?", title)
            if symbol_match:
                symbol = symbol_match.group().strip()
                # 移除括号和参数
                symbol = re.sub(r"\s*\([^)]*\)", "", symbol)
                if symbol:
                    return symbol

        # 从内容中提取 API 签名
        for pattern in self._api_patterns:
            match = pattern.search(content)
            if match:
                return match.group(1)

        return ""

    def _extract_module(self, source_file: str) -> str:
        """从文件路径提取模块名"""
        if not source_file:
            return ""

        path = Path(source_file)
        # 取父目录名作为模块名
        parts = path.parts

        # 过滤掉常见的无意义目录名
        skip_dirs = {"html", "htm", "docs", "doc", "pages", "content", "extracted"}
        meaningful_parts = [p for p in parts[:-1] if p.lower() not in skip_dirs]

        if meaningful_parts:
            return "/".join(meaningful_parts[-2:])  # 取最后两级目录
        return ""

    def _extract_chapter(self, source_file: str) -> str:
        """从文件路径提取章节路径"""
        if not source_file:
            return ""

        path = Path(source_file)
        # 返回相对路径（不含扩展名）
        return str(path.with_suffix(""))

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """合并过小的相邻 chunks"""
        if not chunks:
            return chunks

        merged: List[Chunk] = []
        current: Optional[Chunk] = None

        for chunk in chunks:
            if current is None:
                current = chunk
                continue

            current_tokens = self.count_tokens(current.content)
            chunk_tokens = self.count_tokens(chunk.content)

            # 如果当前 chunk 太小，且合并后不超过限制，则合并
            if current_tokens < self.min_tokens:
                combined_tokens = self.count_tokens(
                    current.content + "\n\n" + chunk.content
                )
                if combined_tokens <= self.max_tokens:
                    # 合并
                    current = Chunk(
                        content=current.content + "\n\n" + chunk.content,
                        metadata=current.metadata,  # 保留第一个的元数据
                    )
                    continue

            # 不合并，保存当前 chunk
            merged.append(current)
            current = chunk

        if current:
            merged.append(current)

        return merged

    def chunk_documents(self, docs: List[CleanedDocument]) -> List[Chunk]:
        """对多个文档进行切块"""
        all_chunks: List[Chunk] = []
        for doc in docs:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def save_chunks(self, chunks: List[Chunk], output_path: Path) -> None:
        """保存 chunks 到 JSON 文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [chunk.to_dict() for chunk in chunks]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_chunks(self, input_path: Path) -> List[Chunk]:
        """从 JSON 文件加载 chunks"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Chunk.from_dict(d) for d in data]
