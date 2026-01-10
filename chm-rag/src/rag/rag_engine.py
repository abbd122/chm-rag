"""RAG 引擎 - 核心模块"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..chunker import Chunk, SemanticChunker
from ..cleaner import CleanedDocument, HTMLCleaner
from ..demo import DemoScanner, DemoChunker
from ..embedding import BaseEmbedding, OpenAIEmbedding
from ..llm import BaseLLM, OpenAILLM
from ..unpacker import CHMUnpacker
from ..vectorstore import BaseVectorStore, FAISSVectorStore, SearchResult
from ..config import get_config


# 固定的 Prompt 模板（按 PRD 要求）
SYSTEM_PROMPT = """你是一个专业的开发文档助手，负责解答开发相关问题。

## 来源说明
- 【CHM】标记的片段来自官方 API 文档，是权威规范说明
- 【Demo】标记的片段来自示例工程，是真实使用示例

## 回答格式要求（必须严格遵守）

你的回答必须明确区分信息来源，使用以下标记：

1. **文档明确记载的内容**：直接陈述，无需特殊标记
   - 这是从检索到的文档片段中直接获取的信息
   - 包括 API 名称、参数、返回值、功能说明等

2. **模型推测/补充的内容**：必须用 `[推测]` 标记
   - 文档未明确说明，但基于上下文或工程经验推断的内容
   - 格式示例：`[推测] 该参数可能需要先调用 Init 函数初始化`

3. **通用工程建议**：必须用 `[建议]` 标记
   - 与具体 API 无关的通用编程建议
   - 格式示例：`[建议] 建议在调用前进行空指针检查`

## 回答原则
1. 优先使用文档中的准确描述，直接引用原文
2. 当文档信息不完整时，明确标注哪些是推测
3. CHM 与 Demo 冲突时，以 CHM 官方文档为准
4. 即使未找到完全匹配的内容，也应给出标注过的推测性建议"""

RAG_PROMPT_TEMPLATE = """【参考文档片段】
{retrieved_chunks}

{history_section}【用户问题】
{user_question}

请根据以上文档片段回答问题。

回答要求：
1. 文档中明确记载的内容 → 直接陈述
2. 文档未提及但你推测的内容 → 必须标注 [推测]
3. 通用工程建议 → 必须标注 [建议]

请确保用户能清楚分辨哪些信息来自文档、哪些是你的推测。"""


@dataclass
class ChatMessage:
    """对话消息"""
    role: str  # "user" | "assistant"
    content: str


@dataclass
class RAGResponse:
    """RAG 回答结果"""

    answer: str  # 生成的回答
    sources: List[SearchResult] = field(default_factory=list)  # 来源文档片段
    query: str = ""  # 原始查询


class RAGEngine:
    """RAG 引擎，整合检索与生成"""

    def __init__(
        self,
        embedding: Optional[BaseEmbedding] = None,
        vectorstore: Optional[BaseVectorStore] = None,
        llm: Optional[BaseLLM] = None,
        top_k: int = 5,
    ):
        """
        初始化 RAG 引擎

        Args:
            embedding: Embedding 模块（默认使用 OpenAI）
            vectorstore: 向量存储模块（默认使用 FAISS）
            llm: LLM 模块（默认使用 OpenAI）
            top_k: 检索的文档数量
        """
        config = get_config()

        # 初始化 Embedding
        if embedding is None:
            embedding_config = config.embedding
            self.embedding = OpenAIEmbedding(
                model=embedding_config.get("model", "text-embedding-3-small"),
                batch_size=embedding_config.get("batch_size", 100),
            )
        else:
            self.embedding = embedding

        # 初始化向量存储
        if vectorstore is None:
            self.vectorstore = FAISSVectorStore(dimension=self.embedding.dimension)
        else:
            self.vectorstore = vectorstore

        # 初始化 LLM
        if llm is None:
            llm_config = config.llm
            self.llm = OpenAILLM(
                model=llm_config.get("model", "gpt-4o-mini"),
                temperature=llm_config.get("temperature", 0.1),
                max_tokens=llm_config.get("max_tokens", 2000),
            )
        else:
            self.llm = llm

        self.top_k = top_k

        # CHM 辅助模块
        config_7z_path = config.get("tools.7z_path", "")
        self.unpacker = CHMUnpacker(config_7z_path=config_7z_path if config_7z_path else None)
        self.cleaner = HTMLCleaner()
        self.chunker = SemanticChunker(
            max_tokens=config.rag.get("max_chunk_tokens", 800)
        )

        # Demo 辅助模块
        self.demo_scanner = DemoScanner()
        self.demo_chunker = DemoChunker()

    # ==================== CHM 相关方法 ====================

    def extract_chunks(
        self,
        chm_path: Path,
        output_dir: Optional[Path] = None,
        chunks_path: Optional[Path] = None,
        show_progress: bool = True,
    ) -> List[Chunk]:
        """
        提取 CHM 文档片段（不进行向量化）

        步骤: 解包 CHM → 清洗 HTML → 语义切块 → 保存 JSON

        Args:
            chm_path: CHM 文件路径
            output_dir: 解包输出目录（默认使用配置）
            chunks_path: chunks 保存路径（默认使用配置）
            show_progress: 是否显示进度

        Returns:
            切块列表
        """
        chm_path = Path(chm_path)
        config = get_config()

        if output_dir is None:
            output_dir = Path(config.get("chm.output_dir", "./data/chm_extracted"))

        if chunks_path is None:
            chunks_path = Path(config.get("paths.chunks_cache", "./data/chunks.json"))

        # 1. 解包 CHM
        if show_progress:
            print(f"[1/3] 解包 CHM 文件: {chm_path}")

        html_files = self.unpacker.unpack(chm_path, output_dir)
        if show_progress:
            print(f"      找到 {len(html_files)} 个 HTML 文件")

        # 2. 清洗 HTML
        if show_progress:
            print("[2/3] 清洗 HTML 文件...")

        cleaned_docs: List[CleanedDocument] = []
        for html_file in html_files:
            try:
                doc = self.cleaner.clean_file(html_file)
                if doc.content.strip():
                    cleaned_docs.append(doc)
            except Exception as e:
                if show_progress:
                    print(f"      跳过文件 {html_file}: {e}")

        if show_progress:
            print(f"      清洗完成，有效文档: {len(cleaned_docs)}")

        # 3. 语义切块
        if show_progress:
            print("[3/3] 语义切块...")

        chunks = self.chunker.chunk_documents(cleaned_docs)

        # 为 CHM chunks 添加 source_type
        for chunk in chunks:
            chunk.metadata["source_type"] = "chm"

        if show_progress:
            print(f"      生成 {len(chunks)} 个文档片段")

        # 保存 chunks 到 JSON
        self.chunker.save_chunks(chunks, chunks_path)
        if show_progress:
            print(f"      已保存到: {chunks_path}")

        return chunks

    def vectorize_chunks(
        self,
        chunks_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        show_progress: bool = True,
    ) -> int:
        """
        向量化并存储文档片段

        从 JSON 加载 chunks → 向量化 → 存储到 FAISS

        Args:
            chunks_path: chunks JSON 文件路径（默认使用配置）
            index_path: 索引保存路径（默认使用配置）
            show_progress: 是否显示进度

        Returns:
            向量化的 chunk 数量
        """
        config = get_config()

        if chunks_path is None:
            chunks_path = Path(config.get("paths.chunks_cache", "./data/chunks.json"))

        if index_path is None:
            index_path = Path(config.vectorstore.get("index_path", "./data/index"))

        # 加载 chunks
        if show_progress:
            print(f"[1/2] 加载文档片段: {chunks_path}")

        chunks = self.chunker.load_chunks(chunks_path)
        if show_progress:
            print(f"      加载 {len(chunks)} 个文档片段")

        # 向量化并存储
        if show_progress:
            print("[2/2] 向量化并存储...")

        contents = [chunk.content for chunk in chunks]
        vectors = self.embedding.embed(contents)
        self.vectorstore.add(chunks, vectors)

        # 保存索引
        self.vectorstore.save(index_path)
        if show_progress:
            print(f"      索引完成，共 {self.vectorstore.count()} 个向量")
            print(f"      已保存到: {index_path}")

        return len(chunks)

    def index_chm(
        self,
        chm_path: Path,
        output_dir: Optional[Path] = None,
        show_progress: bool = True,
    ) -> int:
        """
        索引 CHM 文件

        Args:
            chm_path: CHM 文件路径
            output_dir: 解包输出目录（默认使用配置）
            show_progress: 是否显示进度

        Returns:
            索引的 chunk 数量
        """
        chm_path = Path(chm_path)
        config = get_config()

        if output_dir is None:
            output_dir = Path(config.paths.get("chm_output", "./data/chm_extracted"))

        # 1. 解包 CHM
        if show_progress:
            print(f"[1/4] 解包 CHM 文件: {chm_path}")

        html_files = self.unpacker.unpack(chm_path, output_dir)
        if show_progress:
            print(f"      找到 {len(html_files)} 个 HTML 文件")

        # 2. 清洗 HTML
        if show_progress:
            print("[2/4] 清洗 HTML 文件...")

        cleaned_docs: List[CleanedDocument] = []
        for html_file in html_files:
            try:
                doc = self.cleaner.clean_file(html_file)
                if doc.content.strip():
                    cleaned_docs.append(doc)
            except Exception as e:
                if show_progress:
                    print(f"      跳过文件 {html_file}: {e}")

        if show_progress:
            print(f"      清洗完成，有效文档: {len(cleaned_docs)}")

        # 3. 语义切块
        if show_progress:
            print("[3/4] 语义切块...")

        chunks = self.chunker.chunk_documents(cleaned_docs)

        # 为 CHM chunks 添加 source_type
        for chunk in chunks:
            chunk.metadata["source_type"] = "chm"

        if show_progress:
            print(f"      生成 {len(chunks)} 个文档片段")

        # 4. 向量化并存储
        if show_progress:
            print("[4/4] 向量化并存储...")

        contents = [chunk.content for chunk in chunks]
        vectors = self.embedding.embed(contents)
        self.vectorstore.add(chunks, vectors)

        if show_progress:
            print(f"      索引完成，共 {self.vectorstore.count()} 个向量")

        return len(chunks)

    # ==================== Demo 相关方法 ====================

    def extract_demo_chunks(
        self,
        demo_dir: Path,
        demo_name: Optional[str] = None,
        chunks_path: Optional[Path] = None,
        show_progress: bool = True,
    ) -> List[Chunk]:
        """
        提取 Demo 工程文档片段（不进行向量化）

        步骤: 扫描文件 → 解析代码 → 生成切块 → 保存 JSON

        Args:
            demo_dir: Demo 工程目录
            demo_name: 工程名称（默认使用目录名）
            chunks_path: chunks 保存路径
            show_progress: 是否显示进度

        Returns:
            切块列表
        """
        demo_dir = Path(demo_dir)
        config = get_config()

        if chunks_path is None:
            chunks_path = Path(config.get("demo.chunks_path", "./data/demo_chunks.json"))

        # 1. 扫描文件
        if show_progress:
            print(f"[1/2] 扫描 Demo 工程: {demo_dir}")

        files = self.demo_scanner.scan(demo_dir, demo_name=demo_name)
        stats = self.demo_scanner.get_file_stats(files)

        if show_progress:
            print(f"      找到 {stats['total']} 个有效文件")
            for ext, count in stats["by_extension"].items():
                print(f"        {ext}: {count}")

        # 2. 解析并切块
        if show_progress:
            print("[2/2] 解析并切块...")

        chunks = self.demo_chunker.chunk_files(files, show_progress=show_progress)

        if show_progress:
            print(f"      生成 {len(chunks)} 个文档片段")

        # 保存 chunks 到 JSON
        self.demo_chunker.save_chunks(chunks, chunks_path)
        if show_progress:
            print(f"      已保存到: {chunks_path}")

        return chunks

    def vectorize_demo_chunks(
        self,
        chunks_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        show_progress: bool = True,
    ) -> int:
        """
        向量化 Demo chunks 并添加到索引

        Args:
            chunks_path: Demo chunks JSON 路径
            index_path: 索引保存路径
            show_progress: 是否显示进度

        Returns:
            向量化的 chunk 数量
        """
        config = get_config()

        if chunks_path is None:
            chunks_path = Path(config.get("demo.chunks_path", "./data/demo_chunks.json"))

        if index_path is None:
            index_path = Path(config.vectorstore.get("index_path", "./data/index"))

        # 加载 chunks
        if show_progress:
            print(f"[1/2] 加载 Demo 文档片段: {chunks_path}")

        chunks = self.demo_chunker.load_chunks(chunks_path)
        if show_progress:
            print(f"      加载 {len(chunks)} 个文档片段")

        # 向量化并存储
        if show_progress:
            print("[2/2] 向量化并存储...")

        contents = [chunk.content for chunk in chunks]
        vectors = self.embedding.embed(contents)
        self.vectorstore.add(chunks, vectors)

        # 保存索引
        self.vectorstore.save(index_path)
        if show_progress:
            print(f"      索引完成，共 {self.vectorstore.count()} 个向量")
            print(f"      已保存到: {index_path}")

        return len(chunks)

    def index_demo(
        self,
        demo_dir: Path,
        demo_name: Optional[str] = None,
        show_progress: bool = True,
    ) -> int:
        """
        一键索引 Demo 工程

        Args:
            demo_dir: Demo 工程目录
            demo_name: 工程名称
            show_progress: 是否显示进度

        Returns:
            索引的 chunk 数量
        """
        demo_dir = Path(demo_dir)

        # 1. 扫描文件
        if show_progress:
            print(f"[1/3] 扫描 Demo 工程: {demo_dir}")

        files = self.demo_scanner.scan(demo_dir, demo_name=demo_name)
        stats = self.demo_scanner.get_file_stats(files)

        if show_progress:
            print(f"      找到 {stats['total']} 个有效文件")

        # 2. 解析并切块
        if show_progress:
            print("[2/3] 解析并切块...")

        chunks = self.demo_chunker.chunk_files(files, show_progress=show_progress)

        if show_progress:
            print(f"      生成 {len(chunks)} 个文档片段")

        # 3. 向量化并存储
        if show_progress:
            print("[3/3] 向量化并存储...")

        contents = [chunk.content for chunk in chunks]
        vectors = self.embedding.embed(contents)
        self.vectorstore.add(chunks, vectors)

        if show_progress:
            print(f"      索引完成，共 {self.vectorstore.count()} 个向量")

        return len(chunks)

    # ==================== 通用方法 ====================

    def index_chunks(self, chunks: List[Chunk], show_progress: bool = True) -> int:
        """
        直接索引 chunks

        Args:
            chunks: 文档片段列表
            show_progress: 是否显示进度

        Returns:
            索引的 chunk 数量
        """
        if not chunks:
            return 0

        if show_progress:
            print(f"向量化 {len(chunks)} 个文档片段...")

        contents = [chunk.content for chunk in chunks]
        vectors = self.embedding.embed(contents)
        self.vectorstore.add(chunks, vectors)

        if show_progress:
            print(f"索引完成，共 {self.vectorstore.count()} 个向量")

        return len(chunks)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        return_sources: bool = True,
        history: Optional[List[ChatMessage]] = None,
    ) -> RAGResponse:
        """
        查询问答

        Args:
            question: 用户问题
            top_k: 检索数量（默认使用初始化时的值）
            filters: 元数据过滤条件
            source: 数据源过滤 ("chm" | "demo" | None=全部)
            return_sources: 是否返回来源
            history: 对话历史（用于上下文关联）

        Returns:
            RAG 回答结果
        """
        if top_k is None:
            top_k = self.top_k

        # 处理 source 过滤
        if source:
            filters = filters or {}
            filters["source_type"] = source

        # 构建历史上下文部分
        history_section = ""
        if history:
            history_lines = ["【对话历史】"]
            for msg in history[-6:]:  # 最多保留最近 3 轮对话 (6 条消息)
                role_label = "用户" if msg.role == "user" else "助手"
                # 截断过长的历史消息
                content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                history_lines.append(f"{role_label}: {content}")
            history_lines.append("")
            history_section = "\n".join(history_lines) + "\n"

        # 1. 向量检索
        query_vector = self.embedding.embed_query(question)
        search_results = self.vectorstore.search(
            query_vector, top_k=top_k, filters=filters
        )

        # 2. 构造 Prompt
        if not search_results:
            no_doc_prompt = f"""【参考文档片段】
未检索到直接相关的文档片段。

{history_section}【用户问题】
{question}

请基于通用工程经验尝试回答此问题。如果是项目特定的 API 问题，请说明需要查阅具体文档，并给出可能的方向建议。"""
            answer = self.llm.generate(no_doc_prompt, system_prompt=SYSTEM_PROMPT)
            return RAGResponse(
                answer=answer,
                sources=[],
                query=question,
            )

        # 格式化检索到的文档片段
        chunks_text = self._format_chunks(search_results)
        prompt = RAG_PROMPT_TEMPLATE.format(
            retrieved_chunks=chunks_text,
            history_section=history_section,
            user_question=question,
        )

        # 3. 调用 LLM 生成回答
        answer = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)

        return RAGResponse(
            answer=answer,
            sources=search_results if return_sources else [],
            query=question,
        )

    def _format_chunks(self, results: List[SearchResult]) -> str:
        """格式化检索结果为 Prompt 文本"""
        formatted_parts = []

        for i, result in enumerate(results, 1):
            chunk = result.chunk
            metadata = chunk.metadata

            # 确定来源类型标记
            source_type = metadata.get("source_type", "chm")
            source_label = "【CHM】" if source_type == "chm" else "【Demo】"

            # 构建来源信息
            source_info = []
            if metadata.get("symbol"):
                source_info.append(f"符号: {metadata['symbol']}")
            if metadata.get("title"):
                source_info.append(f"标题: {metadata['title']}")
            if metadata.get("chapter"):
                source_info.append(f"章节: {metadata['chapter']}")
            if metadata.get("demo_name"):
                source_info.append(f"工程: {metadata['demo_name']}")
            if metadata.get("file_path"):
                source_info.append(f"文件: {metadata['file_path']}")

            header = f"{source_label} [片段 {i}]"
            if source_info:
                header += f" ({', '.join(source_info)})"

            formatted_parts.append(f"{header}\n{chunk.content}")

        return "\n\n---\n\n".join(formatted_parts)

    def save_index(self, path: Optional[Path] = None) -> None:
        """
        保存索引

        Args:
            path: 保存路径（默认使用配置）
        """
        config = get_config()
        if path is None:
            path = Path(config.vectorstore.get("index_path", "./data/index"))

        self.vectorstore.save(path)
        print(f"索引已保存到: {path}")

    def load_index(self, path: Optional[Path] = None) -> None:
        """
        加载索引

        Args:
            path: 索引路径（默认使用配置）
        """
        config = get_config()
        if path is None:
            path = Path(config.vectorstore.get("index_path", "./data/index"))

        self.vectorstore.load(path)
        print(f"已加载索引: {self.vectorstore.count()} 个向量")

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        # 统计各来源的 chunk 数量
        all_chunks = self.vectorstore.get_all_chunks()
        chm_count = sum(1 for c in all_chunks if c.metadata.get("source_type") != "demo")
        demo_count = sum(1 for c in all_chunks if c.metadata.get("source_type") == "demo")

        return {
            "total_chunks": self.vectorstore.count(),
            "chm_chunks": chm_count,
            "demo_chunks": demo_count,
            "embedding_model": getattr(self.embedding, "model", "unknown"),
            "embedding_dimension": self.embedding.dimension,
            "llm_model": getattr(self.llm, "model", "unknown"),
        }
