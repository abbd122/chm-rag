"""Demo 切块器"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..chunker import Chunk
from .scanner import ScannedFile
from .cpp_parser import CppParser, CppSymbol, ApiCall
from .ui_parser import UIParser, UIFile
from .def_parser import DefParser, DefExport
from .tcmd_parser import TcmdParser, TcmdCommand


class DemoChunker:
    """Demo 工程切块器

    将 Demo 代码解析结果转换为 RAG Chunks。
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
        context_lines: int = 5,
    ):
        """
        初始化 Demo 切块器

        Args:
            max_chunk_size: 单个 chunk 最大字符数
            context_lines: API 调用上下文行数
        """
        self.max_chunk_size = max_chunk_size
        self.cpp_parser = CppParser(context_lines=context_lines)
        self.ui_parser = UIParser()
        self.def_parser = DefParser()
        self.tcmd_parser = TcmdParser()

    def chunk_file(self, file: ScannedFile) -> List[Chunk]:
        """
        对单个文件进行切块

        Args:
            file: 扫描到的文件

        Returns:
            Chunk 列表
        """
        ext = file.ext.lower()

        if ext in {".h", ".hpp"}:
            return self._chunk_header(file)
        elif ext in {".cpp", ".cc", ".cxx"}:
            return self._chunk_source(file)
        elif ext == ".ui":
            return self._chunk_ui(file)
        elif ext == ".def":
            return self._chunk_def(file)
        elif ext == ".tcmd":
            return self._chunk_tcmd(file)
        else:
            return []

    def chunk_files(self, files: List[ScannedFile], show_progress: bool = False) -> List[Chunk]:
        """
        批量切块

        Args:
            files: 文件列表
            show_progress: 是否显示进度

        Returns:
            所有 Chunk
        """
        all_chunks: List[Chunk] = []

        for i, file in enumerate(files):
            if show_progress:
                print(f"{i}: {file.relative_path}")

            try:
                chunks = self.chunk_file(file)
                all_chunks.extend(chunks)
            except Exception as e:
                if show_progress:
                    print(f"      跳过 {file.relative_path}: {e}")

        return all_chunks

    def _chunk_header(self, file: ScannedFile) -> List[Chunk]:
        """处理头文件"""
        chunks: List[Chunk] = []

        # 解析符号
        symbols = self.cpp_parser.parse_header(file.path)

        # 为每个符号创建 chunk
        for symbol in symbols:
            content = self._format_symbol(symbol, file)
            metadata = self._create_metadata(
                file=file,
                symbol=symbol.name,
                chunk_type="function",
                language="cpp",
                content=content,
            )

            chunks.append(Chunk(content=content, metadata=metadata))

        # 如果没有解析出符号，创建整体 chunk
        if not symbols:
            content = self._read_file_content(file.path)
            if content:
                content = self._truncate(content)
                metadata = self._create_metadata(
                    file=file,
                    symbol=file.path.stem,
                    chunk_type="function",
                    language="cpp",
                    content=content,
                )
                chunks.append(Chunk(content=content, metadata=metadata))

        return chunks

    def _chunk_source(self, file: ScannedFile) -> List[Chunk]:
        """处理源文件"""
        chunks: List[Chunk] = []

        # 解析符号和 API 调用
        symbols, api_calls = self.cpp_parser.parse_file(file.path)

        # 为每个函数/方法创建 chunk
        for symbol in symbols:
            content = self._format_symbol(symbol, file)
            metadata = self._create_metadata(
                file=file,
                symbol=symbol.name,
                chunk_type="function",
                language="cpp",
                content=content,
            )

            if symbol.class_name:
                metadata["class_name"] = symbol.class_name

            chunks.append(Chunk(content=content, metadata=metadata))

        # 为 API 调用创建独立 chunk
        # 去重并限制数量
        seen_apis = set()
        api_chunks = 0
        max_api_chunks = 20  # 每个文件最多 20 个 API 调用 chunk

        for call in api_calls:
            if call.api_name in seen_apis:
                continue
            if api_chunks >= max_api_chunks:
                break

            seen_apis.add(call.api_name)
            api_chunks += 1

            content = self._format_api_call(call, file)
            metadata = self._create_metadata(
                file=file,
                symbol=call.api_name,
                chunk_type="call_snippet",
                language="cpp",
                content=content,
            )
            metadata["line_number"] = call.line_number

            chunks.append(Chunk(content=content, metadata=metadata))

        return chunks

    def _chunk_ui(self, file: ScannedFile) -> List[Chunk]:
        """处理 UI 文件"""
        chunks: List[Chunk] = []

        ui_file = self.ui_parser.parse_file(file.path)
        if not ui_file:
            return chunks

        # 创建整体 UI 结构 chunk
        content = self._format_ui(ui_file, file)
        metadata = self._create_metadata(
            file=file,
            symbol=ui_file.class_name or file.path.stem,
            chunk_type="ui_structure",
            language="qt_ui",
            content=content,
        )

        chunks.append(Chunk(content=content, metadata=metadata))

        # 为重要的控件创建独立 chunk
        all_widgets = self.ui_parser.get_flat_widgets(ui_file.root_widget)
        important_widgets = [
            w for w in all_widgets
            if w.object_name and w.class_name not in {"QWidget", "QFrame", "QGroupBox"}
        ]

        for widget in important_widgets[:10]:  # 限制数量
            widget_content = f"""UI 控件: {widget.object_name}
类型: {widget.class_name}
文件: {file.relative_path}

属性:
{self._format_properties(widget.properties)}
"""
            widget_metadata = self._create_metadata(
                file=file,
                symbol=widget.object_name,
                chunk_type="ui_structure",
                language="qt_ui",
                content=widget_content,
            )

            chunks.append(Chunk(content=widget_content, metadata=widget_metadata))

        return chunks

    def _chunk_def(self, file: ScannedFile) -> List[Chunk]:
        """处理 DEF 文件"""
        exports = self.def_parser.parse_file(file.path)

        if not exports:
            return []

        # 只创建一个汇总 chunk
        export_names = [exp.name for exp in exports]
        content = f"""导出符号列表
文件: {file.relative_path}
工程: {file.demo_name}

共 {len(exports)} 个导出符号:
{chr(10).join(f'- {name}' for name in export_names[:50])}
{'...' if len(export_names) > 50 else ''}
"""

        metadata = self._create_metadata(
            file=file,
            symbol="exports",
            chunk_type="exports",
            language="cpp",
            content=content,
        )

        return [Chunk(content=content, metadata=metadata)]

    def _chunk_tcmd(self, file: ScannedFile) -> List[Chunk]:
        """处理 TCMD 文件"""
        commands = self.tcmd_parser.parse_file(file.path)

        if not commands:
            return []

        # 创建命令 chunk
        content = f"""命令脚本: {file.relative_path}
工程: {file.demo_name}

命令列表:
"""
        for cmd in commands:
            if cmd.description:
                content += f"\n# {cmd.description}\n"
            content += f"{cmd.command}\n"

        metadata = self._create_metadata(
            file=file,
            symbol=file.path.stem,
            chunk_type="command",
            language="tcmd",
            content=content,
        )

        return [Chunk(content=content, metadata=metadata)]

    def _format_symbol(self, symbol: CppSymbol, file: ScannedFile) -> str:
        """格式化符号为文本"""
        parts = []

        # 来源信息
        parts.append(f"文件: {file.relative_path}")
        parts.append(f"工程: {file.demo_name}")

        if symbol.class_name:
            parts.append(f"类: {symbol.class_name}")

        parts.append("")

        # 注释
        if symbol.comments:
            parts.append(f"/* {symbol.comments} */")

        # 签名
        parts.append(symbol.signature)

        # 函数体
        if symbol.body:
            body = self._truncate(symbol.body, 1500)
            parts.append(body)

        return "\n".join(parts)

    def _format_api_call(self, call: ApiCall, file: ScannedFile) -> str:
        """格式化 API 调用"""
        return f"""API 调用示例: {call.api_name}
文件: {file.relative_path}
工程: {file.demo_name}
行号: {call.line_number}

调用上下文:
```cpp
{call.context}
```

调用语句:
{call.full_statement}
"""

    def _format_ui(self, ui_file: UIFile, file: ScannedFile) -> str:
        """格式化 UI 文件"""
        parts = [
            f"Qt UI 文件: {file.relative_path}",
            f"工程: {file.demo_name}",
            f"类名: {ui_file.class_name}",
            "",
            "控件层级结构:",
            self.ui_parser.widget_to_text(ui_file.root_widget),
        ]

        if ui_file.connections:
            parts.append("")
            parts.append("信号槽连接:")
            for conn in ui_file.connections:
                parts.append(f"  {conn.sender}.{conn.signal} -> {conn.receiver}.{conn.slot}")

        return "\n".join(parts)

    def _format_properties(self, properties: Dict[str, str]) -> str:
        """格式化属性"""
        if not properties:
            return "  (无)"
        return "\n".join(f"  {k}: {v}" for k, v in properties.items())

    def _create_metadata(
        self,
        file: ScannedFile,
        symbol: str,
        chunk_type: str,
        language: str,
        content: str,
    ) -> Dict[str, Any]:
        """创建 chunk metadata"""
        return {
            "source_type": "demo",
            "demo_name": file.demo_name,
            "file_path": file.relative_path,
            "ext": file.ext,
            "symbol": symbol,
            "chunk_type": chunk_type,
            "language": language,
            "hash": self._hash_content(content),
        }

    def _hash_content(self, content: str) -> str:
        """计算内容 hash"""
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _truncate(self, text: str, max_len: Optional[int] = None) -> str:
        """截断文本"""
        max_len = max_len or self.max_chunk_size
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """读取文件内容"""
        encodings = ["utf-8", "gbk", "latin-1"]
        for enc in encodings:
            try:
                return file_path.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return None

    def save_chunks(self, chunks: List[Chunk], output_path: Path) -> None:
        """保存 chunks 到 JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [{"content": c.content, "metadata": c.metadata} for c in chunks]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_chunks(self, input_path: Path) -> List[Chunk]:
        """从 JSON 加载 chunks"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [Chunk(content=d["content"], metadata=d["metadata"]) for d in data]
