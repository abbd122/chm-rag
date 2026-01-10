"""DEF 文件解析器"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DefExport:
    """DEF 导出符号"""

    name: str  # 符号名
    ordinal: Optional[int] = None  # 序号（可选）
    is_data: bool = False  # 是否是数据符号


class DefParser:
    """MSVC .def 文件解析器"""

    # EXPORTS 段开始
    EXPORTS_PATTERN = re.compile(r"^\s*EXPORTS\s*$", re.IGNORECASE | re.MULTILINE)

    # 其他段落定义（用于终止 EXPORTS 解析）
    SECTION_PATTERN = re.compile(
        r"^\s*(LIBRARY|NAME|DESCRIPTION|STACKSIZE|HEAPSIZE|VERSION|SECTIONS)\b",
        re.IGNORECASE
    )

    # 导出符号模式
    # 格式: symbolName [@ordinal] [NONAME] [DATA] [PRIVATE]
    SYMBOL_PATTERN = re.compile(
        r"""
        ^\s*
        (?P<name>\w+)           # 符号名
        (?:\s*@\s*(?P<ordinal>\d+))?  # 可选的序号
        (?:\s+NONAME)?          # NONAME 标记
        (?:\s+(?P<data>DATA))?  # DATA 标记
        (?:\s+PRIVATE)?         # PRIVATE 标记
        \s*$
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    def parse_file(self, file_path: Path) -> List[DefExport]:
        """
        解析 .def 文件

        Args:
            file_path: DEF 文件路径

        Returns:
            导出符号列表
        """
        file_path = Path(file_path)
        content = self._read_file(file_path)

        if not content:
            return []

        return self._parse_content(content)

    def _read_file(self, file_path: Path) -> Optional[str]:
        """读取文件"""
        encodings = ["utf-8", "gbk", "latin-1"]

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue

        return None

    def _parse_content(self, content: str) -> List[DefExport]:
        """解析 DEF 内容"""
        exports: List[DefExport] = []

        # 查找 EXPORTS 段
        match = self.EXPORTS_PATTERN.search(content)
        if not match:
            return exports

        # 获取 EXPORTS 之后的内容
        exports_section = content[match.end() :]

        # 逐行解析
        for line in exports_section.split("\n"):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith(";"):
                continue

            # 遇到新的段落定义则停止
            if self.SECTION_PATTERN.match(line):
                break

            # 解析符号
            symbol_match = self.SYMBOL_PATTERN.match(line)
            if symbol_match:
                name = symbol_match.group("name")
                ordinal_str = symbol_match.group("ordinal")
                is_data = symbol_match.group("data") is not None

                ordinal = int(ordinal_str) if ordinal_str else None

                exports.append(
                    DefExport(
                        name=name,
                        ordinal=ordinal,
                        is_data=is_data,
                    )
                )

        return exports

    def get_export_names(self, file_path: Path) -> List[str]:
        """
        获取所有导出符号名

        Args:
            file_path: DEF 文件路径

        Returns:
            符号名列表
        """
        exports = self.parse_file(file_path)
        return [exp.name for exp in exports]
