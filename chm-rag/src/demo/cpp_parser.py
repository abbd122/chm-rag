"""C++ 代码解析器"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class CppSymbol:
    """C++ 符号（函数/类/方法）"""

    name: str  # 符号名称
    kind: str  # "function" | "class" | "method" | "struct"
    signature: str  # 完整签名
    body: str  # 函数体/类定义
    comments: str  # 相关注释
    line_start: int  # 起始行号
    line_end: int  # 结束行号
    class_name: Optional[str] = None  # 所属类名（仅方法有）


@dataclass
class ApiCall:
    """API 调用点"""

    api_name: str  # 被调用的 API 名称
    context: str  # 调用上下文（前后 N 行）
    line_number: int  # 行号
    full_statement: str  # 完整调用语句


class CppParser:
    """C++ 代码解析器"""

    # 函数/方法定义模式（简化版，避免灾难性回溯）
    FUNCTION_PATTERN = re.compile(
        r"""
        # 返回类型和函数名（简化匹配）
        (?P<signature>
            ^\s*
            (?:template\s*<[^>]*>\s*)?           # 模板（可选）
            (?:(?:static|virtual|inline|explicit|constexpr|override|const|extern|__declspec\([^)]*\))\s+)*  # 修饰符
            [\w:*&<>]+                           # 返回类型（不含空格）
            (?:\s+[\w:*&<>]+)*                   # 更多类型词（如 unsigned int）
            \s+
            (?P<name>(?:\w+::)*\w+)              # 函数名（可能含类名::）
            \s*
            \([^)]*\)                            # 参数列表
            (?:\s*const)?                        # const 方法
            (?:\s*noexcept)?                     # noexcept
            (?:\s*override)?                     # override
        )
        \s*
        (?P<body>\{)                             # 函数体开始
        """,
        re.MULTILINE | re.VERBOSE,
    )

    # 类/结构体定义模式（简化版）
    CLASS_PATTERN = re.compile(
        r"""
        # class/struct 定义
        (?P<signature>
            ^\s*
            (?:template\s*<[^>]*>\s*)?           # 模板
            (?P<kind>class|struct)
            \s+
            (?:__declspec\([^)]*\)\s+)?          # declspec
            (?P<name>\w+)                        # 类名
            (?:\s*:\s*[^{]+)?                    # 继承列表
        )
        \s*
        (?P<body>\{)                             # 类体开始
        """,
        re.MULTILINE | re.VERBOSE,
    )

    # API 调用模式（常见的 API 调用特征）
    API_CALL_PATTERNS = [
        # 对象方法调用: obj->Method() 或 obj.Method()
        re.compile(r"(\w+)\s*(?:->|\.)\s*(\w+)\s*\([^)]*\)"),
        # 命名空间函数调用: Namespace::Function()
        re.compile(r"(\w+(?:::\w+)+)\s*\([^)]*\)"),
        # 全局函数调用（大写开头，可能是 API）
        re.compile(r"\b([A-Z]\w+)\s*\([^)]*\)"),
    ]

    # 单行注释
    LINE_COMMENT_PATTERN = re.compile(r"//[^\n]*")
    # 块注释
    BLOCK_COMMENT_PATTERN = re.compile(r"/\*[\s\S]*?\*/")

    def __init__(self, context_lines: int = 5):
        """
        初始化 C++ 解析器

        Args:
            context_lines: API 调用上下文的行数（前后各 N 行）
        """
        self.context_lines = context_lines
        self._line_index: List[int] = []
        self._current_content: str = ""

    def _build_line_index(self, content: str) -> None:
        """预计算行号索引（每行的起始位置）"""
        if content is self._current_content:
            return  # 已缓存

        self._current_content = content
        self._line_index = [0]  # 第1行从位置0开始

        for i, c in enumerate(content):
            if c == '\n':
                self._line_index.append(i + 1)

    def _get_line_number(self, position: int) -> int:
        """用二分查找获取位置对应的行号"""
        # 二分查找
        left, right = 0, len(self._line_index) - 1
        while left < right:
            mid = (left + right + 1) // 2
            if self._line_index[mid] <= position:
                left = mid
            else:
                right = mid - 1
        return left + 1  # 行号从1开始

    def parse_file(self, file_path: Path) -> Tuple[List[CppSymbol], List[ApiCall]]:
        """
        解析 C++ 文件

        Args:
            file_path: 文件路径

        Returns:
            (符号列表, API 调用列表)
        """
        file_path = Path(file_path)
        content = self._read_file(file_path)

        if not content:
            return [], []

        symbols = self._extract_symbols(content)
        api_calls = self._extract_api_calls(content)

        return symbols, api_calls

    def _read_file(self, file_path: Path) -> Optional[str]:
        """读取文件内容"""
        encodings = ["utf-8", "gbk", "gb2312", "latin-1"]

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue

        return None

    def _extract_symbols(self, content: str) -> List[CppSymbol]:
        """提取符号（函数、类、方法）"""
        symbols: List[CppSymbol] = []

        # 预计算行号索引
        self._build_line_index(content)

        # 提取类定义
        for match in self.CLASS_PATTERN.finditer(content):
            name = match.group("name")
            kind = match.group("kind")
            signature = match.group("signature").strip()

            # 使用二分查找计算行号
            line_start = self._get_line_number(match.start())

            # 找到匹配的右括号
            body_start = match.end() - 1
            body_end = self._find_matching_brace(content, body_start)
            body = content[body_start : body_end + 1] if body_end > body_start else ""

            line_end = self._get_line_number(body_end) if body_end > 0 else line_start

            symbols.append(
                CppSymbol(
                    name=name,
                    kind=kind,
                    signature=signature,
                    body=body[:2000],  # 限制长度
                    comments="",
                    line_start=line_start,
                    line_end=line_end,
                )
            )

        # 提取函数/方法定义
        for match in self.FUNCTION_PATTERN.finditer(content):
            full_name = match.group("name")
            signature = match.group("signature").strip()

            # 解析类名::方法名
            if "::" in full_name:
                parts = full_name.rsplit("::", 1)
                class_name = parts[0]
                name = parts[1]
                kind = "method"
            else:
                class_name = None
                name = full_name
                kind = "function"

            # 使用二分查找计算行号
            line_start = self._get_line_number(match.start())

            # 找到匹配的右括号
            body_start = match.end() - 1
            body_end = self._find_matching_brace(content, body_start)
            body = content[body_start : body_end + 1] if body_end > body_start else ""

            line_end = self._get_line_number(body_end) if body_end > 0 else line_start

            symbols.append(
                CppSymbol(
                    name=name,
                    kind=kind,
                    signature=signature,
                    body=body[:2000],  # 限制长度
                    comments="",
                    line_start=line_start,
                    line_end=line_end,
                    class_name=class_name,
                )
            )

        return symbols

    def _extract_api_calls(self, content: str) -> List[ApiCall]:
        """提取 API 调用"""
        api_calls: List[ApiCall] = []
        lines = content.split("\n")
        seen_calls = set()  # 避免重复

        for line_num, line in enumerate(lines, 1):
            # 跳过注释行
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*"):
                continue

            for pattern in self.API_CALL_PATTERNS:
                for match in pattern.finditer(line):
                    api_name = match.group(0)
                    # 提取纯函数名
                    if "::" in api_name:
                        api_name = api_name.split("(")[0]
                    elif "->" in api_name or "." in api_name:
                        # obj->Method() 取 Method
                        api_name = match.group(2) if match.lastindex >= 2 else match.group(1)
                    else:
                        api_name = api_name.split("(")[0]

                    # 去重
                    call_key = (api_name, line_num)
                    if call_key in seen_calls:
                        continue
                    seen_calls.add(call_key)

                    # 获取上下文
                    start_line = max(0, line_num - 1 - self.context_lines)
                    end_line = min(len(lines), line_num + self.context_lines)
                    context = "\n".join(lines[start_line:end_line])

                    api_calls.append(
                        ApiCall(
                            api_name=api_name,
                            context=context,
                            line_number=line_num,
                            full_statement=line.strip(),
                        )
                    )

        return api_calls

    def _find_matching_brace(self, content: str, start: int) -> int:
        """找到匹配的右括号位置"""
        if start >= len(content) or content[start] != "{":
            return start

        depth = 1
        i = start + 1
        in_string = False
        in_char = False
        string_char = None

        while i < len(content) and depth > 0:
            c = content[i]
            prev = content[i - 1] if i > 0 else ""

            # 处理字符串
            if c in "\"'" and prev != "\\":
                if not in_string and not in_char:
                    in_string = c == '"'
                    in_char = c == "'"
                    string_char = c
                elif (in_string and c == '"') or (in_char and c == "'"):
                    in_string = False
                    in_char = False

            # 只在非字符串中计数
            if not in_string and not in_char:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1

            i += 1

        return i - 1 if depth == 0 else start

    def _clean_comments(self, comments: str) -> str:
        """清理注释文本"""
        if not comments:
            return ""

        # 移除 // 前缀
        lines = []
        for line in comments.split("\n"):
            line = line.strip()
            if line.startswith("//"):
                line = line[2:].strip()
            elif line.startswith("/*"):
                line = line[2:].strip()
            elif line.startswith("*"):
                line = line[1:].strip()
            elif line.endswith("*/"):
                line = line[:-2].strip()
            if line:
                lines.append(line)

        return "\n".join(lines)

    def parse_header(self, file_path: Path) -> List[CppSymbol]:
        """
        解析头文件（只提取声明，不需要函数体）

        Args:
            file_path: 头文件路径

        Returns:
            符号列表
        """
        content = self._read_file(file_path)
        if not content:
            return []

        # 预计算行号索引
        self._build_line_index(content)

        symbols: List[CppSymbol] = []

        # 函数声明模式（简化版，避免灾难性回溯）
        decl_pattern = re.compile(
            r"""
            # 函数声明
            (?P<signature>
                ^\s*
                (?:(?:static|virtual|inline|explicit|const|extern|__declspec\([^)]*\))\s+)*
                [\w:*&<>]+                           # 返回类型（不含空格）
                (?:\s+[\w:*&<>]+)*                   # 更多类型词
                \s+
                (?P<name>\w+)
                \s*
                \([^)]*\)
                (?:\s*const)?
            )
            \s*;
            """,
            re.MULTILINE | re.VERBOSE,
        )

        for match in decl_pattern.finditer(content):
            name = match.group("name")
            signature = match.group("signature").strip()
            line_start = self._get_line_number(match.start())

            symbols.append(
                CppSymbol(
                    name=name,
                    kind="function",
                    signature=signature,
                    body="",  # 声明没有函数体
                    comments="",
                    line_start=line_start,
                    line_end=line_start,
                )
            )

        return symbols
