"""TCMD 文件解析器"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TcmdCommand:
    """TCMD 命令"""

    command: str  # 命令内容
    description: str  # 命令说明（如有）
    line_number: int  # 行号


class TcmdParser:
    """TCMD 文件解析器

    只解析包含有效命令的文件，用于构建/测试/运行说明。
    """

    # 命令特征模式（用于判断是否包含有效命令）
    COMMAND_INDICATORS = [
        r"\b(run|execute|build|compile|test|start|call|invoke)\b",
        r"\b(msbuild|devenv|cmake|make|nmake)\b",
        r"\b(python|node|npm|pip)\b",
        r"\.exe\b",
        r"--\w+",  # 命令行参数
        r"-[a-zA-Z]",  # 短参数
    ]

    # 注释模式
    COMMENT_PATTERNS = [
        re.compile(r"^\s*#"),  # Shell 注释
        re.compile(r"^\s*//"),  # C++ 风格注释
        re.compile(r"^\s*REM\b", re.IGNORECASE),  # Batch 注释
        re.compile(r"^\s*;"),  # 某些配置文件注释
    ]

    def __init__(self, min_commands: int = 1):
        """
        初始化解析器

        Args:
            min_commands: 最少命令数量（低于此值认为无效）
        """
        self.min_commands = min_commands
        self._indicator_pattern = re.compile(
            "|".join(self.COMMAND_INDICATORS), re.IGNORECASE
        )

    def parse_file(self, file_path: Path) -> List[TcmdCommand]:
        """
        解析 .tcmd 文件

        Args:
            file_path: TCMD 文件路径

        Returns:
            命令列表（如果文件无效则返回空列表）
        """
        file_path = Path(file_path)
        content = self._read_file(file_path)

        if not content:
            return []

        commands = self._parse_content(content)

        # 检查是否达到最小命令数量
        if len(commands) < self.min_commands:
            return []

        return commands

    def _read_file(self, file_path: Path) -> Optional[str]:
        """读取文件"""
        encodings = ["utf-8", "gbk", "latin-1"]

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue

        return None

    def _parse_content(self, content: str) -> List[TcmdCommand]:
        """解析内容"""
        commands: List[TcmdCommand] = []
        current_description = ""

        for line_num, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()

            if not stripped:
                continue

            # 检查是否是注释
            is_comment = any(p.match(stripped) for p in self.COMMENT_PATTERNS)

            if is_comment:
                # 提取注释内容作为下一条命令的说明
                for p in self.COMMENT_PATTERNS:
                    stripped = p.sub("", stripped).strip()
                if stripped:
                    current_description = stripped
                continue

            # 检查是否包含命令特征
            if self._is_command_line(stripped):
                commands.append(
                    TcmdCommand(
                        command=stripped,
                        description=current_description,
                        line_number=line_num,
                    )
                )
                current_description = ""

        return commands

    def _is_command_line(self, line: str) -> bool:
        """判断是否是命令行"""
        # 检查是否包含命令指示符
        if self._indicator_pattern.search(line):
            return True

        # 检查是否以常见命令开头
        first_word = line.split()[0] if line.split() else ""
        common_commands = {
            "cd",
            "dir",
            "ls",
            "echo",
            "set",
            "export",
            "copy",
            "move",
            "del",
            "rm",
            "mkdir",
            "rmdir",
            "call",
            "start",
            "if",
            "for",
            "goto",
        }
        if first_word.lower() in common_commands:
            return True

        return False

    def is_valid_tcmd(self, file_path: Path) -> bool:
        """
        检查文件是否是有效的 TCMD 文件

        Args:
            file_path: 文件路径

        Returns:
            是否有效
        """
        commands = self.parse_file(file_path)
        return len(commands) >= self.min_commands

    def get_commands_text(self, commands: List[TcmdCommand]) -> str:
        """
        将命令列表转换为文本

        Args:
            commands: 命令列表

        Returns:
            格式化的文本
        """
        lines = []
        for cmd in commands:
            if cmd.description:
                lines.append(f"# {cmd.description}")
            lines.append(cmd.command)
            lines.append("")

        return "\n".join(lines)
