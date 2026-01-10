"""HTML 清洗模块"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


@dataclass
class CleanedDocument:
    """清洗后的文档"""

    content: str  # 清洗后的 Markdown 内容
    title: str  # 文档标题
    source_file: str  # 源文件路径
    headings: List[str] = field(default_factory=list)  # 标题列表


class HTMLCleaner:
    """HTML 清洗器，将 HTML 转换为干净的 Markdown"""

    # 需要移除的标签
    REMOVE_TAGS = {
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
        "iframe",
        "form",
        "button",
        "input",
        "select",
        "textarea",
        "meta",
        "link",
    }

    # 需要移除的 class 名称模式
    REMOVE_CLASS_PATTERNS = [
        r"nav",
        r"menu",
        r"sidebar",
        r"footer",
        r"header",
        r"toc",
        r"table-of-contents",
        r"breadcrumb",
        r"pagination",
        r"advertisement",
        r"ad-",
        r"copyright",
    ]

    # 代码块标签
    CODE_TAGS = {"pre", "code"}

    def __init__(self, remove_toc: bool = True, keep_links: bool = False):
        """
        初始化 HTML 清洗器

        Args:
            remove_toc: 是否移除目录
            keep_links: 是否保留链接
        """
        self.remove_toc = remove_toc
        self.keep_links = keep_links
        self._class_pattern = re.compile(
            "|".join(self.REMOVE_CLASS_PATTERNS), re.IGNORECASE
        )

    def clean_file(self, file_path: Path, encoding: str = "utf-8") -> CleanedDocument:
        """
        清洗 HTML 文件

        Args:
            file_path: HTML 文件路径
            encoding: 文件编码

        Returns:
            清洗后的文档
        """
        file_path = Path(file_path)

        # 尝试不同编码读取
        content = None
        for enc in [encoding, "utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                content = file_path.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise ValueError(f"无法读取文件: {file_path}")

        return self.clean_html(content, source_file=str(file_path))

    def clean_html(self, html: str, source_file: str = "") -> CleanedDocument:
        """
        清洗 HTML 内容

        Args:
            html: HTML 内容
            source_file: 源文件路径（用于记录）

        Returns:
            清洗后的文档
        """
        soup = BeautifulSoup(html, "lxml")

        # 提取标题
        title = self._extract_title(soup)

        # 移除不需要的元素
        # self._remove_unwanted_elements(soup)

        # 转换为 Markdown
        content, headings = self._to_markdown(soup)

        # 清理多余空白
        content = self._clean_whitespace(content)

        return CleanedDocument(
            content=content,
            title=title,
            source_file=source_file,
            headings=headings,
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """提取文档标题"""
        # 优先从 <title> 标签获取
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # 其次从 <h1> 获取
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        return ""

    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """移除不需要的元素"""
        # 移除注释
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 移除指定标签
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # 移除匹配 class 模式的元素
        for tag in soup.find_all(True):
            if not isinstance(tag, Tag):
                continue
            classes = tag.get("class", [])
            if isinstance(classes, str):
                classes = [classes]
            for cls in classes:
                if self._class_pattern.search(cls):
                    tag.decompose()
                    break

        # 移除目录相关元素
        if self.remove_toc:
            self._remove_toc_elements(soup)

    def _remove_toc_elements(self, soup: BeautifulSoup) -> None:
        """移除目录相关元素"""
        # 常见的目录 ID
        toc_ids = ["toc", "table-of-contents", "contents", "menu", "sidebar"]
        for toc_id in toc_ids:
            for tag in soup.find_all(id=re.compile(toc_id, re.IGNORECASE)):
                tag.decompose()

    def _to_markdown(self, soup: BeautifulSoup) -> tuple[str, List[str]]:
        """将 HTML 转换为 Markdown"""
        body = soup.find("body") or soup
        lines: List[str] = []
        headings: List[str] = []

        def process_element(element, in_code_block: bool = False) -> None:
            if isinstance(element, NavigableString):
                text = str(element)
                if not in_code_block:
                    text = text.strip()
                if text:
                    lines.append(text)
                return

            if not isinstance(element, Tag):
                return

            tag_name = element.name.lower()

            # 处理标题
            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag_name[1])
                text = element.get_text(strip=True)
                if text:
                    lines.append("")
                    lines.append(f"{'#' * level} {text}")
                    lines.append("")
                    headings.append(text)
                return

            # 处理代码块
            if tag_name == "pre":
                code_text = element.get_text()
                lines.append("")
                lines.append("```")
                lines.append(code_text.rstrip())
                lines.append("```")
                lines.append("")
                return

            # 处理行内代码
            if tag_name == "code" and element.parent.name != "pre":
                text = element.get_text()
                lines.append(f"`{text}`")
                return

            # 处理表格
            if tag_name == "table":
                table_md = self._table_to_markdown(element)
                if table_md:
                    lines.append("")
                    lines.append(table_md)
                    lines.append("")
                return

            # 处理列表
            if tag_name in ("ul", "ol"):
                lines.append("")
                for i, li in enumerate(element.find_all("li", recursive=False)):
                    prefix = f"{i + 1}. " if tag_name == "ol" else "- "
                    text = li.get_text(strip=True)
                    lines.append(f"{prefix}{text}")
                lines.append("")
                return

            # 处理段落
            if tag_name == "p":
                text = element.get_text(strip=True)
                if text:
                    lines.append("")
                    lines.append(text)
                    lines.append("")
                return

            # 处理换行
            if tag_name == "br":
                lines.append("")
                return

            # 处理链接
            if tag_name == "a" and self.keep_links:
                href = element.get("href", "")
                text = element.get_text(strip=True)
                if text and href:
                    lines.append(f"[{text}]({href})")
                    return

            # 递归处理子元素
            for child in element.children:
                process_element(child, in_code_block)

        process_element(body)

        return "\n".join(lines), headings

    def _table_to_markdown(self, table: Tag) -> str:
        """将表格转换为 Markdown 格式"""
        rows: List[List[str]] = []

        for tr in table.find_all("tr"):
            cells = []
            for cell in tr.find_all(["th", "td"]):
                text = cell.get_text(strip=True)
                # 替换管道符以避免表格格式问题
                text = text.replace("|", "\\|")
                cells.append(text)
            if cells:
                rows.append(cells)

        if not rows:
            return ""

        # 确定列数
        max_cols = max(len(row) for row in rows)

        # 规范化行
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        # 生成 Markdown 表格
        lines = []

        # 表头
        lines.append("| " + " | ".join(rows[0]) + " |")
        # 分隔线
        lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        # 数据行
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def _clean_whitespace(self, text: str) -> str:
        """清理多余空白"""
        # 合并多个空行为两个
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 移除行尾空白
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        # 移除首尾空白
        text = text.strip()
        return text
