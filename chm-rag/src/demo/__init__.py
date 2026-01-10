"""Demo 工程解析模块

支持解析 C++/Qt Demo 工程，提取代码示例作为 RAG 知识源。
"""

from .scanner import DemoScanner, ScannedFile
from .cpp_parser import CppParser, CppSymbol, ApiCall
from .ui_parser import UIParser, UIWidget
from .def_parser import DefParser
from .tcmd_parser import TcmdParser
from .demo_chunker import DemoChunker

__all__ = [
    "DemoScanner",
    "ScannedFile",
    "CppParser",
    "CppSymbol",
    "ApiCall",
    "UIParser",
    "UIWidget",
    "DefParser",
    "TcmdParser",
    "DemoChunker",
]
