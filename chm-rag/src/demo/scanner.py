"""Demo 文件扫描器"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set


@dataclass
class ScannedFile:
    """扫描到的文件"""

    path: Path  # 文件绝对路径
    relative_path: str  # 相对于 demo 根目录的路径
    ext: str  # 扩展名（小写，含点号）
    demo_name: str  # 工程名


class DemoScanner:
    """Demo 工程文件扫描器"""

    # 强有效文件类型（全文解析）
    STRONG_EXTENSIONS: Set[str] = {
        ".h",
        ".hpp",
        ".cpp",
        ".cc",
        ".cxx",
        ".ui",
    }

    # 条件有效文件类型（部分解析）
    CONDITIONAL_EXTENSIONS: Set[str] = {
        ".def",
        ".tcmd",
    }

    # 明确排除的文件类型
    SKIP_EXTENSIONS: Set[str] = {
        ".vcxproj",
        ".filters",
        ".sln",
        ".suo",
        ".user",
        ".ncb",
        ".aps",
        ".pdb",
        ".ilk",
        ".obj",
        ".exe",
        ".dll",
        ".lib",
        ".exp",
        ".idb",
        ".pch",
        ".ipch",
        ".tlog",
        ".log",
        ".cache",
    }

    # 排除的目录
    SKIP_DIRS: Set[str] = {
        ".git",
        ".svn",
        ".vs",
        ".vscode",
        "Debug",
        "Release",
        "x64",
        "x86",
        "build",
        "out",
        "bin",
        "obj",
        "__pycache__",
        "node_modules",
        "GeneratedFiles",
    }

    def __init__(self, skip_dirs: Optional[Set[str]] = None):
        """
        初始化扫描器

        Args:
            skip_dirs: 额外需要跳过的目录名
        """
        self.skip_dirs = self.SKIP_DIRS.copy()
        if skip_dirs:
            self.skip_dirs.update(skip_dirs)

    def scan(self, demo_dir: Path, demo_name: Optional[str] = None) -> List[ScannedFile]:
        """
        扫描 Demo 目录

        Args:
            demo_dir: Demo 根目录
            demo_name: 工程名（默认使用目录名）

        Returns:
            扫描到的文件列表
        """
        demo_dir = Path(demo_dir).resolve()

        if not demo_dir.exists():
            raise FileNotFoundError(f"Demo 目录不存在: {demo_dir}")

        if not demo_dir.is_dir():
            raise ValueError(f"路径不是目录: {demo_dir}")

        if demo_name is None:
            demo_name = demo_dir.name

        files: List[ScannedFile] = []

        for file_path in self._walk_files(demo_dir):
            ext = file_path.suffix.lower()

            # 跳过明确排除的类型
            if ext in self.SKIP_EXTENSIONS:
                continue

            # 只处理有效类型
            if ext not in self.STRONG_EXTENSIONS and ext not in self.CONDITIONAL_EXTENSIONS:
                continue

            relative_path = str(file_path.relative_to(demo_dir))

            files.append(
                ScannedFile(
                    path=file_path,
                    relative_path=relative_path,
                    ext=ext,
                    demo_name=demo_name,
                )
            )

        return files

    def _walk_files(self, root: Path):
        """递归遍历目录"""
        for item in root.iterdir():
            if item.is_dir():
                if item.name not in self.skip_dirs:
                    yield from self._walk_files(item)
            elif item.is_file():
                yield item

    def scan_multiple(
        self, demo_dirs: List[Path], demo_names: Optional[List[str]] = None
    ) -> List[ScannedFile]:
        """
        扫描多个 Demo 目录

        Args:
            demo_dirs: Demo 目录列表
            demo_names: 对应的工程名列表

        Returns:
            所有扫描到的文件列表
        """
        if demo_names and len(demo_names) != len(demo_dirs):
            raise ValueError("demo_dirs 和 demo_names 长度不匹配")

        all_files: List[ScannedFile] = []

        for i, demo_dir in enumerate(demo_dirs):
            name = demo_names[i] if demo_names else None
            files = self.scan(demo_dir, demo_name=name)
            all_files.extend(files)

        return all_files

    def get_file_stats(self, files: List[ScannedFile]) -> dict:
        """获取文件统计信息"""
        stats = {
            "total": len(files),
            "by_extension": {},
            "by_demo": {},
        }

        for file in files:
            # 按扩展名统计
            ext = file.ext
            stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1

            # 按工程统计
            demo = file.demo_name
            stats["by_demo"][demo] = stats["by_demo"].get(demo, 0) + 1

        return stats
