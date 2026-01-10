"""CHM 解包模块 - 跨平台实现"""

import platform
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


class CHMUnpacker:
    """CHM 文件解包器，支持 Windows 和 Linux"""

    # Windows 上 7-Zip 的常见安装路径
    WINDOWS_7Z_PATHS = [
        "C:/Program Files/7-Zip/7z.exe",
        "C:/Program Files (x86)/7-Zip/7z.exe",
        "D:/Program Files/7-Zip/7z.exe",
        "D:/Program Files (x86)/7-Zip/7z.exe",
        "E:/Program Files/7-Zip/7z.exe",
        "C:/7-Zip/7z.exe",
        "D:/7-Zip/7z.exe",
    ]

    def __init__(self, config_7z_path: Optional[str] = None):
        """
        初始化解包器
        
        Args:
            config_7z_path: 从配置文件读取的 7z 路径
        """
        self._tool: Optional[str] = None
        self._tool_path: Optional[str] = None
        self._config_7z_path = config_7z_path
        self._detect_tool()

    def _detect_tool(self) -> None:
        """检测可用的解包工具"""
        system = platform.system().lower()

        # 1. 优先使用配置文件中指定的 7z 路径
        if self._config_7z_path and Path(self._config_7z_path).exists():
            self._tool = "7z"
            self._tool_path = self._config_7z_path
            return

        # 2. 检查 PATH 中是否有 7z
        if shutil.which("7z"):
            self._tool = "7z"
            self._tool_path = shutil.which("7z")
            return

        # 3. Windows 上检查常见安装路径
        if system == "windows":
            for path in self.WINDOWS_7Z_PATHS:
                if Path(path).exists():
                    self._tool = "7z"
                    self._tool_path = path
                    return

            # 4. Windows 上检测 hh.exe
            hh_path = shutil.which("hh")
            if hh_path or Path("C:/Windows/hh.exe").exists():
                self._tool = "hh"
                self._tool_path = hh_path or "C:/Windows/hh.exe"
                return

        elif system == "linux":
            # Linux 上检测 extract_chmLib
            if shutil.which("extract_chmLib"):
                self._tool = "extract_chmLib"
                self._tool_path = "extract_chmLib"
                return

    def is_available(self) -> bool:
        """检查解包工具是否可用"""
        return self._tool is not None

    def get_tool_name(self) -> Optional[str]:
        """获取当前使用的工具名称"""
        return self._tool

    def get_tool_path(self) -> Optional[str]:
        """获取当前使用的工具路径"""
        return self._tool_path

    def unpack(self, input_path: Path, output_dir: Path) -> List[Path]:
        """
        解包 CHM 文件

        Args:
            input_path: CHM 文件路径
            output_dir: 输出目录

        Returns:
            解包后的 HTML 文件路径列表

        Raises:
            FileNotFoundError: CHM 文件不存在
            RuntimeError: 解包工具不可用或解包失败
        """
        input_path = Path(input_path).resolve()
        output_dir = Path(output_dir).resolve()

        # 检查输入文件
        if not input_path.exists():
            raise FileNotFoundError(f"CHM 文件不存在: {input_path}")

        if not input_path.suffix.lower() == ".chm":
            raise ValueError(f"不是 CHM 文件: {input_path}")

        # 检查工具可用性
        if not self.is_available():
            raise RuntimeError(
                "未找到可用的解包工具。请选择以下方式之一:\n"
                "1. 在 config.yaml 中设置 tools.7z_path 指定 7z.exe 路径\n"
                "2. 将 7-Zip 安装到默认路径\n"
                "3. 将 7z.exe 所在目录添加到系统 PATH\n"
                "4. Windows 系统可使用自带的 hh.exe"
            )

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 根据工具执行解包
        if self._tool == "7z":
            self._unpack_with_7z(input_path, output_dir)
        elif self._tool == "hh":
            self._unpack_with_hh(input_path, output_dir)
        elif self._tool == "extract_chmLib":
            self._unpack_with_extract_chmlib(input_path, output_dir)

        # 收集解包后的 HTML 文件
        html_files = self._collect_html_files(output_dir)

        if not html_files:
            raise RuntimeError(f"解包完成但未找到 HTML 文件: {output_dir}")

        return html_files

    def _unpack_with_7z(self, input_path: Path, output_dir: Path) -> None:
        """使用 7z 解包"""
        cmd = [self._tool_path, "x", "-y", f"-o{output_dir}", str(input_path)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"7z 解包失败: {result.stderr}")

    def _unpack_with_hh(self, input_path: Path, output_dir: Path) -> None:
        """使用 Windows hh.exe 解包"""
        cmd = [self._tool_path, "-decompile", str(output_dir), str(input_path)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=True,
        )
        # hh.exe 的返回码不可靠，检查输出目录是否有文件
        if not any(output_dir.iterdir()):
            raise RuntimeError(f"hh.exe 解包失败: {result.stderr}")

    def _unpack_with_extract_chmlib(self, input_path: Path, output_dir: Path) -> None:
        """使用 extract_chmLib 解包 (Linux)"""
        cmd = ["extract_chmLib", str(input_path), str(output_dir)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"extract_chmLib 解包失败: {result.stderr}")

    def _collect_html_files(self, directory: Path) -> List[Path]:
        """收集目录下所有 HTML 文件"""
        html_extensions = {".html", ".htm", ".xhtml"}
        html_files = []

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in html_extensions:
                html_files.append(file_path)

        html_files.sort()
        return html_files
