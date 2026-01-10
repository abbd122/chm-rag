"""CHM 解包模块 - 抽象基类"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class BaseUnpacker(ABC):
    """解包器抽象基类"""

    @abstractmethod
    def unpack(self, input_path: Path, output_dir: Path) -> List[Path]:
        """
        解包文件到指定目录

        Args:
            input_path: 输入文件路径
            output_dir: 输出目录

        Returns:
            解包后的文件路径列表

        Raises:
            FileNotFoundError: 输入文件不存在
            RuntimeError: 解包失败
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查解包工具是否可用"""
        pass
