"""Qt UI 文件解析器"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class UIWidget:
    """Qt UI Widget"""

    object_name: str  # objectName
    class_name: str  # Qt 类名 (QWidget, QPushButton, etc.)
    children: List["UIWidget"] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)
    row: Optional[int] = None  # Grid 布局行
    column: Optional[int] = None  # Grid 布局列


@dataclass
class UIConnection:
    """信号槽连接"""

    sender: str
    signal: str
    receiver: str
    slot: str


@dataclass
class UIFile:
    """UI 文件解析结果"""

    root_widget: UIWidget
    connections: List[UIConnection]
    class_name: str  # 生成的类名
    version: str  # UI 文件版本


class UIParser:
    """Qt Designer UI 文件解析器"""

    def parse_file(self, file_path: Path) -> Optional[UIFile]:
        """
        解析 .ui 文件

        Args:
            file_path: UI 文件路径

        Returns:
            解析结果，失败返回 None
        """
        file_path = Path(file_path)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except ET.ParseError:
            return None

        # 获取版本
        version = root.get("version", "unknown")

        # 获取类名
        class_elem = root.find("class")
        class_name = class_elem.text if class_elem is not None else ""

        # 解析根 widget
        widget_elem = root.find("widget")
        if widget_elem is None:
            return None

        root_widget = self._parse_widget(widget_elem)

        # 解析连接
        connections = self._parse_connections(root)

        return UIFile(
            root_widget=root_widget,
            connections=connections,
            class_name=class_name,
            version=version,
        )

    def _parse_widget(self, elem: ET.Element) -> UIWidget:
        """递归解析 widget 元素"""
        class_name = elem.get("class", "")
        object_name = elem.get("name", "")

        # 解析属性
        properties = {}
        for prop in elem.findall("property"):
            prop_name = prop.get("name", "")
            if prop_name:
                # 获取属性值（可能是多种类型）
                for child in prop:
                    if child.text:
                        properties[prop_name] = child.text
                        break

        # 解析子 widget
        children: List[UIWidget] = []

        # 直接子 widget
        for child_elem in elem.findall("widget"):
            child = self._parse_widget(child_elem)
            children.append(child)

        # 布局中的 widget
        for layout in elem.findall(".//layout"):
            for item in layout.findall("item"):
                row = item.get("row")
                column = item.get("column")
                widget_elem = item.find("widget")
                if widget_elem is not None:
                    child = self._parse_widget(widget_elem)
                    if row is not None:
                        child.row = int(row)
                    if column is not None:
                        child.column = int(column)
                    children.append(child)

        return UIWidget(
            object_name=object_name,
            class_name=class_name,
            children=children,
            properties=properties,
        )

    def _parse_connections(self, root: ET.Element) -> List[UIConnection]:
        """解析信号槽连接"""
        connections = []

        for conn in root.findall(".//connection"):
            sender = self._get_text(conn, "sender")
            signal = self._get_text(conn, "signal")
            receiver = self._get_text(conn, "receiver")
            slot = self._get_text(conn, "slot")

            if sender and signal and receiver and slot:
                connections.append(
                    UIConnection(
                        sender=sender,
                        signal=signal,
                        receiver=receiver,
                        slot=slot,
                    )
                )

        return connections

    def _get_text(self, elem: ET.Element, tag: str) -> str:
        """获取子元素文本"""
        child = elem.find(tag)
        return child.text if child is not None and child.text else ""

    def widget_to_text(self, widget: UIWidget, indent: int = 0) -> str:
        """将 widget 树转换为文本描述"""
        lines = []
        prefix = "  " * indent

        # Widget 基本信息
        line = f"{prefix}- {widget.class_name}"
        if widget.object_name:
            line += f" (objectName: {widget.object_name})"
        lines.append(line)

        # 重要属性
        important_props = ["text", "title", "windowTitle", "placeholderText"]
        for prop in important_props:
            if prop in widget.properties:
                lines.append(f"{prefix}  {prop}: {widget.properties[prop]}")

        # 递归处理子控件
        for child in widget.children:
            lines.append(self.widget_to_text(child, indent + 1))

        return "\n".join(lines)

    def get_flat_widgets(self, widget: UIWidget) -> List[UIWidget]:
        """获取扁平化的 widget 列表"""
        result = [widget]
        for child in widget.children:
            result.extend(self.get_flat_widgets(child))
        return result
