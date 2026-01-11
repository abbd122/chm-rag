#!/usr/bin/env python3
"""CHM RAG 知识问答系统 - 命令行工具"""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# 将 src 添加到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.rag import RAGEngine, ChatMessage
from src.config import get_config

app = typer.Typer(
    name="chm-rag",
    help="CHM 文档 RAG 知识问答系统",
    add_completion=False,
)
console = Console()


def read_multiline_input() -> str:
    """
    读取多行输入，单独一行输入 '>>>' 结束

    Returns:
        用户输入的多行文本
    """
    console.print("[dim]（多行输入模式：输入内容，单独一行 '>>>' 结束）[/dim]")
    lines = []

    while True:
        try:
            line = console.input("[dim]...[/dim] ")
            if line.strip() == ">>>":
                break
            lines.append(line)
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print("\n[dim]已取消输入[/dim]")
            return ""

    return "\n".join(lines)


# ==================== 配置获取函数 ====================

def get_chm_file_from_config() -> Optional[Path]:
    """从配置文件获取 CHM 文件路径"""
    config = get_config()
    chm_path = config.get("chm.file_path", "")
    if chm_path:
        return Path(chm_path)
    return None


def get_output_dir_from_config() -> Path:
    """从配置文件获取解包输出目录"""
    config = get_config()
    return Path(config.get("chm.output_dir", "./data/chm_extracted"))


def get_index_path_from_config() -> Path:
    """从配置文件获取索引路径"""
    config = get_config()
    return Path(config.get("vectorstore.index_path", "./data/index"))


def get_top_k_from_config() -> int:
    """从配置文件获取 top_k 参数"""
    config = get_config()
    return config.get("rag.top_k", 5)


def get_chunks_path_from_config() -> Path:
    """从配置文件获取 chunks 缓存路径"""
    config = get_config()
    return Path(config.get("paths.chunks_cache", "./data/chunks.json"))


def get_demo_chunks_path_from_config() -> Path:
    """从配置文件获取 Demo chunks 缓存路径"""
    config = get_config()
    return Path(config.get("demo.chunks_path", "./data/demo_chunks.json"))


# ==================== CHM 命令 ====================

@app.command()
def extract(
    chm_file: Optional[Path] = typer.Argument(
        None, help="CHM 文件路径（可选，默认从 config.yaml 读取）"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="解包输出目录"
    ),
    chunks_path: Optional[Path] = typer.Option(
        None, "--chunks", "-c", help="chunks 保存路径"
    ),
):
    """
    提取 CHM 文档片段（不调用 API）

    解包 CHM → 清洗 HTML → 语义切块 → 保存 JSON
    """
    if chm_file is None:
        chm_file = get_chm_file_from_config()
        if chm_file is None:
            console.print(
                "[red]错误: 未指定 CHM 文件路径[/red]\n"
                "请通过命令行参数指定，或在 config.yaml 中配置 chm.file_path"
            )
            raise typer.Exit(1)

    if output_dir is None:
        output_dir = get_output_dir_from_config()

    if chunks_path is None:
        chunks_path = get_chunks_path_from_config()

    if not chm_file.exists():
        console.print(f"[red]错误: 文件不存在: {chm_file}[/red]")
        raise typer.Exit(1)

    if not chm_file.suffix.lower() == ".chm":
        console.print(f"[red]错误: 不是 CHM 文件: {chm_file}[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"提取文档片段: [bold]{chm_file}[/bold]", title="CHM RAG"))

    try:
        engine = RAGEngine()
        chunks = engine.extract_chunks(
            chm_file, output_dir=output_dir, chunks_path=chunks_path
        )

        console.print()
        console.print(
            Panel(
                f"[green]提取完成![/green]\n\n"
                f"文档片段数: {len(chunks)}\n"
                f"保存位置: {chunks_path}\n\n"
                f"[dim]下一步: 运行 [bold]python cli.py vectorize[/bold] 进行向量化[/dim]",
                title="完成",
            )
        )

    except Exception as e:
        console.print(f"[red]提取失败: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def vectorize(
    chunks_path: Optional[Path] = typer.Option(
        None, "--chunks", "-c", help="chunks JSON 文件路径"
    ),
    index_path: Optional[Path] = typer.Option(
        None, "--index", "-i", help="索引保存路径"
    ),
):
    """
    向量化 CHM chunks（调用 Embedding API）

    加载 chunks JSON → 向量化 → 存储到 FAISS
    """
    if chunks_path is None:
        chunks_path = get_chunks_path_from_config()

    if index_path is None:
        index_path = get_index_path_from_config()

    if not chunks_path.exists():
        console.print(
            f"[red]错误: chunks 文件不存在: {chunks_path}[/red]\n"
            "请先运行 [bold]python cli.py extract[/bold] 提取文档片段"
        )
        raise typer.Exit(1)

    console.print(Panel(f"向量化: [bold]{chunks_path}[/bold]", title="CHM RAG"))

    try:
        engine = RAGEngine()
        chunk_count = engine.vectorize_chunks(
            chunks_path=chunks_path, index_path=index_path
        )

        console.print()
        console.print(
            Panel(
                f"[green]向量化完成![/green]\n\n"
                f"文档片段数: {chunk_count}\n"
                f"索引位置: {index_path}\n\n"
                f"[dim]现在可以运行 [bold]python cli.py query \"问题\"[/bold] 进行问答[/dim]",
                title="完成",
            )
        )

    except Exception as e:
        console.print(f"[red]向量化失败: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def index(
    chm_file: Optional[Path] = typer.Argument(
        None, help="CHM 文件路径（可选，默认从 config.yaml 读取）"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="解包输出目录"
    ),
    index_path: Optional[Path] = typer.Option(
        None, "--index", "-i", help="索引保存路径"
    ),
):
    """
    一键索引 CHM（extract + vectorize）

    解包 CHM → 清洗 HTML → 语义切块 → 向量化 → 存储
    """
    if chm_file is None:
        chm_file = get_chm_file_from_config()
        if chm_file is None:
            console.print(
                "[red]错误: 未指定 CHM 文件路径[/red]\n"
                "请通过命令行参数指定，或在 config.yaml 中配置 chm.file_path"
            )
            raise typer.Exit(1)

    if output_dir is None:
        output_dir = get_output_dir_from_config()

    if index_path is None:
        index_path = get_index_path_from_config()

    if not chm_file.exists():
        console.print(f"[red]错误: 文件不存在: {chm_file}[/red]")
        raise typer.Exit(1)

    if not chm_file.suffix.lower() == ".chm":
        console.print(f"[red]错误: 不是 CHM 文件: {chm_file}[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"开始索引: [bold]{chm_file}[/bold]", title="CHM RAG"))

    try:
        engine = RAGEngine()
        chunk_count = engine.index_chm(chm_file, output_dir=output_dir)

        # 保存索引
        engine.save_index(index_path)

        console.print()
        console.print(
            Panel(
                f"[green]索引完成![/green]\n\n"
                f"文档片段数: {chunk_count}\n"
                f"向量数量: {engine.vectorstore.count()}",
                title="完成",
            )
        )

    except Exception as e:
        console.print(f"[red]索引失败: {e}[/red]")
        raise typer.Exit(1)


# ==================== Demo 命令 ====================

@app.command("demo-extract")
def demo_extract(
    demo_dir: Path = typer.Argument(..., help="Demo 工程目录"),
    demo_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="工程名称（默认使用目录名）"
    ),
    chunks_path: Optional[Path] = typer.Option(
        None, "--chunks", "-c", help="chunks 保存路径"
    ),
):
    """
    提取 Demo 工程片段（不调用 API）

    扫描文件 → 解析代码 → 生成切块 → 保存 JSON

    支持文件类型: .h, .hpp, .cpp, .cc, .cxx, .ui, .def, .tcmd
    """
    if not demo_dir.exists():
        console.print(f"[red]错误: 目录不存在: {demo_dir}[/red]")
        raise typer.Exit(1)

    if not demo_dir.is_dir():
        console.print(f"[red]错误: 不是目录: {demo_dir}[/red]")
        raise typer.Exit(1)

    if chunks_path is None:
        chunks_path = get_demo_chunks_path_from_config()

    console.print(Panel(f"提取 Demo 片段: [bold]{demo_dir}[/bold]", title="Demo RAG"))

    try:
        engine = RAGEngine()
        chunks = engine.extract_demo_chunks(
            demo_dir, demo_name=demo_name, chunks_path=chunks_path
        )

        console.print()
        console.print(
            Panel(
                f"[green]提取完成![/green]\n\n"
                f"文档片段数: {len(chunks)}\n"
                f"保存位置: {chunks_path}\n\n"
                f"[dim]下一步: 运行 [bold]python cli.py demo-vectorize[/bold] 进行向量化[/dim]",
                title="完成",
            )
        )

    except Exception as e:
        console.print(f"[red]提取失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("demo-vectorize")
def demo_vectorize(
    chunks_path: Optional[Path] = typer.Option(
        None, "--chunks", "-c", help="Demo chunks JSON 文件路径"
    ),
    index_path: Optional[Path] = typer.Option(
        None, "--index", "-i", help="索引保存路径"
    ),
):
    """
    向量化 Demo chunks（调用 Embedding API）

    加载 Demo chunks JSON → 向量化 → 添加到 FAISS 索引
    """
    if chunks_path is None:
        chunks_path = get_demo_chunks_path_from_config()

    if index_path is None:
        index_path = get_index_path_from_config()

    if not chunks_path.exists():
        console.print(
            f"[red]错误: Demo chunks 文件不存在: {chunks_path}[/red]\n"
            "请先运行 [bold]python cli.py demo-extract <DIR>[/bold] 提取片段"
        )
        raise typer.Exit(1)

    console.print(Panel(f"向量化 Demo: [bold]{chunks_path}[/bold]", title="Demo RAG"))

    try:
        engine = RAGEngine()

        # 尝试加载现有索引
        try:
            engine.load_index(index_path)
            console.print(f"[dim]已加载现有索引: {engine.vectorstore.count()} 个向量[/dim]")
        except FileNotFoundError:
            console.print("[dim]创建新索引[/dim]")

        chunk_count = engine.vectorize_demo_chunks(
            chunks_path=chunks_path, index_path=index_path
        )

        console.print()
        console.print(
            Panel(
                f"[green]向量化完成![/green]\n\n"
                f"新增片段数: {chunk_count}\n"
                f"索引总量: {engine.vectorstore.count()}\n"
                f"索引位置: {index_path}",
                title="完成",
            )
        )

    except Exception as e:
        console.print(f"[red]向量化失败: {e}[/red]")
        raise typer.Exit(1)


@app.command("demo-index")
def demo_index(
    demo_dir: Path = typer.Argument(..., help="Demo 工程目录"),
    demo_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="工程名称"
    ),
    index_path: Optional[Path] = typer.Option(
        None, "--index", "-i", help="索引保存路径"
    ),
):
    """
    一键索引 Demo 工程

    扫描 → 解析 → 切块 → 向量化 → 存储
    """
    if not demo_dir.exists():
        console.print(f"[red]错误: 目录不存在: {demo_dir}[/red]")
        raise typer.Exit(1)

    if index_path is None:
        index_path = get_index_path_from_config()

    console.print(Panel(f"索引 Demo: [bold]{demo_dir}[/bold]", title="Demo RAG"))

    try:
        engine = RAGEngine()

        # 尝试加载现有索引
        try:
            engine.load_index(index_path)
            console.print(f"[dim]已加载现有索引: {engine.vectorstore.count()} 个向量[/dim]")
        except FileNotFoundError:
            console.print("[dim]创建新索引[/dim]")

        chunk_count = engine.index_demo(demo_dir, demo_name=demo_name)

        # 保存索引
        engine.save_index(index_path)

        console.print()
        console.print(
            Panel(
                f"[green]索引完成![/green]\n\n"
                f"Demo 片段数: {chunk_count}\n"
                f"索引总量: {engine.vectorstore.count()}",
                title="完成",
            )
        )

    except Exception as e:
        console.print(f"[red]索引失败: {e}[/red]")
        raise typer.Exit(1)


# ==================== 查询命令 ====================

@app.command()
def query(
    question: str = typer.Argument(..., help="问题"),
    index_path: Optional[Path] = typer.Option(
        None, "--index", "-i", help="索引路径"
    ),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", "-k", help="检索文档数量"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="数据源过滤: chm, demo, 或留空查询全部"
    ),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="显示来源"),
):
    """
    查询问答

    基于已索引的 CHM 文档和 Demo 代码回答问题
    """
    if index_path is None:
        index_path = get_index_path_from_config()

    if top_k is None:
        top_k = get_top_k_from_config()

    # 验证 source 参数
    if source and source not in ("chm", "demo"):
        console.print(f"[red]错误: 无效的 source 参数: {source}[/red]")
        console.print("有效值: chm, demo, 或留空查询全部")
        raise typer.Exit(1)

    try:
        engine = RAGEngine(top_k=top_k)
        engine.load_index(index_path)

    except FileNotFoundError:
        console.print("[red]错误: 索引不存在，请先运行 index 或 demo-index 命令[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]加载索引失败: {e}[/red]")
        raise typer.Exit(1)

    title = f"查询 [{source or '全部'}]" if source else "查询"
    console.print(Panel(f"问题: [bold]{question}[/bold]", title=title))

    try:
        response = engine.query(
            question, top_k=top_k, source=source, return_sources=show_sources
        )

        # 显示回答
        console.print()
        console.print("[bold green]回答:[/bold green]")
        console.print(Markdown(response.answer))

        # 显示来源
        if show_sources and response.sources:
            console.print()
            table = Table(title="来源文档片段")
            table.add_column("排名", style="cyan", width=4)
            table.add_column("来源", style="magenta", width=6)
            table.add_column("分数", style="green", width=8)
            table.add_column("符号", style="yellow", width=20)
            table.add_column("文件", style="dim", width=35)

            for result in response.sources:
                metadata = result.chunk.metadata
                source_type = metadata.get("source_type", "chm")
                source_label = "CHM" if source_type == "chm" else "Demo"

                file_info = metadata.get("file_path") or metadata.get("chapter") or metadata.get("source_file", "-")
                if len(file_info) > 35:
                    file_info = "..." + file_info[-32:]

                table.add_row(
                    str(result.rank),
                    source_label,
                    f"{result.score:.4f}",
                    metadata.get("symbol", "-")[:20],
                    file_info,
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]查询失败: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    index_path: Optional[Path] = typer.Option(
        None, "--index", "-i", help="索引路径"
    ),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", "-k", help="检索文档数量"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="数据源过滤: chm, demo, 或留空查询全部"
    ),
):
    """
    交互式问答

    进入交互模式，持续回答问题
    """
    if index_path is None:
        index_path = get_index_path_from_config()

    if top_k is None:
        top_k = get_top_k_from_config()

    # 验证 source 参数
    if source and source not in ("chm", "demo"):
        console.print(f"[red]错误: 无效的 source 参数: {source}[/red]")
        console.print("有效值: chm, demo, 或留空查询全部")
        raise typer.Exit(1)

    try:
        engine = RAGEngine(top_k=top_k)
        engine.load_index(index_path)

    except FileNotFoundError:
        console.print("[red]错误: 索引不存在，请先运行 index 命令[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]加载索引失败: {e}[/red]")
        raise typer.Exit(1)

    stats = engine.get_stats()
    source_label = f" [{source}]" if source else ""
    console.print(
        Panel(
            f"进入交互式问答模式{source_label}\n"
            f"索引: CHM {stats.get('chm_chunks', 0)} + Demo {stats.get('demo_chunks', 0)} 片段\n\n"
            f"[bold]操作说明:[/bold]\n"
            f"  • 输入问题后按回车\n"
            f"  • 输入 [cyan]<<<[/cyan] 进入多行输入模式（输入 [cyan]>>>[/cyan] 结束）\n"
            f"  • 输入 [cyan]exit[/cyan]/[cyan]quit[/cyan] 退出\n"
            f"  • 输入 [cyan]clear[/cyan] 清空上下文",
            title="CHM RAG 交互模式",
        )
    )

    # 对话历史
    history: List[ChatMessage] = []

    while True:
        try:
            console.print()
            question = console.input("[bold cyan]问题:[/bold cyan] ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit", "q"):
                console.print("[dim]再见![/dim]")
                break

            if question.lower() == "clear":
                history.clear()
                console.print("[dim]已清空对话上下文[/dim]")
                continue

            # === 多行模式 ===
            if question == "<<<":
                question = read_multiline_input().strip()
                if not question:
                    continue

            response = engine.query(
                question, top_k=top_k, source=source,
                return_sources=False, history=history
            )
            console.print()
            console.print("[bold green]回答:[/bold green]")
            console.print(Markdown(response.answer))

            # 保存到对话历史
            history.append(ChatMessage(role="user", content=question))
            history.append(ChatMessage(role="assistant", content=response.answer))

        except KeyboardInterrupt:
            console.print("\n[dim]再见![/dim]")
            break
        except Exception as e:
            console.print(f"[red]查询失败: {e}[/red]")


@app.command()
def inspect(
    index_path: Optional[Path] = typer.Option(
        None, "--index", "-i", help="索引路径"
    ),
):
    """
    查看索引状态

    显示索引的统计信息
    """
    if index_path is None:
        index_path = get_index_path_from_config()

    try:
        engine = RAGEngine()
        engine.load_index(index_path)

    except FileNotFoundError:
        console.print("[red]错误: 索引不存在[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]加载索引失败: {e}[/red]")
        raise typer.Exit(1)

    stats = engine.get_stats()
    config = get_config()

    table = Table(title="索引信息")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="green")

    table.add_row("文档片段总数", str(stats["total_chunks"]))
    table.add_row("  - CHM 片段", str(stats.get("chm_chunks", "-")))
    table.add_row("  - Demo 片段", str(stats.get("demo_chunks", "-")))
    table.add_row("Embedding 模型", stats["embedding_model"])
    table.add_row("向量维度", str(stats["embedding_dimension"]))
    table.add_row("LLM 模型", stats["llm_model"])
    table.add_row("索引路径", str(index_path))
    table.add_row("CHM 文件", config.get("chm.file_path", "-"))

    console.print(table)


@app.command()
def pack(
    output: Path = typer.Argument(
        ..., help="输出目录路径"
    ),
    index_path: Optional[Path] = typer.Option(
        None, "--index", "-i", help="索引路径"
    ),
):
    """
    打包 chat 模式所需的最小文件集

    打包后可复制到其他 PC 直接使用 chat 模式，无需原始 CHM/Demo 文件
    """
    import shutil

    if index_path is None:
        index_path = get_index_path_from_config()

    # 验证索引存在
    if not index_path.exists():
        console.print(f"[red]错误: 索引不存在: {index_path}[/red]")
        raise typer.Exit(1)

    # 创建输出目录
    output = Path(output)
    if output.exists():
        console.print(f"[yellow]警告: 输出目录已存在，将覆盖: {output}[/yellow]")
        shutil.rmtree(output)

    output.mkdir(parents=True)

    console.print(Panel(f"打包到: [bold]{output}[/bold]", title="打包 Chat 模式"))

    try:
        # 复制必需文件
        base_dir = Path(__file__).parent

        # 1. 复制 cli.py
        shutil.copy2(base_dir / "cli.py", output / "cli.py")
        console.print("  [green]✓[/green] cli.py")

        # 2. 复制 config.yaml（清理敏感信息）
        config_src = base_dir / "config.yaml"
        config_dst = output / "config.yaml"
        if config_src.exists():
            content = config_src.read_text(encoding="utf-8")
            # 替换 API key 为占位符
            import re
            content = re.sub(
                r'(api_key:\s*["\']?)sk-[^"\'"\n]+(["\']?)',
                r'\1YOUR_API_KEY_HERE\2',
                content
            )
            config_dst.write_text(content, encoding="utf-8")
            console.print("  [green]✓[/green] config.yaml (API key 已清除)")
        else:
            console.print("  [yellow]![/yellow] config.yaml 不存在，跳过")

        # 3. 复制 requirements.txt
        req_src = base_dir / "requirements.txt"
        if req_src.exists():
            shutil.copy2(req_src, output / "requirements.txt")
            console.print("  [green]✓[/green] requirements.txt")

        # 4. 复制 src 目录
        src_dir = base_dir / "src"
        if src_dir.exists():
            shutil.copytree(src_dir, output / "src")
            console.print("  [green]✓[/green] src/")

        # 5. 复制索引目录
        index_dst = output / "data" / "index"
        index_dst.parent.mkdir(parents=True)
        shutil.copytree(index_path, index_dst)
        console.print("  [green]✓[/green] data/index/")

        # 计算总大小
        total_size = sum(
            f.stat().st_size for f in output.rglob("*") if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)

        console.print()
        console.print(
            Panel(
                f"[green]打包完成![/green]\n\n"
                f"输出目录: {output}\n"
                f"总大小: {size_mb:.1f} MB\n\n"
                f"[dim]使用说明:[/dim]\n"
                f"1. 复制整个目录到目标 PC\n"
                f"2. pip install -r requirements.txt\n"
                f"3. 编辑 config.yaml 填入 API key\n"
                f"4. python cli.py chat",
                title="完成",
            )
        )

    except Exception as e:
        console.print(f"[red]打包失败: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """显示版本信息"""
    console.print("CHM RAG 知识问答系统 v0.2.0")
    console.print("  - 支持 CHM 文档索引")
    console.print("  - 支持 C++/Qt Demo 工程索引")


def main():
    """主入口"""
    app()


if __name__ == "__main__":
    main()
