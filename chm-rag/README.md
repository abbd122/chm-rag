# CHM + Demo RAG 知识问答系统

将 CHM 格式开发文档和 C++/Qt Demo 工程转换为 RAG 知识库，基于文档内容回答开发问题。

## 特性

- **双数据源支持**：同时索引 CHM 官方文档和 Demo 示例代码
- **智能混合检索**：CHM 提供规范说明，Demo 提供真实用例
- **C++ 代码解析**：提取函数、类、API 调用等符号
- **Qt UI 解析**：解析 .ui 文件的控件结构和信号槽
- **语义切块**：基于内容结构智能分割文档

## 安装

```bash
cd chm-rag
pip install -r requirements.txt
```

## 配置

编辑 `config.yaml` 进行配置：

```yaml
# OpenAI API 配置
openai:
  api_key: "your-api-key-here"    # 必填：OpenAI API Key
  base_url: "https://api.openai.com"  # 可选：API 代理地址

# CHM 文档配置
chm:
  file_path: "./docs/api.chm"     # CHM 文件路径
  output_dir: ./data/chm_extracted # 解包输出目录

# 工具配置
tools:
  7z_path: ""                     # 可选：7z.exe 完整路径（自动检测失败时手动指定）

# Embedding 配置
embedding:
  provider: openai
  model: text-embedding-3-large
  batch_size: 100

# 向量存储配置
vectorstore:
  provider: faiss
  index_path: ./data/index        # 索引保存路径

# LLM 配置
llm:
  provider: openai
  model: gpt-4o                   # 使用 gpt-4o 模型
  temperature: 0.1
  max_tokens: 2000

# RAG 配置
rag:
  top_k: 10                       # 检索文档数量
  max_chunk_tokens: 800           # 单个切块最大 token 数

# Demo 工程配置
demo:
  directories:
    - path: "D:/Projects/Demo1"   # Demo 工程路径
      name: "Demo1"               # 工程名（可选）
  parser:
    context_lines: 5              # API 调用上下文行数
    min_commands: 1               # TCMD 最少命令数
  chunks_path: ./data/demo_chunks.json
```

### 配置优先级

所有参数支持两种方式配置：
1. **配置文件** (`config.yaml`) - 推荐，一次配置多次使用
2. **命令行参数** - 临时覆盖配置文件

命令行参数优先级高于配置文件。

## 使用方法

### 工作流程概览

系统支持两类数据源，各自独立处理后合并索引：

| 数据源 | 提取命令 | 向量化命令 | 一键命令 |
|--------|----------|------------|----------|
| **CHM 文档** | `extract` | `vectorize` | `index` |
| **Demo 工程** | `demo-extract` | `demo-vectorize` | `demo-index` |

**推荐分步执行**：先运行 `extract` 检查切块结果，确认无误后再运行 `vectorize`。

---

## CHM 文档处理

### 1. 分步执行（推荐）

#### 步骤一：提取文档片段

```bash
python cli.py extract
```

此命令执行：解包 CHM → 清洗 HTML → 语义切块 → 保存 chunks.json

**特点：不调用任何 API，可反复调试**

**可选参数：**
- `CHM_PATH`: CHM 文件路径（覆盖配置文件）
- `-o, --output`: 解包输出目录
- `-c, --chunks`: chunks.json 保存路径（默认：`./data/chunks.json`）

#### 步骤二：向量化存储

```bash
python cli.py vectorize
```

此命令执行：加载 chunks.json → 调用 Embedding API 向量化 → 存储到 FAISS

**可选参数：**
- `-c, --chunks`: chunks.json 路径（默认：`./data/chunks.json`）
- `-i, --index`: 索引保存路径

### 2. 一键执行

```bash
python cli.py index
```

等同于 `extract` + `vectorize`，适合配置确认无误后使用。

---

## Demo 工程处理

### 支持的文件类型

| 类型 | 扩展名 | 处理方式 |
|------|--------|----------|
| C++ 源码 | .h, .hpp, .cpp, .cc, .cxx | 函数/类/API 调用提取 |
| Qt UI | .ui | 控件结构和信号槽解析 |
| DEF 导出 | .def | 导出符号提取 |
| 命令脚本 | .tcmd | 有效命令提取 |

**排除的文件**：.vcxproj, .filters, .sln, 编译产物等

### 1. 分步执行（推荐）

#### 步骤一：提取 Demo 片段

```bash
python cli.py demo-extract DEMO_DIR
```

此命令执行：扫描目录 → 解析代码 → 提取符号 → 保存 demo_chunks.json

**参数：**
- `DEMO_DIR`: Demo 工程目录路径
- `-n, --name`: 工程名（默认使用目录名）
- `-c, --chunks`: 输出文件路径（默认：`./data/demo_chunks.json`）

**示例：**
```bash
python cli.py demo-extract "D:/Projects/MyDemo" -n "MyProject"
```

#### 步骤二：向量化存储

```bash
python cli.py demo-vectorize
```

此命令执行：加载 demo_chunks.json → 向量化 → 合并到索引

**可选参数：**
- `-c, --chunks`: demo_chunks.json 路径
- `-i, --index`: 索引保存路径

### 2. 一键执行

```bash
python cli.py demo-index DEMO_DIR
```

等同于 `demo-extract` + `demo-vectorize`。

---

## 查询问答

### 单次查询

```bash
python cli.py query "某接口的参数含义是什么？"
```

**可选参数：**
- `-k, --top-k`: 检索文档数量
- `-i, --index`: 索引路径
- `-s, --source`: 数据源过滤（`chm` | `demo` | `all`，默认 `all`）
- `--no-sources`: 不显示来源文档

**示例：**
```bash
# 仅从 CHM 文档检索
python cli.py query "CreatePart 函数参数" -s chm

# 仅从 Demo 工程检索
python cli.py query "如何使用 CreatePart" -s demo

# 混合检索（默认）
python cli.py query "零件创建的完整流程"
```

### 交互式问答

```bash
python cli.py chat
```

进入交互模式后，直接输入问题即可。输入 `exit` 或 `quit` 退出。

### 查看索引状态

```bash
python cli.py inspect
```

显示当前索引的统计信息，包括 CHM 和 Demo 片段数、模型信息等。

---

## 命令速查表

| 命令 | 功能 | API 调用 |
|------|------|----------|
| `extract` | CHM: 解包 → 切块 → 保存 JSON | 无 |
| `vectorize` | CHM: 加载 JSON → 向量化 → 存储 | Embedding |
| `index` | CHM: 一键完成上述步骤 | Embedding |
| `demo-extract` | Demo: 解析代码 → 提取符号 → 保存 JSON | 无 |
| `demo-vectorize` | Demo: 加载 JSON → 向量化 → 合并索引 | Embedding |
| `demo-index` | Demo: 一键完成上述步骤 | Embedding |
| `query` | 单次问答（支持 --source 过滤） | Embedding + LLM |
| `chat` | 交互式问答 | Embedding + LLM |
| `inspect` | 查看索引状态 | 无 |

---

## 项目结构

```
chm-rag/
├── src/
│   ├── config.py           # 配置管理
│   ├── unpacker/           # CHM 解包
│   ├── cleaner/            # HTML 清洗
│   ├── chunker/            # 语义切块
│   ├── embedding/          # 向量化
│   ├── vectorstore/        # 向量存储 (FAISS)
│   ├── llm/                # LLM 接口
│   ├── rag/                # RAG 引擎
│   └── demo/               # Demo 解析模块
│       ├── scanner.py      # 文件扫描器
│       ├── cpp_parser.py   # C++ 代码解析
│       ├── ui_parser.py    # Qt UI 解析
│       ├── def_parser.py   # DEF 文件解析
│       ├── tcmd_parser.py  # TCMD 解析
│       └── demo_chunker.py # Demo 切块器
├── data/
│   ├── chm_extracted/      # CHM 解包输出
│   ├── chunks.json         # CHM 切块结果
│   ├── demo_chunks.json    # Demo 切块结果
│   └── index/              # FAISS 索引
├── cli.py                  # 命令行工具
├── config.yaml             # 配置文件
└── requirements.txt        # 依赖
```

---

## 快速开始示例

### CHM 文档索引

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 编辑配置文件，设置 api_key 和 chm.file_path

# 3. 提取文档片段
python cli.py extract

# 4. 向量化存储
python cli.py vectorize

# 5. 开始问答
python cli.py query "CreatePart 函数的参数含义"
```

### Demo 工程索引

```bash
# 1. 提取 Demo 代码
python cli.py demo-extract "D:/Projects/MyDemo" -n "MyProject"

# 2. 向量化并合并到索引
python cli.py demo-vectorize

# 3. 混合问答
python cli.py query "CreatePart 的使用示例"

# 4. 查看索引状态
python cli.py inspect
```

### 完整索引（CHM + Demo）

```bash
# CHM
python cli.py index

# Demo
python cli.py demo-index "D:/Projects/Demo1"
python cli.py demo-index "D:/Projects/Demo2"

# 问答
python cli.py chat
```

---

## 系统要求

- Python 3.8+
- CHM 解包工具（任一）：
  - 7-Zip (`7z`) - 可在 `config.yaml` 的 `tools.7z_path` 指定路径
  - Windows: `hh.exe`（系统自带）
  - Linux: `extract_chmLib`

## 常见问题

### 7z 检测不到

Windows 系统如果 `shutil.which("7z")` 检测失败，可在 `config.yaml` 中手动指定：

```yaml
tools:
  7z_path: "D:/7-Zip/7z.exe"
```

### 使用 API 代理

如需使用 OpenAI API 代理服务，配置 `base_url`：

```yaml
openai:
  api_key: "your-api-key"
  base_url: "https://your-proxy-url.com"
```

### CHM 和 Demo 答案冲突

系统默认以 CHM 官方文档为准，Demo 作为示例参考。查询时会标注来源：
- 【CHM】：来自官方 API 文档
- 【Demo】：来自示例工程代码

### 仅查询特定数据源

使用 `--source` 参数过滤：

```bash
python cli.py query "问题" -s chm    # 仅 CHM
python cli.py query "问题" -s demo   # 仅 Demo
python cli.py query "问题" -s all    # 混合（默认）
```

### 迁移步骤
```bash
# 1. 复制必需文件
chm-rag/
├── cli.py
├── config.yaml
├── requirements.txt
├── src/
└── data/index/
# 2. 安装依赖
pip install -r requirements.txt
# 3. 修改 config.yaml 中的 API key
# 4. 直接使用
python cli.py chat

总结：索引一次，到处使用。data/index/ 已包含所有向量化数据，不再依赖源文件
```