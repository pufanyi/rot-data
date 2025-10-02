# rot-data

数据集加载和管理工具，支持实时任务追踪。

## 功能特性

- 📦 支持多种数据集（CO3D 等）
- 🎯 实时任务追踪和可视化
- 🔄 智能缓存机制
- 🧵 多线程并行加载
- 📊 美观的 Web 界面监控

## 快速开始

### 安装依赖

```bash
uv sync
```

### 最简单的使用方式 - 带任务板！

运行数据准备并实时查看任务进度：

```bash
uv run python -m rot_data.pipeline.prepare_data --use-taskboard --no-push
```

然后在浏览器打开 `http://localhost:8765` 查看实时任务板！

详细说明请查看 [快速开始指南](QUICKSTART.md)。

### 基本使用（不带任务板）

```python
from rot_data.dataloader.co3d import CO3DDataLoader

# 创建数据加载器
loader = CO3DDataLoader(cache_dir="cache", num_threads=4)

# 加载数据
for data in loader.load():
    print(data)
```

### 使用任务板追踪进度

#### 方法一：使用 prepare_data.py（推荐）

最简单的方式，只需添加 `--use-taskboard` 参数：

```bash
uv run python -m rot_data.pipeline.prepare_data \
    --dataset-name co3d \
    --cache-dir cache \
    --num-threads 4 \
    --use-taskboard \
    --no-push
```

然后在浏览器打开 `http://localhost:8765` 查看实时任务进度！

#### 方法二：在代码中使用

```python
from rot_data.dataloader.co3d import CO3DDataLoader
from rot_data.taskboard.server import start_server

# 启动任务板服务器
server = start_server(host="0.0.0.0", port=8765)
print("打开浏览器访问: http://localhost:8765")

# 启用任务板追踪
loader = CO3DDataLoader(
    cache_dir="cache",
    num_threads=4,
    use_taskboard=True  # 启用任务板
)

# 加载数据（进度会实时显示在 Web 界面）
for data in loader.load():
    pass
```

## 示例

### 运行任务板演示

我们提供了一个模拟 441 个任务的演示脚本：

```bash
uv run python examples/taskboard_demo.py
```

然后在浏览器中打开 `http://localhost:8765` 查看实时任务状态。

### 运行完整数据加载示例

```bash
uv run python examples/taskboard_example.py
```

## 任务板功能

任务板提供了以下功能：

- ✅ **实时更新**：通过 WebSocket 实时推送任务状态
- 📊 **统计面板**：显示总任务数、待处理、运行中、已完成、失败等统计
- 🎨 **美观界面**：现代化的渐变色卡片设计
- 🔍 **任务过滤**：按状态过滤任务（全部、待处理、运行中、已完成、失败）
- 📈 **进度条**：实时显示每个任务的完成百分比
- ⚠️ **错误信息**：失败任务会显示详细错误信息

详细使用说明请参考 [任务板文档](rot_data/taskboard/README.md)。

## 项目结构

```
rot-data/
├── rot_data/
│   ├── dataloader/      # 数据加载器
│   │   ├── co3d.py      # CO3D 数据集加载器
│   │   └── data.py      # 基础抽象类
│   ├── taskboard/       # 任务板模块
│   │   ├── manager.py   # 任务管理核心
│   │   ├── server.py    # FastAPI 服务器
│   │   └── static/      # 前端静态文件
│   ├── utils/           # 工具函数
│   └── pipeline/        # 数据处理流程
├── examples/            # 示例脚本
│   ├── taskboard_demo.py    # 任务板演示
│   └── taskboard_example.py # 完整示例
└── tests/              # 测试文件

```

## 开发

### 代码检查

```bash
uv run ruff check rot_data main.py
```

### 运行测试

```bash
uv run pytest
```

## 许可

MIT License
