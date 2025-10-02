# Task Board

任务板系统提供了一个可视化界面来实时追踪数据加载任务的状态。

## 功能特性

- 🎯 **实时追踪**：通过 WebSocket 实时更新任务状态
- 📊 **可视化界面**：美观的 Web 界面显示所有任务的进度
- 🔄 **可复用**：通用设计，可用于任何数据加载器
- 🧵 **线程安全**：支持多线程环境下的任务追踪
- 📈 **统计信息**：显示任务总数、完成数、失败数等统计

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 启动任务板服务器并运行数据加载

```python
from rot_data.dataloader.co3d import CO3DDataLoader
from rot_data.taskboard.server import start_server

# 启动任务板服务器
server = start_server(host="0.0.0.0", port=8765)
print("Task board: http://localhost:8765")

# 创建数据加载器并启用任务板
loader = CO3DDataLoader(
    cache_dir="cache",
    num_threads=4,
    use_taskboard=True,  # 启用任务板追踪
)

# 加载数据
for data in loader.load():
    # 处理数据...
    pass
```

### 3. 查看任务板

在浏览器中打开 `http://localhost:8765` 查看实时的任务状态。

## 运行示例

我们提供了一个完整的示例脚本：

```bash
uv run python examples/taskboard_example.py
```

然后在浏览器中打开 `http://localhost:8765` 即可查看任务板。

## 架构

### 核心组件

1. **TaskBoard** (`manager.py`)
   - 任务管理核心
   - 线程安全的单例模式
   - 支持任务的添加、更新、完成、失败等操作
   - 订阅者模式支持实时通知

2. **TaskBoardServer** (`server.py`)
   - FastAPI 服务器
   - WebSocket 支持实时推送
   - RESTful API 接口

3. **前端界面** (`static/index.html`)
   - 现代化的 Web 界面
   - 实时更新（无需刷新）
   - 任务过滤和搜索
   - 响应式设计

### 任务状态

任务可以有以下状态：

- `PENDING` - 等待执行
- `RUNNING` - 正在执行
- `COMPLETED` - 已完成
- `FAILED` - 失败
- `CANCELLED` - 已取消

## API 使用

### 基本用法

```python
from rot_data.taskboard import TaskBoard, TaskStatus

# 获取任务板实例
taskboard = TaskBoard()

# 添加任务
task = taskboard.add_task(
    task_id="task_1",
    name="Load CO3D data",
    total=100,
    metadata={"category": "apple"}
)

# 开始任务
taskboard.start_task("task_1")

# 更新进度
taskboard.update_task("task_1", progress=50)

# 完成任务
taskboard.complete_task("task_1")

# 或标记为失败
# taskboard.fail_task("task_1", error="Something went wrong")
```

### 订阅任务更新

```python
def on_update():
    print("Tasks updated!")
    summary = taskboard.get_summary()
    print(f"Total: {summary['total']}, Completed: {summary['by_status']['completed']}")

taskboard.subscribe(on_update)
```

### 集成到自定义 DataLoader

```python
from rot_data.dataloader.data import DataLoader
from rot_data.taskboard import get_task_board

class MyDataLoader(DataLoader):
    def __init__(self, use_taskboard: bool = False):
        self.use_taskboard = use_taskboard
        self.taskboard = get_task_board() if use_taskboard else None
    
    def load(self):
        if self.taskboard:
            # 创建任务
            task_id = "my_task"
            self.taskboard.add_task(task_id, "Loading data", total=100)
            self.taskboard.start_task(task_id)
            
            try:
                # 执行加载逻辑
                for i in range(100):
                    # 处理数据...
                    self.taskboard.update_task(task_id, progress=i+1)
                    yield data
                
                self.taskboard.complete_task(task_id)
            except Exception as e:
                self.taskboard.fail_task(task_id, str(e))
                raise
```

## 配置选项

### 服务器配置

```python
from rot_data.taskboard.server import start_server

server = start_server(
    host="0.0.0.0",  # 绑定地址
    port=8765,        # 端口号
)
```

## 技术栈

- **后端**：FastAPI, WebSocket, Uvicorn
- **前端**：原生 HTML/CSS/JavaScript
- **数据管理**：线程安全的 Python 数据结构

## 注意事项

1. 任务板使用单例模式，确保全局只有一个实例
2. 所有操作都是线程安全的
3. WebSocket 连接会自动重连
4. 服务器在后台线程中运行，不会阻塞主程序

