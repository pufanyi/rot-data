# 任务板使用指南

## 快速开始

### 方法一：运行演示（推荐新手）

最简单的方式是运行包含 441 个模拟任务的演示：

```bash
# 使用快捷脚本
./start_taskboard.sh

# 或直接运行
uv run python examples/taskboard_demo.py
```

然后在浏览器中打开 `http://localhost:8765` 即可看到任务板界面。

### 方法二：与实际数据加载集成

如果你想在实际的数据加载过程中使用任务板：

```python
from rot_data.dataloader.co3d import CO3DDataLoader
from rot_data.taskboard.server import start_server
import time

# 1. 启动任务板服务器
server = start_server(host="0.0.0.0", port=8765)
print("任务板: http://localhost:8765")
time.sleep(2)  # 等待服务器启动

# 2. 创建数据加载器并启用任务板
loader = CO3DDataLoader(
    cache_dir="cache",
    num_threads=4,
    use_taskboard=True  # 关键：启用任务板
)

# 3. 加载数据
for data in loader.load():
    # 处理数据...
    pass
```

## 任务板界面说明

### 顶部统计面板

显示以下统计信息（带彩色渐变卡片）：

- **总任务数**：所有任务的总数
- **待处理**：黄色，等待执行的任务
- **运行中**：蓝色，正在执行的任务
- **已完成**：绿色，成功完成的任务
- **失败**：红色，执行失败的任务

### 任务列表

每个任务卡片显示：

- **任务名称**：如 "apple - 000"
- **状态标签**：pending/running/completed/failed
- **进度条**：显示完成百分比（仅运行中和已完成的任务）
- **进度文本**：如 "50 / 100 (50.0%)"
- **元数据**：任务 ID、开始时间、结束时间
- **错误信息**：失败任务会显示错误详情

### 过滤按钮

使用顶部的过滤按钮可以按状态筛选任务：

- **All**：显示所有任务
- **Pending**：只显示待处理的任务
- **Running**：只显示正在运行的任务
- **Completed**：只显示已完成的任务
- **Failed**：只显示失败的任务

### 连接状态

右上角显示 WebSocket 连接状态：

- 🟢 **Connected**：正常连接，实时更新
- 🔴 **Disconnected**：连接断开，会自动重连

## 在自定义 DataLoader 中使用

如果你想在自己的 DataLoader 中集成任务板：

```python
from rot_data.dataloader.data import DataLoader
from rot_data.taskboard import get_task_board

class MyDataLoader(DataLoader):
    def __init__(self, use_taskboard: bool = False):
        self.use_taskboard = use_taskboard
        self.taskboard = get_task_board() if use_taskboard else None
    
    def load(self):
        jobs = self._collect_jobs()  # 获取所有任务
        
        for idx, job in enumerate(jobs):
            # 1. 添加任务
            task_id = f"job_{idx}"
            if self.taskboard:
                self.taskboard.add_task(
                    task_id=task_id,
                    name=f"Processing job {idx}",
                    total=100,  # 如果知道总工作量
                    metadata={"job": job}
                )
            
            # 2. 开始任务
            if self.taskboard:
                self.taskboard.start_task(task_id)
            
            try:
                # 3. 执行任务并更新进度
                for i in range(100):
                    # 做一些工作...
                    if self.taskboard:
                        self.taskboard.update_task(task_id, progress=i+1)
                    yield result
                
                # 4. 完成任务
                if self.taskboard:
                    self.taskboard.complete_task(task_id)
            
            except Exception as e:
                # 5. 标记任务失败
                if self.taskboard:
                    self.taskboard.fail_task(task_id, str(e))
                raise
```

## 配置选项

### 服务器配置

```python
from rot_data.taskboard.server import start_server

server = start_server(
    host="0.0.0.0",  # 绑定所有网络接口（允许远程访问）
    port=8765,       # 服务器端口
)
```

### 远程访问

如果你在服务器上运行，可以通过 SSH 端口转发访问任务板：

```bash
ssh -L 8765:localhost:8765 user@server
```

然后在本地浏览器访问 `http://localhost:8765`。

## 技术实现

### 数据收集模式

当启用任务板时，数据处理流程会有所不同：

- **不使用任务板**：使用 `Dataset.from_generator` 流式处理
- **使用任务板**：先收集所有数据到内存，然后使用 `Dataset.from_list` 创建数据集

**原因**：
- 任务板包含 WebSocket 连接和订阅者回调
- 这些对象无法被 pickle 序列化
- `Dataset.from_generator` 会尝试序列化 generator 函数进行哈希

**影响**：
- ✅ 数据加载仍然是多线程并发的（由 `--num-threads` 控制）
- ✅ 完全避免了序列化问题
- ⚠️ 需要额外的内存来存储所有数据项（通常不是问题）

## 常见问题

### Q: 任务板显示 "Disconnected"？

**A:** 检查以下几点：

1. 确认服务器是否正在运行
2. 检查端口 8765 是否被占用
3. 查看控制台是否有错误信息
4. 页面会自动重连，稍等片刻

### Q: 看不到任务更新？

**A:** 确保：

1. 数据加载器启用了 `use_taskboard=True`
2. WebSocket 连接状态显示为 "Connected"
3. 刷新页面重新连接

### Q: 如何更改端口？

**A:** 修改 `start_server()` 的 `port` 参数：

```python
server = start_server(host="0.0.0.0", port=9000)
```

### Q: 任务数量太多怎么办？

**A:** 使用过滤按钮筛选特定状态的任务，或者只关注统计面板。

### Q: 可以同时运行多个数据加载器吗？

**A:** 可以！TaskBoard 是单例模式，所有 DataLoader 的任务都会显示在同一个任务板上。

## 技术细节

- **后端**：FastAPI + WebSocket
- **前端**：原生 HTML/CSS/JavaScript
- **通信**：实时 WebSocket 推送
- **线程安全**：所有操作都是线程安全的
- **自动重连**：连接断开后自动重连

## 示例输出

运行演示后，你会在任务板上看到：

```
统计面板:
- 总任务数: 441
- 待处理: 350
- 运行中: 40
- 已完成: 45
- 失败: 6

任务列表:
[蓝色边框] Load CO3D Subset 1/441
  状态: running
  进度: ▓▓▓▓▓▓▓░░░ 75/100 (75.0%)
  ID: load_task_0
  开始时间: 14:35:20

[绿色边框] Load CO3D Subset 2/441
  状态: completed
  进度: ▓▓▓▓▓▓▓▓▓▓ 150/150 (100.0%)
  ID: load_task_1
  开始时间: 14:35:18
  结束时间: 14:35:45

...
```

## 下一步

- 查看 [任务板模块文档](rot_data/taskboard/README.md) 了解更多 API 细节
- 查看 [示例代码](examples/) 了解更多使用场景
- 根据你的需求定制任务板功能

