# 快速开始指南

## 使用任务板运行数据准备

### 最简单的方式

运行 `prepare_data.py` 并启用任务板：

```bash
uv run python -m rot_data.pipeline.prepare_data --use-taskboard --no-push
```

这将：
1. ✅ 自动启动任务板服务器（http://localhost:8765）
2. ✅ 加载 CO3D 数据集
3. ✅ 实时显示所有任务的进度
4. ✅ 跳过推送到 Hugging Face（使用 `--no-push`）

### 在浏览器中查看

打开 `http://localhost:8765`，你会看到：

- 📊 **统计面板**：总任务数、待处理、运行中、已完成、失败
- 📋 **任务列表**：每个任务的详细状态和进度
- 🔍 **过滤功能**：按状态筛选任务
- 🎨 **实时更新**：无需刷新，自动更新

### 完整命令参数

```bash
uv run python -m rot_data.pipeline.prepare_data \
    --dataset-name co3d \              # 数据集名称（默认 co3d）
    --cache-dir cache \                # 缓存目录（默认 cache）
    --num-threads 4 \                  # 并发线程数（默认 32）
    --num-proc 4 \                     # 处理进程数（默认 32）*
    --use-taskboard \                  # 启用任务板 ⭐
    --taskboard-port 8765 \            # 任务板端口（默认 8765）
    --no-push \                        # 不推送到 Hugging Face
    --log-level INFO                   # 日志级别（默认 INFO）
```

**注意**：启用任务板时，会先收集所有数据到内存再创建数据集，以避免序列化问题。数据加载仍然是多线程并发的。

### 只运行演示

如果你只想看看任务板的效果（不实际加载数据）：

```bash
# 使用快捷脚本
./start_taskboard.sh

# 或直接运行
uv run python examples/taskboard_demo.py
```

这会模拟 441 个任务，让你快速体验任务板功能。

## 常用场景

### 场景 1: 开发调试

加载数据并实时监控，不推送到 Hub：

```bash
uv run python -m rot_data.pipeline.prepare_data \
    --use-taskboard \
    --no-push \
    --num-threads 4
```

### 场景 2: 生产环境

完整流程，加载并推送数据：

```bash
uv run python -m rot_data.pipeline.prepare_data \
    --dataset-name co3d \
    --repo-id your-username/your-dataset \
    --use-taskboard \
    --num-threads 32
```

### 场景 3: 远程服务器

在服务器上运行，通过 SSH 端口转发查看任务板：

```bash
# 本地机器上执行
ssh -L 8765:localhost:8765 user@server

# 服务器上执行
uv run python -m rot_data.pipeline.prepare_data --use-taskboard
```

然后在本地浏览器访问 `http://localhost:8765`

## 任务板功能说明

### 界面元素

1. **连接状态**（右上角）
   - 🟢 Connected - 正常连接
   - 🔴 Disconnected - 断线（会自动重连）

2. **统计卡片**
   - 总任务数 - 灰色
   - 待处理 - 黄色
   - 运行中 - 蓝色
   - 已完成 - 绿色
   - 失败 - 红色/橙色

3. **任务卡片**
   - 任务名称：如 "apple - 000"
   - 状态标签：当前状态
   - 进度条：完成百分比
   - 时间信息：开始/结束时间
   - 错误信息：失败任务显示错误详情

### 过滤功能

使用顶部的过滤按钮：
- **All** - 显示所有任务
- **Pending** - 只显示待处理的任务
- **Running** - 只显示运行中的任务
- **Completed** - 只显示已完成的任务
- **Failed** - 只显示失败的任务

## 实际效果

运行后你会看到类似这样的输出：

```
============================================================
Task Board Enabled!
Open http://localhost:8765 in your browser
============================================================
2025-10-02 15:30:45 | INFO | Starting task board server at http://0.0.0.0:8765
2025-10-02 15:30:47 | SUCCESS | Task board available at http://localhost:8765
2025-10-02 15:30:47 | INFO | Open the URL in your browser to view task progress
2025-10-02 15:30:49 | INFO | Building Hugging Face dataset using 'co3d' loader
...
```

浏览器中会实时显示：

```
统计面板:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总任务数: 441
待处理: 320
运行中: 4
已完成: 115
失败: 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

任务列表:
┌─────────────────────────────┐
│ apple - 000                 │
│ ● running                   │
│ ▓▓▓▓▓▓▓░░░ 75.0%           │
│ Started: 15:30:50           │
└─────────────────────────────┘
...
```

## 提示和技巧

1. **多开任务板**：可以在多个浏览器标签中打开任务板，所有标签都会实时同步

2. **移动设备**：任务板是响应式设计，可以在手机上查看

3. **性能**：任务板开销很小，不会显著影响数据加载性能

4. **自定义端口**：如果 8765 端口被占用，使用 `--taskboard-port 9000` 更换端口

5. **日志级别**：使用 `--log-level DEBUG` 查看更详细的日志

## 故障排除

### 问题：浏览器显示无法连接

**解决方案**：
1. 确认任务板服务器已启动（查看控制台日志）
2. 检查端口是否被占用
3. 尝试更换端口：`--taskboard-port 9000`

### 问题：任务列表不更新

**解决方案**：
1. 检查 WebSocket 连接状态（页面右上角）
2. 刷新页面重新连接
3. 检查防火墙设置

### 问题：任务板界面显示空白

**解决方案**：
1. 等待几秒，服务器可能还在启动
2. 检查浏览器控制台是否有错误
3. 尝试使用 Chrome/Firefox 等现代浏览器

## 下一步

- 查看 [详细文档](TASKBOARD_USAGE.md) 了解更多功能
- 查看 [示例代码](examples/) 学习如何在代码中使用
- 查看 [模块文档](rot_data/taskboard/README.md) 了解 API 细节

---

享受实时任务追踪的乐趣！🎉

