# 🚀 开始使用任务板

## ✅ 集成完成！

任务板已成功集成到 `prepare_data.py`，现在你可以用**一条命令**来运行带任务板的数据准备了！

## 🎯 立即开始

### 方式一：运行真实数据加载（推荐）

```bash
uv run python -m rot_data.pipeline.prepare_data --use-taskboard --no-push
```

这将：
- ✅ 自动启动任务板服务器
- ✅ 加载 CO3D 数据集
- ✅ 在浏览器实时显示 441 个任务的状态
- ✅ 不推送到 Hugging Face

**然后打开浏览器访问**：`http://localhost:8765`

### 方式二：运行演示（快速体验）

如果只想看看任务板的效果：

```bash
./start_taskboard.sh
```

这会模拟 441 个任务，让你快速体验任务板功能。

## 📺 你会看到什么？

### 控制台输出

```
============================================================
Task Board Enabled!
Open http://localhost:8765 in your browser
============================================================
2025-10-02 15:30:45 | INFO | Starting task board server...
2025-10-02 15:30:47 | SUCCESS | Task board available at http://localhost:8765
2025-10-02 15:30:47 | INFO | Open the URL in your browser to view task progress
```

### 浏览器界面

打开 `http://localhost:8765` 后，你会看到：

#### 📊 顶部统计面板（彩色卡片）
```
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│   总任务    │   待处理    │   运行中    │   已完成    │    失败     │
│    441      │     320     │      4      │     115     │      2      │
│   (灰色)    │   (黄色)    │   (蓝色)    │   (绿色)    │   (红色)    │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

#### 📋 任务列表（实时更新）
```
┌─────────────────────────────────────────┐
│ 🔵 apple - 000                          │
│ ● running                               │
│ ▓▓▓▓▓▓▓░░░ 75/100 (75.0%)             │
│ ID: load_apple_0                        │
│ 开始时间: 15:30:50                      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ 🟢 banana - 001                         │
│ ✓ completed                             │
│ ▓▓▓▓▓▓▓▓▓▓ 150/150 (100.0%)           │
│ ID: load_banana_1                       │
│ 开始: 15:30:48 | 结束: 15:31:15         │
└─────────────────────────────────────────┘
```

#### 🔍 过滤按钮
- **All** - 显示所有任务
- **Pending** - 只显示待处理
- **Running** - 只显示运行中
- **Completed** - 只显示已完成
- **Failed** - 只显示失败的

## 🛠️ 完整命令参数

```bash
uv run python -m rot_data.pipeline.prepare_data \
    --dataset-name co3d \          # 数据集名称（默认 co3d）
    --cache-dir cache \            # 缓存目录
    --num-threads 4 \              # 并发线程数
    --num-proc 4 \                 # 处理进程数*
    --use-taskboard \              # ⭐ 启用任务板
    --taskboard-port 8765 \        # 任务板端口
    --no-push \                    # 不推送到 HF Hub
    --log-level INFO               # 日志级别
```

**⚠️ 重要提示**：启用任务板时，会先收集所有数据到内存再创建数据集，以避免序列化问题。数据加载仍然使用 `--num-threads` 参数进行多线程并发。

## 💡 使用场景

### 本地开发
```bash
uv run python -m rot_data.pipeline.prepare_data \
    --use-taskboard \
    --no-push \
    --num-threads 4
```

### 生产环境
```bash
uv run python -m rot_data.pipeline.prepare_data \
    --use-taskboard \
    --repo-id your-username/your-dataset
```

### 远程服务器
```bash
# 服务器上运行
uv run python -m rot_data.pipeline.prepare_data --use-taskboard

# 本地建立 SSH 隧道
ssh -L 8765:localhost:8765 user@server

# 本地浏览器访问 http://localhost:8765
```

## ✨ 功能特性

- 🎯 **实时追踪** - WebSocket 自动推送，无需刷新
- 📊 **统计面板** - 一目了然的任务状态统计
- 🎨 **美观界面** - 现代化渐变色设计
- 🔍 **任务过滤** - 按状态筛选查看
- 📈 **进度可视化** - 进度条 + 百分比
- ⚠️ **错误展示** - 失败任务显示详细错误
- 🔄 **自动重连** - 连接断开自动恢复
- 📱 **响应式** - 支持手机查看

## 📚 更多文档

- **快速开始指南**：[QUICKSTART.md](QUICKSTART.md) - 详细的快速入门
- **使用说明**：[TASKBOARD_USAGE.md](TASKBOARD_USAGE.md) - 完整功能说明
- **技术文档**：[rot_data/taskboard/README.md](rot_data/taskboard/README.md) - API 文档
- **集成完成**：[INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) - 集成说明
- **实现总结**：[TASKBOARD_SUMMARY.md](TASKBOARD_SUMMARY.md) - 技术总结

## ✅ 验证状态

- ✅ 任务板核心模块已实现
- ✅ FastAPI 服务器正常运行
- ✅ 前端界面完整
- ✅ 集成到 CO3DDataLoader
- ✅ 集成到 prepare_data.py
- ✅ 所有测试通过（8/8）
- ✅ 代码检查通过
- ✅ 文档齐全

## 🎊 开始使用吧！

现在就试试吧：

```bash
uv run python -m rot_data.pipeline.prepare_data --use-taskboard --no-push
```

然后在浏览器打开 `http://localhost:8765`，享受实时任务追踪的乐趣！🚀

---

**提示**：第一次运行可能需要下载数据，请耐心等待。你可以在任务板上实时看到每个任务的进度！

