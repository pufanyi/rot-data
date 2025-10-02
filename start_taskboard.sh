#!/bin/bash

# 快速启动任务板演示脚本

echo "========================================="
echo "  启动任务板演示"
echo "========================================="
echo ""
echo "这将启动一个包含 441 个模拟任务的演示"
echo "任务板将在 http://localhost:8765 可用"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""
echo "========================================="
echo ""

# 运行演示脚本
uv run python examples/taskboard_demo.py

