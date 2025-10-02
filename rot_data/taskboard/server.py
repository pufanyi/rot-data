"""FastAPI server for the task board with WebSocket support."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .manager import get_task_board


class TaskBoardServer:
    """FastAPI server for the task board."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        """
        Initialize the task board server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        self.host = host
        self.port = port
        self.app = FastAPI(title="Task Board", version="0.1.0")
        self.taskboard = get_task_board()
        self.active_connections: list[WebSocket] = []
        self._setup_routes()
        self._server_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Subscribe to task board updates
        self.taskboard.subscribe(self._on_task_update)

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        static_dir = Path(__file__).parent / "static"

        # Serve static files
        if static_dir.exists():
            self.app.mount(
                "/static", StaticFiles(directory=str(static_dir)), name="static"
            )

        @self.app.get("/")
        async def root() -> FileResponse:
            """Serve the main HTML page."""
            index_file = static_dir / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return HTMLResponse(content="<h1>Task Board</h1><p>No UI available</p>")

        @self.app.get("/api/tasks")
        async def get_tasks() -> dict[str, Any]:
            """Get all tasks."""
            return self.taskboard.get_summary()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                # Send initial state
                await websocket.send_json(self.taskboard.get_summary())

                # Keep connection alive
                while True:
                    # Wait for messages (ping/pong)
                    try:
                        await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    except TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.send_json({"type": "ping"})
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

    def _on_task_update(self) -> None:
        """Callback when tasks are updated."""
        # Schedule broadcast in the event loop
        asyncio.run(self._broadcast_update())

    async def _broadcast_update(self) -> None:
        """Broadcast task updates to all connected clients."""
        if not self.active_connections:
            return

        summary = self.taskboard.get_summary()
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(summary)
            except Exception:
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

    def start_background(self) -> None:
        """Start the server in a background thread."""
        if self._server_thread and self._server_thread.is_alive():
            return

        def run_server() -> None:
            import uvicorn

            uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

    def stop(self) -> None:
        """Stop the server."""
        self._stop_event.set()
        self.taskboard.unsubscribe(self._on_task_update)


def start_server(host: str = "0.0.0.0", port: int = 8765) -> TaskBoardServer:
    """
    Start the task board server.

    Args:
        host: Host to bind to
        port: Port to bind to

    Returns:
        The server instance
    """
    server = TaskBoardServer(host=host, port=port)
    server.start_background()
    return server

