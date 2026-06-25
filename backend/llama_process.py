from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import IO, Any

import httpx

from .config import Settings


class LlamaProcessError(RuntimeError):
    pass


class LlamaProcessManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process: subprocess.Popen[str] | None = None
        self.started_by_app = False
        self.status = "stopped"
        self.message = "模型服务尚未启动"
        self._lock = asyncio.Lock()
        self._log_handle: IO[str] | None = None
        self.context_size = settings.n_ctx

    def set_context_size(self, value: int) -> None:
        if value not in {32768, 65536}:
            raise ValueError("context_size_must_be_32768_or_65536")
        self.context_size = value

    def _binary_path(self) -> str | None:
        configured = Path(self.settings.llama_server_bin).expanduser()
        if configured.is_absolute() or "/" in self.settings.llama_server_bin:
            return str(configured) if configured.is_file() else None
        return shutil.which(self.settings.llama_server_bin)

    async def is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                response = await client.get(f"{self.settings.llama_base_url}/health")
            return response.status_code == 200 and response.json().get("status") == "ok"
        except (httpx.HTTPError, ValueError):
            return False

    async def start(self) -> dict[str, Any]:
        async with self._lock:
            if await self.is_healthy():
                self.status = "ready"
                if not self.started_by_app:
                    self.message = "已连接到正在运行的 llama-server"
                return await self.runtime_info()

            if self.process is not None and self.process.poll() is None:
                self.status = "loading"
                return await self.runtime_info(check_health=False)

            binary = self._binary_path()
            if binary is None:
                self.status = "error"
                self.message = "未找到 llama-server，请先安装 llama.cpp"
                raise LlamaProcessError(self.message)
            if not self.settings.model_path.is_file():
                self.status = "error"
                self.message = f"未找到模型文件：{self.settings.model_path}"
                raise LlamaProcessError(self.message)

            log_path = self.settings.project_root / "data" / "llama-server.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = log_path.open("a", encoding="utf-8")
            command = [
                binary,
                "--model",
                str(self.settings.model_path),
                "--host",
                self.settings.llama_host,
                "--port",
                str(self.settings.llama_port),
                "--ctx-size",
                str(self.context_size),
                "--n-gpu-layers",
                self.settings.n_gpu_layers,
                "--parallel",
                str(self.settings.n_parallel),
                "--cache-ram",
                str(self.settings.prompt_cache_mb),
                "--cache-type-k",
                self.settings.cache_type_k,
                "--cache-type-v",
                self.settings.cache_type_v,
                "--reasoning",
                self.settings.reasoning,
                "--no-ui",
                "--log-colors",
                "off",
            ]
            self.status = "loading"
            self.message = "正在加载本地模型…"
            try:
                self.process = subprocess.Popen(
                    command,
                    cwd=self.settings.project_root,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            except OSError as exc:
                self._close_log()
                self.status = "error"
                self.message = f"llama-server 启动失败：{exc}"
                raise LlamaProcessError(self.message) from exc
            self.started_by_app = True

            deadline = asyncio.get_running_loop().time() + self.settings.llama_start_timeout
            while asyncio.get_running_loop().time() < deadline:
                if self.process.poll() is not None:
                    exit_code = self.process.returncode
                    self.status = "error"
                    self.message = f"模型服务异常退出（代码 {exit_code}），请查看 data/llama-server.log"
                    self._close_log()
                    raise LlamaProcessError(self.message)
                if await self.is_healthy():
                    self.status = "ready"
                    self.message = "本地模型已就绪"
                    return await self.runtime_info(check_health=False)
                await asyncio.sleep(0.75)

            await self._stop_unlocked()
            self.status = "error"
            self.message = "模型加载超时，请查看 data/llama-server.log"
            raise LlamaProcessError(self.message)

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        process = self.process
        if process is not None and process.poll() is None and self.started_by_app:
            process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=8)
            except TimeoutError:
                process.kill()
                await asyncio.to_thread(process.wait)
        self.process = None
        self.started_by_app = False
        self.status = "stopped"
        self.message = "模型服务已停止"
        self._close_log()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    async def runtime_info(self, *, check_health: bool = True) -> dict[str, Any]:
        healthy = await self.is_healthy() if check_health else self.status == "ready"
        if healthy:
            self.status = "ready"
            if self.message in {"模型服务尚未启动", "模型服务已停止"}:
                self.message = "本地模型已就绪"
        elif self.process is not None and self.process.poll() is not None:
            self.status = "error"
            self.message = (
                f"模型服务异常退出（代码 {self.process.returncode}），"
                "请查看 data/llama-server.log"
            )
        return {
            "status": self.status,
            "message": self.message,
            "healthy": healthy,
            "started_by_app": self.started_by_app,
            "model_name": self.settings.model_path.name,
            "model_path": str(self.settings.model_path),
            "llama_url": self.settings.llama_base_url,
            "context_size": self.context_size,
            "cache_type_k": self.settings.cache_type_k,
            "cache_type_v": self.settings.cache_type_v,
            "reasoning": self.settings.reasoning,
        }
