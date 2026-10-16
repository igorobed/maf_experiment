    """Вариант б: OpenCode в режиме сервера, наш код отвечает на его запросы прав.

Чем отличается от варианта а (opencode_cli_loop.py):
- в CLI (`opencode run`) право "ask" некому подтвердить — OpenCode сам отвечает
  отказом. В режиме сервера запрос прав повисает и ждёт ответа по HTTP, поэтому
  решение «можно / нельзя» принимает наш код (здесь — спрашивает пользователя);
- сессию можно прервать запросом abort, не убивая процесс.

Схема одного запуска:
    1. opencode serve                       -> локальный HTTP-сервер OpenCode
    2. POST /session                        -> новая сессия
    3. POST /session/{id}/prompt_async      -> задача, ответ приходит сразу
    4. раз в секунду:
         GET  /permission                   -> есть запросы прав?
         POST /permission/{id}/reply        -> наш ответ: once / always / reject
         GET  /session/status               -> OpenCode ещё работает?
    5. GET /session/{id}/message            -> итог: текст и вызванные инструменты
    6. останавливаем сервер

Используются «классические» методы API из документации OpenCode
(https://opencode.ai/docs/server). Полная спецификация запущенного сервера:
GET http://127.0.0.1:<порт>/doc.

OpenCodeServerAgent — агент MAF (наследник BaseAgent). Здесь он запускается
напрямую, к главному агенту подключается как в opencode_agent.py:
tools=[coder.as_tool()]. Подробности и подводные камни — в OPENCODE_NOTES.md.

    .venv\\Scripts\\python opencode_server.py
    .venv\\Scripts\\python opencode_server.py --workdir D:\\tmp\\ws_srv "задача"
"""

import argparse
import asyncio
import base64
import json
import os
import secrets
import socket
import time
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    BaseAgent,
    Content,
    ResponseStream,
    normalize_messages,
)

from minimal_agent import OPENROUTER_API_KEY

HERE = Path(__file__).parent
OPENCODE = HERE / "bin" / "opencode.exe"  # OpenCode CLI v1.19.1
MODEL = {"providerID": "openrouter", "modelID": "qwen/qwen3.8-27b"}
DEFAULT_TASK = "Создай hello.py, который печатает «Привет», и запусти его."

# Функция одобрения: получает запрос прав OpenCode, возвращает "once",
# "always" или "reject". В консоли спрашивает человека, в UI показала бы карточку.
Approve = Callable[[dict[str, Any]], Awaitable[str]]


def opencode_config() -> dict[str, Any]:
    """Конфиг сервера OpenCode (схема 1.x). Главное отличие от варианта а — "ask"."""
    return {
        "agent": {
            "coder": {
                "description": "Пишет и запускает код",
                "mode": "primary",
                "prompt": (
                    "Ты программист. Работай только в текущей папке. "
                    "Если действие отклонено, не повторяй его, а объясни, что не так."
                ),
                "steps": 25,
            }
        },
        "permission": {
            "edit": "ask",  # каждую запись файла подтверждает наш код
            "bash": "ask",  # каждую команду тоже
            "external_directory": "deny",  # вне рабочей папки — отказ без вопросов
            "webfetch": "deny",
            "task": "deny",  # без встроенных субагентов OpenCode
        },
        "provider": {"openrouter": {"options": {"apiKey": OPENROUTER_API_KEY}}},
        "autoupdate": False,
        "share": "disabled",
    }


class OpenCodeServer:
    """Минимальный клиент: запустить `opencode serve`, делать запросы, остановить."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.port = free_port()
        # Пароль на сервер: без него любой процесс на машине может управлять
        # OpenCode через этот порт. Логин по умолчанию — "opencode".
        self.password = secrets.token_urlsafe(16)
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        env = {
            **os.environ,
            "OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config()),
            "OPENCODE_SERVER_PASSWORD": self.password,
            # Изоляция от чужих скиллов и правил (~/.claude, ~/.agents).
            "OPENCODE_DISABLE_CLAUDE_CODE": "1",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        }
        self.process = await asyncio.create_subprocess_exec(
            str(OPENCODE), "serve", "--pure",
            "--hostname", "127.0.0.1", "--port", str(self.port),
            cwd=self.workdir,  # папка проекта для запросов без ?directory=
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        # Ждём, пока сервер начнёт отвечать (холодный старт — несколько секунд).
        for _ in range(60):
            try:
                await self.call("GET", "/global/health")
                return
            except OSError:
                await asyncio.sleep(0.5)
        raise RuntimeError("opencode serve не поднялся за 30 с")

    async def call(self, method: str, path: str, body: Any = None) -> Any:
        """HTTP-запрос к серверу. urllib синхронный, поэтому уносим его в поток."""
        return await asyncio.to_thread(self._call, method, path, body)

    def _call(self, method: str, path: str, body: Any) -> Any:
        token = base64.b64encode(f"opencode:{self.password}".encode()).decode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            method=method,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Basic {token}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        return json.loads(raw) if raw else None  # prompt_async отвечает пустым 204

    async def stop(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.kill()
            await self.process.wait()


class OpenCodeServerAgent(BaseAgent):
    """Агент MAF: задача уходит в OpenCode-сервер, запросы прав решает approve."""

    def __init__(
        self, *, workdir: Path, approve: Approve, timeout: float = 600, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.workdir = workdir.resolve()
        self.approve = approve
        self.timeout = timeout

    # Договор с MAF как в opencode_agent.py: stream=True -> ResponseStream.
    def run(  # type: ignore[override]
        self, messages: Any = None, *, stream: bool = False, session: Any = None,
        **kwargs: Any,
    ) -> Any:
        normalized = normalize_messages(messages)
        task = normalized[-1].text if normalized else ""
        updates = self._updates(task)
        if stream:
            return ResponseStream(updates, finalizer=AgentResponse.from_updates)
        return self._collect(updates)

    @staticmethod
    async def _collect(updates: AsyncIterator[AgentResponseUpdate]) -> AgentResponse:
        return AgentResponse.from_updates([u async for u in updates])

    async def _updates(self, task: str) -> AsyncIterator[AgentResponseUpdate]:
        self.workdir.mkdir(parents=True, exist_ok=True)
        server = OpenCodeServer(self.workdir)
        await server.start()
        session_id = None
        try:
            session = await server.call("POST", "/session", {"title": "maf"})
            session_id = session["id"]
            # prompt_async возвращается сразу; работа идёт на сервере.
            await server.call("POST", f"/session/{session_id}/prompt_async", {
                "parts": [{"type": "text", "text": task}],
                "model": MODEL,
                "agent": "coder",
            })
            deadline = time.monotonic() + self.timeout
            while True:
                # 1. Запросы прав нашей сессии. Пока мы не ответим, OpenCode ждёт.
                for request in await server.call("GET", "/permission") or []:
                    if request.get("sessionID") != session_id:
                        continue  # чужая сессия (на сервере их может быть несколько)
                    decision = await self.approve(request)
                    reply: dict[str, Any] = {"reply": decision}
                    if decision == "reject":
                        # Текст отказа доходит до модели как обратная связь.
                        reply["message"] = "Пользователь отклонил это действие."
                    path = f"/permission/{request['id']}/reply"
                    await server.call("POST", path, reply)
                    yield say(f"[права] {request['permission']} "
                              f"{request.get('patterns')} -> {decision}\n")
                # 2. Закончил ли OpenCode.
                if await self._finished(server, session_id):
                    break
                # 3. Таймаут: прерываем сессию запросом, процесс жив до finally.
                if time.monotonic() > deadline:
                    await server.call("POST", f"/session/{session_id}/abort")
                    yield say(f"[opencode] прерван по таймауту {self.timeout:.0f} с\n")
                    break
                await asyncio.sleep(1)  # опрос раз в секунду: просто и достаточно
            yield say(await self._summary(server, session_id))
        finally:
            # Ctrl+C или отмена задачи тоже приходят сюда: сервер не оставляем.
            await server.stop()

    @staticmethod
    async def _finished(server: OpenCodeServer, session_id: str) -> bool:
        """Сессия свободна и последнее сообщение — завершённый ответ модели."""
        status = await server.call("GET", "/session/status") or {}
        # В карте статусов бывают только занятые сессии; нет записи — свободна.
        if status.get(session_id, {}).get("type", "idle") != "idle":
            return False
        messages = await server.call("GET", f"/session/{session_id}/message") or []
        last = messages[-1]["info"] if messages else {}
        # Сразу после prompt_async сессия ещё может числиться свободной,
        # поэтому ждём именно завершённого ответа ассистента.
        return last.get("role") == "assistant" and bool(
            (last.get("time") or {}).get("completed")
        )

    @staticmethod
    async def _summary(server: OpenCodeServer, session_id: str) -> str:
        """Итог из истории сессии: вызванные инструменты, текст ответа, стоимость."""
        messages = await server.call("GET", f"/session/{session_id}/message") or []
        lines, cost = [], 0.0
        for message in messages:
            info = message["info"]
            if info.get("role") != "assistant":
                continue
            cost += info.get("cost") or 0.0
            for part in message.get("parts", []):
                if part.get("type") == "tool":
                    state = part.get("state") or {}
                    lines.append(f"[opencode] {part.get('tool')} {state.get('status')}")
                elif part.get("type") == "text" and part.get("text", "").strip():
                    lines.append(part["text"].strip())
        lines.append(f"[opencode] стоимость ${cost:.4f}")
        return "\n".join(lines) + "\n"


def say(text: str) -> AgentResponseUpdate:
    """Кусок ответа агента с текстом."""
    return AgentResponseUpdate(role="assistant", contents=[Content.from_text(text)])


def free_port() -> int:
    """Свободный локальный порт: ОС выдаёт его при привязке к порту 0."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def ask_in_console(request: dict[str, Any]) -> str:
    """Одобрение в консоли. request: permission (edit, bash…), patterns, metadata."""
    print(f"\n[одобрение] OpenCode просит: {request['permission']} "
          f"{request.get('patterns')}")
    answer = await asyncio.to_thread(input, "да / всегда / нет > ")
    return {"да": "once", "всегда": "always"}.get(answer.strip().lower(), "reject")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=HERE / "workspace_server")
    parser.add_argument("task", nargs="*")
    args = parser.parse_args()
    task = " ".join(args.task) or DEFAULT_TASK

    coder = OpenCodeServerAgent(
        workdir=args.workdir, approve=ask_in_console, name="coder",
        description="Пишет и запускает код через OpenCode-сервер с одобрением действий",
    )
    print(f"Рабочая папка: {coder.workdir}\nЗадача: {task}")
    async for update in coder.run(task, stream=True):
        print(update.text, end="", flush=True)
    print(f"файлы в рабочей папке: {sorted(p.name for p in coder.workdir.iterdir())}")


if __name__ == "__main__":
    asyncio.run(main())
