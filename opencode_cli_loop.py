"""Вариант а: OpenCode через CLI + наш внешний цикл проверки.

Идея: модель пишет код, а проверяет его НАША программа, а не сама модель.

    1. Кладём в рабочую папку тесты — это «контракт» задачи.
    2. opencode run "задача"                    -> OpenCode пишет код.
    3. python -m pytest                          -> проверяем мы.
    4. Тесты упали? opencode run --session <id> "вот ошибки, исправь"
       (та же сессия OpenCode: он помнит, что уже сделал).
    5. Повторяем до MAX_ATTEMPTS раз и возвращаем итог.

Что ещё показывает файл:
- свой агент OpenCode (промпт + права) задаётся в конфиге, а не файлами в папке;
- скилл подключается из папки ВНЕ рабочей (opencode_ext/skills), чтобы
  OpenCode не мог его переписать;
- тесты защищены правом edit: менять их OpenCode не может;
- OpenCodeLoopAgent — обычный агент MAF (наследник BaseAgent). Здесь он
  запускается напрямую, а к главному агенту подключается так же, как в
  opencode_agent.py: tools=[coder.as_tool()].

Подробности и подводные камни — в OPENCODE_NOTES.md.

    .venv\\Scripts\\python opencode_cli_loop.py
    .venv\\Scripts\\python opencode_cli_loop.py --workdir D:\\tmp\\ws_cli
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
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
OPENCODE_MODEL = "openrouter/qwen/qwen3.8-27b"  # формат OpenCode: provider/model
SKILLS_DIR = HERE / "opencode_ext" / "skills"  # скиллы лежат вне рабочей папки

# Промпт агента OpenCode. Он ЗАМЕНЯЕТ базовый промпт OpenCode; окружение,
# AGENTS.md и список скиллов OpenCode всё равно добавит сам.
CODER_PROMPT = (
    "Ты программист. Работай только в текущей папке. Перед тем как писать "
    "Python-код, загрузи скилл python-style. Тесты не меняй и не удаляй. "
    "В конце коротко перечисли, какие файлы создал или изменил."
)

# Контракт задачи: эти тесты мы кладём в папку сами, OpenCode их только читает.
TESTS = '''\
from slugify import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_punctuation():
    assert slugify("  Hello, World!  ") == "hello-world"


def test_many_separators():
    assert slugify("a  --  b__c") == "a-b-c"


def test_empty():
    assert slugify("") == ""
'''
TASK = (
    "Создай slugify.py с функцией slugify(text: str) -> str, чтобы проходили "
    "тесты из test_slugify.py."
)


def opencode_config() -> dict[str, Any]:
    """Конфиг OpenCode на запуск: агент, права, скиллы, ключ.

    Уходит в OpenCode через переменную OPENCODE_CONFIG_CONTENT и накладывается
    поверх остальных конфигов. Ключи — в схеме OpenCode 1.x (agent, permission);
    в OpenCode 2.0 они переименованы (см. OPENCODE_NOTES.md).
    """
    return {
        "agent": {
            "coder": {
                "description": "Пишет код по тестам",
                "mode": "primary",  # для `run --agent` нужен primary (или all)
                "prompt": CODER_PROMPT,
                "steps": 25,  # мягкий лимит шагов агента
            }
        },
        "permission": {
            # Правила для edit сравниваются по пути; в папке без git нужен "**/",
            # иначе шаблон не совпадёт (проверено). Побеждает ПОСЛЕДНЕЕ
            # подходящее правило, поэтому "*" идёт первым.
            "edit": {
                "*": "allow",
                "**/test_*.py": "deny",  # тесты — наш контракт
                "**/.opencode/**": "deny",  # чтобы не подложил себе правила
                "**/AGENTS.md": "deny",
            },
            "bash": "deny",  # тесты гоняем мы, оболочка OpenCode не нужна
            # Без этого OpenCode пытается гонять тесты через своего встроенного
            # субагента (инструмент task) — лишние токены без пользы.
            "task": "deny",
            "external_directory": "deny",  # ничего вне рабочей папки
            "webfetch": "deny",
            "skill": "allow",
        },
        "skills": {"paths": [str(SKILLS_DIR)]},
        # Ключ в конфиге важнее ключа из auth.json десктопного OpenCode.
        "provider": {"openrouter": {"options": {"apiKey": OPENROUTER_API_KEY}}},
        "autoupdate": False,
        "share": "disabled",
    }


def opencode_env() -> dict[str, str]:
    """Окружение для процесса OpenCode: наш конфиг + изоляция от чужих настроек."""
    return {
        **os.environ,
        "OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config()),
        # Без этих двух OpenCode подмешает ~/.claude/skills, ~/.agents/skills
        # и ~/.claude/CLAUDE.md — чужие скиллы и правила.
        "OPENCODE_DISABLE_CLAUDE_CODE": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
    }


class OpenCodeLoopAgent(BaseAgent):
    """Агент MAF: OpenCode пишет код, мы проверяем, при ошибке — следующая попытка."""

    def __init__(
        self,
        *,
        workdir: Path,
        check: list[str],
        max_attempts: int = 3,
        timeout: float = 300,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.workdir = workdir.resolve()
        self.check = check  # команда проверки, выполняется в рабочей папке
        self.max_attempts = max_attempts
        self.timeout = timeout  # на один запуск OpenCode

    # Договор с MAF такой же, как в opencode_agent.py: stream=True -> ResponseStream
    # (так вызывает as_tool), stream=False -> awaitable с AgentResponse.
    def run(  # type: ignore[override]
        self, messages: Any = None, *, stream: bool = False, session: Any = None,
        **kwargs: Any,
    ) -> Any:
        normalized = normalize_messages(messages)
        task = normalized[-1].text if normalized else ""
        updates = self._loop(task)
        if stream:
            return ResponseStream(updates, finalizer=AgentResponse.from_updates)
        return self._collect(updates)

    @staticmethod
    async def _collect(updates: AsyncIterator[AgentResponseUpdate]) -> AgentResponse:
        return AgentResponse.from_updates([u async for u in updates])

    async def _loop(self, task: str) -> AsyncIterator[AgentResponseUpdate]:
        """Внешний цикл: OpenCode -> проверка -> при ошибке продолжение сессии."""
        session_id: str | None = None  # id сессии OpenCode между попытками
        prompt = task
        for attempt in range(1, self.max_attempts + 1):
            yield say(f"\n--- попытка {attempt}/{self.max_attempts}\n")
            session_id, report = await self._run_opencode(prompt, session_id)
            yield say(report)
            ok, output = await self._run_check()
            if ok:
                yield say("[проверка] тесты прошли\n")
                return
            tail = output[-1500:]  # хвоста pytest хватает, весь вывод раздует контекст
            yield say(f"[проверка] тесты упали:\n{tail}\n")
            # Обратная связь уходит в ТУ ЖЕ сессию OpenCode (--session).
            prompt = (
                "Проверка не прошла. Вывод pytest:\n"
                f"{tail}\n"
                "Исправь код так, чтобы тесты прошли. Тесты не меняй."
            )
        yield say(f"[итог] не удалось за {self.max_attempts} попыток\n")

    async def _run_opencode(
        self, prompt: str, session_id: str | None
    ) -> tuple[str | None, str]:
        """Один запуск `opencode run`; возвращает id сессии и краткий отчёт."""
        args = [
            str(OPENCODE), "run",
            "--pure",  # без плагинов: и npm, и локальных .opencode/plugins
            "--format", "json",  # события по одному JSON на строку
            "--dir", str(self.workdir),  # папка, которую OpenCode видит и меняет
            "-m", OPENCODE_MODEL,
            "--agent", "coder",  # наш агент из конфига
        ]
        if session_id:
            args += ["--session", session_id]  # продолжить ту же сессию
        args.append(prompt)

        process = await asyncio.create_subprocess_exec(
            *args,
            env=opencode_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            # communicate() читает stdout и stderr целиком — проще, чем построчно,
            # но прогресс виден только после завершения запуска.
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return session_id, f"[opencode] прерван по таймауту {self.timeout:.0f} с\n"

        lines, cost = [], 0.0
        for raw in stdout.decode("utf-8", "replace").splitlines():
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            session_id = event.get("sessionID") or session_id
            part = event.get("part") or {}
            if event.get("type") == "tool_use":
                state = part.get("state") or {}
                target = (state.get("input") or {}).get("filePath") or ""
                status = state.get("status")  # completed или error (например, запрет)
                name = Path(target).name
                lines.append(f"[opencode] {part.get('tool')} {name} {status}")
            elif event.get("type") == "text" and part.get("text", "").strip():
                lines.append(part["text"].strip())
            elif event.get("type") == "step_finish":
                cost += part.get("cost") or 0.0
            elif event.get("type") == "error":
                lines.append(f"[opencode] ошибка: {event.get('error')}")
        if process.returncode != 0 and stderr:  # упал без события error
            error_tail = stderr.decode("utf-8", "replace")[-500:]
            lines.append(f"[opencode] stderr: {error_tail}")
        lines.append(f"[opencode] стоимость запуска ${cost:.4f}")
        return session_id, "\n".join(lines) + "\n"

    async def _run_check(self) -> tuple[bool, str]:
        """Проверка результата НАШЕЙ командой (здесь pytest). Код 0 = успех."""
        process = await asyncio.create_subprocess_exec(
            *self.check,
            cwd=self.workdir,
            # Без UTF-8 вывод Python в pipe на Windows идёт в cp1251, и кириллица
            # в ошибках доходит до модели кракозябрами.
            env={**os.environ, "PYTHONUTF8": "1"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=120)
        return process.returncode == 0, stdout.decode("utf-8", "replace")


def say(text: str) -> AgentResponseUpdate:
    """Кусок ответа агента с текстом."""
    return AgentResponseUpdate(role="assistant", contents=[Content.from_text(text)])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=HERE / "workspace_cli")
    args = parser.parse_args()

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "test_slugify.py").write_text(TESTS, encoding="utf-8")  # контракт

    coder = OpenCodeLoopAgent(
        workdir=workdir,
        # Тот же Python, в котором запущен скрипт (в его .venv стоит pytest).
        check=[sys.executable, "-m", "pytest", "-q", "test_slugify.py"],
        name="coder",
        description="Пишет код по тестам через OpenCode с проверкой и повторами",
    )
    print(f"Рабочая папка: {workdir}\nЗадача: {TASK}")
    # Стримим, чтобы видеть попытки по мере выполнения.
    async for update in coder.run(TASK, stream=True):
        print(update.text, end="", flush=True)
    print(f"\nфайлы в рабочей папке: {sorted(p.name for p in workdir.iterdir())}")


if __name__ == "__main__":
    asyncio.run(main())
