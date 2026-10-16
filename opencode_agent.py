"""Главный агент поручает задачу подагенту, который пишет код через локальный OpenCode.

Что показывает:
- свой подагент: класс OpenCodeAgent наследуется от BaseAgent и внутри запускает
  OpenCode CLI (`opencode run`) как отдельный процесс;
- рабочая папка: OpenCode видит и меняет только её (--dir + запрет внешних путей);
- прерывания: перед запуском OpenCode главный агент спрашивает одобрение
  у пользователя; долгий запуск обрывается по таймауту, по Ctrl+C процесс
  OpenCode убивается.

Модели берутся через OpenRouter, ключ и модель главного агента — из minimal_agent.py.
OpenCode CLI лежит в bin/opencode.exe (официальный релиз v1.19.1).

    .venv\\Scripts\\python opencode_agent.py
    .venv\\Scripts\\python opencode_agent.py --workdir D:\\tmp\\ws "Напиши hello.py"
"""

import argparse
import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agent_framework import (
    Agent,
    AgentResponse,
    AgentResponseUpdate,
    BaseAgent,
    Content,
    Message,
    ResponseStream,
    normalize_messages,
)
from agent_framework.openai import OpenAIChatCompletionClient

from minimal_agent import BASE_URL, MODEL, OPENROUTER_API_KEY, log_tool_calls

HERE = Path(__file__).parent
OPENCODE = HERE / "bin" / "opencode.exe"
# Модель для OpenCode в его формате provider/model.
# Список доступных: bin\opencode.exe models openrouter
OPENCODE_MODEL = "openrouter/qwen/qwen3.8-27b"
DEFAULT_TASK = (
    "Напиши в рабочей папке fizzbuzz.py с функцией fizzbuzz(n), которая возвращает "
    "список строк для чисел от 1 до n, и тесты test_fizzbuzz.py на pytest."
)


class OpenCodeAgent(BaseAgent):
    """Подагент-программист: решает задачу через OpenCode в заданной папке."""

    def __init__(
        self,
        *,
        workdir: Path,
        model: str = OPENCODE_MODEL,
        timeout: float = 600,
        allow_bash: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # name, description и прочее — как у любого агента
        self.workdir = workdir.resolve()
        self.model = model
        self.timeout = timeout
        # Конфиг OpenCode на этот запуск; глобальный конфиг пользователя не трогаем.
        self.config = {
            "permission": {
                "edit": "allow",  # писать файлы можно...
                "external_directory": "deny",  # ...но только внутри рабочей папки
                # bash — настоящая оболочка, рабочей папкой она надёжно не ограничена,
                # поэтому по умолчанию выключена.
                "bash": "allow" if allow_bash else "deny",
                "webfetch": "deny",
            },
            # Ключ задаём в конфиге: он важнее ключа, сохранённого в auth.json
            # десктопного OpenCode (переменная окружения его не перебивает).
            "provider": {"openrouter": {"options": {"apiKey": OPENROUTER_API_KEY}}},
            "autoupdate": False,
            "share": "disabled",
        }

    # MAF вызывает run(...) и с stream=False, и с stream=True (as_tool — всегда со
    # стримом). Со стримом нужно вернуть ResponseStream: у него есть
    # get_final_response(), которого нет у голого async-генератора.
    def run(  # type: ignore[override]
        self,
        messages: str | Message | list[str] | list[Message] | None = None,
        *,
        stream: bool = False,
        session: Any = None,  # OpenCode держит свою историю, сессию MAF не используем
        **kwargs: Any,
    ) -> "ResponseStream[AgentResponseUpdate, AgentResponse] | Any":
        task = self._task_text(messages)
        if stream:
            # finalizer собирает итоговый AgentResponse из всех кусков
            return ResponseStream(
                self._updates(task), finalizer=AgentResponse.from_updates
            )
        return self._run_once(task)

    async def _run_once(self, task: str) -> AgentResponse:
        updates = [update async for update in self._updates(task)]
        return AgentResponse.from_updates(updates)

    @staticmethod
    def _task_text(messages: Any) -> str:
        """Текст задачи — последнее сообщение (as_tool передаёт одну строку task)."""
        normalized = normalize_messages(messages)
        return normalized[-1].text if normalized else ""

    async def _updates(self, task: str) -> AsyncIterator[AgentResponseUpdate]:
        """Запускает OpenCode и превращает его JSON-события в куски ответа агента."""
        self.workdir.mkdir(parents=True, exist_ok=True)

        def say(text: str) -> AgentResponseUpdate:
            """Кусок ответа агента с текстом."""
            content = Content.from_text(text)
            return AgentResponseUpdate(role="assistant", contents=[content])

        process = await asyncio.create_subprocess_exec(
            str(OPENCODE), "run",
            "--pure",  # без внешних плагинов
            "--format", "json",  # события построчно в JSON
            "--dir", str(self.workdir),  # папка проекта: её OpenCode видит и меняет
            "-m", self.model,
            task,
            env={**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps(self.config)},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,  # событие записи файла содержит весь файл
        )
        stderr = asyncio.create_task(process.stderr.read())  # читаем, чтобы не забился
        tokens, cost = 0, 0.0
        try:
            async with asyncio.timeout(self.timeout):  # прерывание по таймауту
                while line := await process.stdout.readline():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    part = event.get("part") or {}
                    kind = event.get("type")
                    if kind == "tool_use":  # OpenCode вызвал свой инструмент
                        args = (part.get("state") or {}).get("input") or {}
                        target = args.get("filePath") or args.get("command") or ""
                        name = Path(target).name if args.get("filePath") else target
                        yield say(f"[opencode] {part.get('tool')} {name}\n")
                    elif kind == "text":  # ответ модели OpenCode
                        yield say(part.get("text", "").strip() + "\n")
                    elif kind == "step_finish":  # шаг закончен: токены и цена
                        tokens += (part.get("tokens") or {}).get("total", 0)
                        cost += part.get("cost") or 0.0
                    elif kind == "error":
                        error = event.get("error") or {}
                        message = (error.get("data") or {}).get("message") or error
                        yield say(f"[opencode] ошибка: {message}\n")
            await process.wait()
            yield say(f"[opencode] готово: {tokens} токенов, ${cost:.4f}\n")
        except TimeoutError:
            yield say(f"[opencode] прерван по таймауту {self.timeout:.0f} с\n")
        finally:
            # Сюда попадаем и при отмене (Ctrl+C, cancel задачи): процесс не бросаем.
            if process.returncode is None:
                process.kill()
                await process.wait()
            stderr.cancel()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=HERE / "workspace",
                        help="папка, которую видит и меняет OpenCode")
    parser.add_argument("task", nargs="*", help="задача для главного агента")
    args = parser.parse_args()
    task = " ".join(args.task) or DEFAULT_TASK

    coder = OpenCodeAgent(
        workdir=args.workdir,
        name="coder",
        description="Пишет и правит код в рабочей папке через OpenCode",
    )
    supervisor = Agent(
        client=OpenAIChatCompletionClient(
            model=MODEL, base_url=BASE_URL, api_key=OPENROUTER_API_KEY
        ),
        name="supervisor",
        instructions=(
            "Задачи по написанию кода поручай инструменту coder, передавая задачу "
            "целиком. Потом кратко перескажи пользователю, что сделано."
        ),
        tools=[
            coder.as_tool(
                arg_name="task",
                # Прерывание: перед запуском OpenCode MAF спросит пользователя.
                approval_mode="always_require",
                # Действия OpenCode печатаются сразу, пока он работает.
                stream_callback=lambda update: print(update.text, end="", flush=True),
            )
        ],
        middleware=[log_tool_calls],
    )

    print(f"Рабочая папка: {coder.workdir}\nЗадача: {task}")
    session = supervisor.create_session()  # одобрение принимается в той же сессии
    result = await supervisor.run(task, session=session)
    while result.user_input_requests:  # главный агент ждёт решения пользователя
        answers = []
        for request in result.user_input_requests:
            call = request.function_call
            print(f"\n[одобрение] {call.name}({call.arguments}) — да/нет?")
            reply = (await asyncio.to_thread(input, "> ")).strip().lower()
            approved = reply in {"да", "y", "yes"}
            answers.append(request.to_function_approval_response(approved=approved))
        result = await supervisor.run(Message("user", answers), session=session)

    print(f"\nсупервизор: {result.text}")
    files = sorted(p.name for p in coder.workdir.iterdir())
    print(f"файлы в рабочей папке: {files}")


if __name__ == "__main__":
    asyncio.run(main())
