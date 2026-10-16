"""Минимальный запуск агента на Microsoft Agent Framework через OpenRouter.

Агент в MAF = клиент модели + инструкции + инструменты + middleware.
Файл показывает:
- инструмент из обычной функции (@tool);
- function-middleware: печать каждого вызова инструмента и его результата;
- chat-middleware: печать всего, что уходит в модель на каждом запросе;
- два режима запуска: один вопрос без стрима и диалог со стримом и сессией.

    .venv\\Scripts\\python minimal_agent.py                  # диалог
    .venv\\Scripts\\python minimal_agent.py "Который час?"   # один вопрос без стрима
"""

import asyncio
import sys
from datetime import datetime
import json

from agent_framework import Agent, FunctionInvocationContext, tool, ChatContext
from agent_framework.openai import OpenAIChatCompletionClient

OPENROUTER_API_KEY = "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"  # вставь свой ключ
BASE_URL = "https://openrouter.ai/api/v1"  # любой OpenAI-совместимый сервер
# Нужна модель с поддержкой tools: https://openrouter.ai/models?supported_parameters=tools
MODEL = "xiaomi/mimo-v2.6-flash"


# --- Инструмент -------------------------------------------------------------
# @tool превращает обычную функцию в инструмент для модели:
# имя функции — имя инструмента, docstring — его описание,
# аннотации параметров — JSON-схема аргументов (здесь аргументов нет).
# Синхронную функцию MAF выполняет в отдельном потоке, async — в event loop.
@tool
def current_time() -> str:
    """Возвращает текущие дату и время с часовым поясом."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --- Function-middleware: вокруг каждого вызова инструмента -------------------
# Вид middleware MAF определяет по аннотации первого параметра:
# FunctionInvocationContext — вызов инструмента, ChatContext — запрос к модели.
async def log_tool_calls(context: FunctionInvocationContext, call_next) -> None:
    """Печатает каждый вызов инструмента и его результат."""
    # До call_next инструмент ещё не выполнен: видны имя и аргументы от модели.
    # Здесь же можно поменять context.arguments или не вызывать call_next —
    # тогда инструмент не выполнится (так делаются гейты).
    args = dict(context.arguments or {})
    print(f"\n[инструмент] {context.function.name}({args})", flush=True)
    await call_next()  # выполнить инструмент (и следующие middleware, если есть)
    # После call_next в context.result лежит то, что получит модель.
    # Результат приходит списком Content; у текстовых частей есть .text.
    result = context.result
    if isinstance(result, list):
        result = " ".join(getattr(part, "text", None) or str(part) for part in result)
    print(f"[результат] {result}", flush=True)


# --- Chat-middleware: вокруг каждого запроса к модели -------------------------
# Срабатывает на КАЖДЫЙ запрос к модели, а не на каждый вопрос пользователя:
# вопрос с инструментом = 2 запроса (вопрос -> вызов инструмента -> ответ).
async def log_llm_input(context: ChatContext, call_next) -> None:
    """Печатает всё, что уходит в модель на каждом её вызове."""
    print("\n===== в модель =====")
    # Системного промпта нет в messages: он лежит в options["instructions"],
    # и клиент добавляет его первым сообщением уже после всех middleware.
    print("instructions:", (context.options or {}).get("instructions"))
    # Полный список: история сессии + новый вопрос + вызовы и результаты
    # инструментов текущего хода. to_dict() — сериализация Message в словарь.
    for message in context.messages:
        print(json.dumps(message.to_dict(), ensure_ascii=False))
    await call_next()  # отправить запрос модели


# --- Сборка агента -------------------------------------------------------------
def build_agent() -> Agent:
    # Chat Completions API: его понимают vLLM, Ollama, OpenRouter.
    # OpenAIChatClient (Responses API) для них не подходит: он рассчитывает,
    # что историю хранит сервер. api_key обязателен, иначе клиент
    # уходит в маршрут Azure и падает с ошибкой про Azure.
    client = OpenAIChatCompletionClient(
        model=MODEL,
        base_url=BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    return Agent(
        client=client,
        name="minimal_agent",  # видно в трассах и при использовании как подагента
        # instructions = системный промпт агента
        instructions=(
            "Ты помощник. Отвечай кратко, по-русски. "
            "Про дату и время спрашивай инструмент current_time."
        ),
        tools=[current_time],  # цикл «модель -> инструмент -> модель» встроен в MAF
        middleware=[log_tool_calls, log_llm_input],
    )


# --- Режим 1: один вопрос без стрима ------------------------------------------
async def ask_once(agent: Agent, question: str) -> None:
    # Без session агент не помнит прошлых вопросов: каждый run — с нуля.
    # await вернёт полный AgentResponse, когда модель закончит
    # (включая все вызовы инструментов).
    result = await agent.run(question)
    print(result.text)  # текст последнего ответа модели
    print_usage(result.usage_details)


# --- Режим 2: диалог со стримом и памятью --------------------------------------
async def chat(agent: Agent) -> None:
    # История диалога живёт в сессии: MAF сам подключает InMemoryHistoryProvider
    # и хранит сообщения в session.state["messages"].
    # session.to_dict() / AgentSession.from_dict() — сохранить и восстановить.
    session = agent.create_session()
    print("Пустая строка или /exit — выход.")
    while True:
        # input() блокирует поток, поэтому уносим его в отдельный поток:
        # event loop остаётся свободным для других задач.
        question = (await asyncio.to_thread(input, "\n> ")).strip()
        if question in {"", "/exit"}:
            return
        # stream=True: run возвращает ResponseStream, его читают через async for.
        stream = agent.run(question, session=session, stream=True)
        async for update in stream:  # AgentResponseUpdate — кусок ответа
            if update.text:
                print(update.text, end="", flush=True)
        # Итоговый AgentResponse собирается из кусков; usage приходит в конце.
        final = await stream.get_final_response()
        print()
        print_usage(final.usage_details)


def print_usage(usage) -> None:
    # UsageDetails — словарь; None, если провайдер не вернул расход токенов.
    if usage:
        tokens_in = usage.get("input_token_count")
        tokens_out = usage.get("output_token_count")
        print(f"[токены] вход {tokens_in}, выход {tokens_out}")


async def main() -> None:
    agent = build_agent()
    if len(sys.argv) > 1:  # есть аргумент — задать один вопрос
        await ask_once(agent, " ".join(sys.argv[1:]))
    else:  # нет аргументов — диалог
        await chat(agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):  # Ctrl+C или конец ввода — тихий выход
        pass
