"""Взаимодействие агентов, независимость сессий и асинхронность в MAF.

Примеры:
- interaction  — супервизор вызывает подагента как инструмент (as_tool);
- independence — один агент, две сессии параллельно, истории не смешиваются;
- concurrency  — n запусков одновременно на одном event loop;
- background   — долгий подагент в фоне, пока идёт диалог (заскриптованный);
- chat         — живой чат: фоновый подагент сам сообщает о конце
                 и останавливается на одобрение пользователя.

Ключ, адрес и модель берутся из minimal_agent.py.

    .venv\\Scripts\\python agents_async.py              # все проверки, кроме chat
    .venv\\Scripts\\python agents_async.py background   # только одна
    .venv\\Scripts\\python agents_async.py chat         # живой чат с фоновым аналитиком
"""

import asyncio
import sys
import time
from typing import Annotated

from agent_framework import Agent, Message, tool
from agent_framework.openai import OpenAIChatCompletionClient

from minimal_agent import BASE_URL, MODEL, OPENROUTER_API_KEY, log_tool_calls

# Один клиент на всех агентов: его можно делить между агентами и параллельными
# запусками в одном event loop. Своё у каждого запуска — сессия и сообщения.
client = OpenAIChatCompletionClient(
    model=MODEL, base_url=BASE_URL, api_key=OPENROUTER_API_KEY
)


def make_translator() -> Agent:
    """Быстрый подагент-переводчик, общий для нескольких примеров."""
    return Agent(
        client=client,
        name="translator",  # станет именем инструмента в as_tool()
        description="Переводит текст на английский",  # станет описанием инструмента
        instructions="Переведи текст на английский. Ответь только переводом.",
    )


# --- interaction: подагент как инструмент ------------------------------------
# as_tool() заворачивает агента в обычный инструмент с одним параметром task.
# Модель супервизора сама решает, когда его вызвать. Пока подагент работает,
# ход супервизора ждёт. Подагент получает только текст task: истории
# супервизора он не видит (у него новая пустая сессия).
async def interaction() -> None:
    """Супервизор вызывает подагента как инструмент."""
    supervisor = Agent(
        client=client,
        name="supervisor",
        instructions="Переводы поручай инструменту translator, затем ответь.",
        # as_tool() всегда запускает подагента со стримом: если провайдер
        # стримит плохо, подагент вернёт пустую строку, а не ошибку.
        tools=[make_translator().as_tool()],
        middleware=[log_tool_calls],  # увидим вызов translator и его ответ
    )
    result = await supervisor.run("Переведи на английский: «Счёт оплачен вчера».")
    print("ответ супервизора:", result.text)


# --- independence: изоляция сессий -------------------------------------------
# Сам агент состояния диалога не хранит: история лежит в сессии. Поэтому один
# экземпляр агента может параллельно вести сколько угодно разговоров.
async def independence() -> None:
    """Один агент, две сессии параллельно: истории не смешиваются."""
    agent = Agent(client=client, name="memo", instructions="Отвечай одной фразой.")
    anna, boris = agent.create_session(), agent.create_session()
    # Первые реплики в обе сессии отправляются одновременно.
    await asyncio.gather(
        agent.run("Меня зовут Анна.", session=anna),
        agent.run("Меня зовут Борис.", session=boris),
    )
    # Каждая сессия должна помнить только своё имя.
    answers = await asyncio.gather(
        agent.run("Как меня зовут?", session=anna),
        agent.run("Как меня зовут?", session=boris),
    )
    print("сессия Анны:", answers[0].text)
    print("сессия Бориса:", answers[1].text)


# --- concurrency: параллельные запуски ---------------------------------------
# Всё в MAF построено на asyncio: пока один запуск ждёт ответа модели по сети,
# event loop двигает остальные. Потоки и процессы для этого не нужны.
async def concurrency(n: int = 5) -> None:
    """n запусков одновременно: общее время близко к самому долгому, а не к сумме."""
    agent = Agent(client=client, name="calc", instructions="Ответь одним числом.")

    async def timed(i: int) -> float:
        """Один запуск агента; возвращает его длительность в секундах."""
        start = time.perf_counter()
        await agent.run(f"Сколько будет {i} + {i}?")
        return time.perf_counter() - start

    start = time.perf_counter()
    # gather запускает все n корутин сразу и ждёт, пока закончатся все.
    durations = await asyncio.gather(*(timed(i) for i in range(n)))
    total = time.perf_counter() - start
    # Если запуски шли параллельно, total будет близко к max(durations),
    # а не к sum(durations). Упрёмся в лимиты провайдера — разница сократится.
    print(f"{n} запусков: сумма времён {sum(durations):.1f} с, по факту {total:.1f} с")


# --- background: долгий подагент в фоне --------------------------------------
# Схема:
#   supervisor --start_analysis--> asyncio.create_task(analyst.run(...))  (фон)
#   supervisor --analysis_result--> jobs[job_id]  (ещё работает / результат)
#   supervisor --translator------> быстрый подагент, отвечает сразу
# MAF не знает, что аналитик связан с супервизором: их связывает замыкание
# (оба инструмента видят analyst и jobs). Задачи живут в памяти процесса.
async def background() -> None:
    """Долгий подагент работает в фоне, пока идёт диалог с главным и вторым."""
    started = time.perf_counter()

    def stamp(text: str) -> None:
        """Печать с отметкой времени от начала примера — видно, что шло параллельно."""
        print(f"[{time.perf_counter() - started:5.1f} с] {text}", flush=True)

    @tool
    async def deep_search(topic: Annotated[str, "Что искать"]) -> str:
        """Долгий поиск по базе (имитация)."""
        stamp(f"аналитик: начал поиск «{topic}»")
        await asyncio.sleep(20)  # async-сон не блокирует event loop
        stamp("аналитик: поиск закончен")
        return "Найдено 3 договора аренды, у двух срок истекает в ноябре."

    analyst = Agent(
        client=client,
        name="analyst",
        instructions="Сначала вызови deep_search, потом кратко перескажи найденное.",
        tools=[deep_search],
    )
    jobs: dict[str, asyncio.Task] = {}  # номер задачи -> фоновый запуск аналитика

    # Инструменты должны быть async: синхронные MAF запускает в отдельном потоке,
    # где нет event loop и create_task не сработает.
    @tool
    async def start_analysis(task: Annotated[str, "Задача для аналитика"]) -> str:
        """Запускает аналитика в фоне и сразу возвращает номер задачи."""
        job_id = f"job-{len(jobs) + 1}"
        # create_task запускает analyst.run и НЕ ждёт его: инструмент сразу
        # возвращает ответ, и ход супервизора заканчивается.
        jobs[job_id] = asyncio.create_task(analyst.run(task))
        return f"Задача {job_id} запущена"

    @tool
    async def analysis_result(job_id: Annotated[str, "Номер задачи"]) -> str:
        """Возвращает результат аналитика или сообщает, что он ещё работает."""
        job = jobs.get(job_id)
        if job is None:
            return f"Задачи {job_id} нет"
        if not job.done():
            return "Аналитик ещё работает"
        return job.result().text  # AgentResponse аналитика -> текст для модели

    supervisor = Agent(
        client=client,
        name="supervisor",
        instructions=(
            "Поиск по базе отдавай аналитику через start_analysis и не жди его. "
            "Переводы поручай translator. Результат аналитика проверяй "
            "через analysis_result."
        ),
        tools=[start_analysis, analysis_result, make_translator().as_tool()],
        middleware=[log_tool_calls],
    )
    # Одна сессия на весь диалог: супервизор помнит номер задачи между ходами.
    session = supervisor.create_session()

    async def say(text: str) -> None:
        """Один ход пользователя: вопрос супервизору и его ответ."""
        stamp(f"пользователь: {text}")
        result = await supervisor.run(text, session=session)
        stamp(f"супервизор: {result.text}")

    # Заскриптованный диалог. Аналитик стартует в первом ходе,
    # следующие три хода идут, пока он ищет.
    await say("Пусть аналитик найдёт договоры аренды с истекающим сроком.")
    await say("Переведи на английский: «Счёт оплачен вчера».")  # второй подагент
    await say("Сколько будет 17 * 3?")  # отвечает сам супервизор
    await say("Аналитик уже закончил?")  # ожидаемо: «ещё работает»
    await asyncio.gather(*jobs.values())  # дождаться долгого подагента
    await say("Что нашёл аналитик?")  # теперь analysis_result отдаст результат


# --- chat: живой чат, уведомление о конце и одобрение ------------------------
# Аналитик и цикл чата общаются через очередь events. В неё приходит:
#   ("approval", request, future) — аналитику нужно одобрение записи;
#   ("done", text)                — аналитик закончил, text — его итог.
# Цикл чата одновременно ждёт строку из консоли и событие из очереди,
# поэтому одобрение «перебивает» чат, а конец работы сообщается без вопроса.
async def chat() -> None:
    """Живой чат: аналитик в фоне сам сообщает о конце и просит одобрение записи."""
    events: asyncio.Queue = asyncio.Queue()  # события от аналитика для чата

    @tool
    async def deep_search(topic: Annotated[str, "Что искать"]) -> str:
        """Долгий поиск по базе (имитация)."""
        print(f"\n[аналитик] ищу «{topic}»…", flush=True)
        await asyncio.sleep(15)
        return "Найдено 3 договора аренды, у двух срок истекает в ноябре."

    # approval_mode="always_require": MAF не выполняет инструмент сам,
    # а возвращает запрос на одобрение в result.user_input_requests.
    @tool(approval_mode="always_require")
    async def save_report(text: Annotated[str, "Текст отчёта"]) -> str:
        """Сохраняет отчёт в 1С (имитация записи)."""
        return "Отчёт сохранён"

    analyst = Agent(
        client=client,
        name="analyst",
        instructions=(
            "Вызови deep_search, затем сохрани краткий отчёт через save_report "
            "и перескажи его."
        ),
        tools=[deep_search, save_report],
    )

    async def run_analyst(task: str) -> None:
        """Фоновая работа аналитика: поиск, одобрение записи, итог в очередь."""
        # С версии 1.20 ответ на одобрение принимается только в той же сессии,
        # где возник запрос. Без сессии MAF молча вернёт пустой ответ.
        session = analyst.create_session()
        result = await analyst.run(task, session=session)
        # Запуск остановился на инструменте с одобрением — спрашиваем человека.
        while result.user_input_requests:
            answers = []
            for request in result.user_input_requests:
                # Future — «обещание» ответа: цикл чата заполнит его,
                # когда пользователь напишет да или нет.
                decision = asyncio.get_running_loop().create_future()
                await events.put(("approval", request, decision))
                approved = await decision  # аналитик спит, пока нет ответа
                answers.append(request.to_function_approval_response(approved=approved))
            # Отправляем решения в ту же сессию: при «да» MAF выполнит
            # инструмент, при «нет» модель получит отказ — и запуск продолжится.
            result = await analyst.run(Message("user", answers), session=session)
        await events.put(("done", result.text))  # сообщить чату, что закончили

    @tool
    async def start_analysis(task: Annotated[str, "Задача для аналитика"]) -> str:
        """Запускает аналитика в фоне; он сам сообщит о результате."""
        asyncio.create_task(run_analyst(task))  # не ждём: аналитик работает в фоне
        return "Аналитик запущен и сам сообщит о результате"

    supervisor = Agent(
        client=client,
        name="supervisor",
        instructions=(
            "Поиск по базе отдавай аналитику через start_analysis и не жди его. "
            "Переводы поручай translator. Сообщения с пометкой [аналитик] "
            "перескажи пользователю."
        ),
        tools=[start_analysis, make_translator().as_tool()],
        middleware=[log_tool_calls],
    )
    session = supervisor.create_session()

    async def tell(text: str) -> None:
        """Отправить супервизору сообщение (от пользователя или от аналитика)."""
        result = await supervisor.run(text, session=session)
        print(f"супервизор: {result.text}", flush=True)

    def read_line() -> asyncio.Task:
        """Чтение строки из консоли в отдельном потоке, оформленное как задача."""
        return asyncio.create_task(asyncio.to_thread(input, "> "))

    print("Пустая строка — выход.")
    # line — ожидание следующей строки; approval — future одобрения, которое
    # сейчас ждёт ответа пользователя (None, если такого нет).
    line, approval = read_line(), None
    while True:
        event = asyncio.create_task(events.get())
        # Ждём, что наступит раньше: строка от пользователя или событие аналитика.
        await asyncio.wait({line, event}, return_when=asyncio.FIRST_COMPLETED)
        if line.done():  # строку, набранную раньше события, обрабатываем первой
            text = line.result().strip()
            if approval is not None:  # идёт одобрение: строка — это да или нет
                approval.set_result(text.lower() in {"да", "y", "yes"})
                approval = None
            elif not text:  # пустая строка — выход
                event.cancel()
                return
            else:  # обычная реплика — супервизору
                await tell(text)
            line = read_line()  # ждать следующую строку
        if not event.done():  # событие могло прийти, пока отвечал супервизор
            event.cancel()  # снимаем ожидание; элемент очереди не теряется
            continue
        kind, *payload = event.result()
        if kind == "approval":  # перебивает чат: следующая строка — ответ
            request, approval = payload
            call = request.function_call  # какой инструмент и с какими аргументами
            print(f"\n[одобрение] {call.name}({call.arguments}) — да/нет?", flush=True)
        else:  # аналитик закончил: супервизор узнаёт сам, без вопроса
            await tell(f"[аналитик] Работа закончена: {payload[0]}")


# Проверки, которые запускаются без аргументов; chat — только явно, он интерактивный.
CHECKS = {
    "interaction": interaction,
    "independence": independence,
    "concurrency": concurrency,
    "background": background,
}


async def main() -> None:
    if sys.argv[1:] == ["chat"]:
        await chat()
        return
    for name in sys.argv[1:] or CHECKS:  # имена из аргументов или все по порядку
        print(f"\n=== {name} ===")
        await CHECKS[name]()


if __name__ == "__main__":
    asyncio.run(main())
