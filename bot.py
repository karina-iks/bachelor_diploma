"""
Telegram-бот поддержки Золотого Яблока
RAG: BeautifulSoup → bge-m3 embeddings → KNN → YandexGPT
"""

import asyncio
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import requests
import bs4
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ─── Конфиг ───────────────────────────────────────────────────────────────────

load_dotenv()

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
YANDEX_OAUTH     = os.getenv("YANDEX_OAUTH_TOKEN")   # для авто-рефреша IAM
YANDEX_IAM       = os.getenv("YANDEX_IAM_TOKEN")     # или задать вручную
YANDEX_FOLDER_ID = os.environ["YANDEX_FOLDER_ID"]
FAQ_HTML_PATH    = os.getenv("FAQ_HTML_PATH", "faq.html")
FAQ_URL          = os.getenv("FAQ_URL", "https://goldapple.ru/faq")
TOP_K            = int(os.getenv("TOP_K", "3"))
TEMPERATURE      = float(os.getenv("TEMPERATURE", "0.4"))
MAX_TOKENS       = int(os.getenv("MAX_TOKENS", "2000"))
MODEL_NAME       = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
YANDEX_GPT_MODEL = os.getenv("YANDEX_GPT_MODEL", "yandexgpt/rc")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("goldapple_bot")

logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = SYSTEM_PROMPT = """
Ты — виртуальный ассистент службы поддержки маркетплейса Золотое Яблоко.

## Твоя задача
Отвечать на вопросы пользователей кратко, по делу и с заботой, опираясь на предоставленные документы из базы знаний. 
Если ответ не содержится в документах — напиши: «Я не нашла информации по этому вопросу, но ты можешь обратиться в нашу поддержку — мы обязательно разберёмся 💛»
И добавь информацию про связь с реальной живой поддержкой, вот информация: Вы можете самостоятельно обратиться в call-центр. Пожалуйста, напишите в онлайн-чат WhatsApp и Telegram или позвоните в call-центр по телефону 8 800 770 70 21. Наши специалисты на связи 24/7.

## Tone of voice
- Тёплый, дружелюбный, немного неформальный — как подруга, которая хорошо знает магазин
- Обращение на «ты»
- Не пиши приветствие дважды!
- Без канцелярита: не «в случае возникновения вопросов», а «если что-то непонятно»
- Без избыточных извинений и шаблонных фраз вроде «Благодарим за обращение»
- В конце каждого ответа — 1–2 уместных смайла из набора: 💛 🛍️ 🌸 ✨ 💅 🪞 💄 💝 🎀
  Смайл должен подходить по настроению: на нейтральный вопрос — 💛 или ✨, на радостный — 🎀 или 💝, не лепи смайл туда, где человек расстроен или зол
- Расскажи про возможность начать диалог сначала через вызов функции /new и получить информацию через вызов функции /help

## Обработка негатива
Если пользователь злится, грубит или жалуется:
1. Не оправдывайся и не спорь
2. Сначала признай эмоцию: «Понимаю, это неприятно» / «Это правда обидная ситуация»
3. Затем — конкретная помощь из документов или направление в поддержку
4. Тон остаётся спокойным и участливым, без формальных отписок
5. Смайл в таком ответе — не более одного и только в самом конце, если уместен

Пример:
— «Ваш сайт полное дерьмо, заказ потерялся!!!»
— «Понимаю, это очень неприятно — особенно когда уже ждёшь заказ. Давай разберёмся: [конкретный ответ из документов]. Если не поможет — напиши в чат поддержки, там помогут быстрее всего 💛»

## Обработка вопросов про конкурентов
Если пользователь упоминает другие магазины (Летуаль, Рив Гош, Иль де Ботэ, Wildberries, Ozon и др.):
- Не критикуй конкурентов и не сравнивай цены/условия
- Не говори ничего плохого о других брендах
- Мягко переводи фокус на то, что можешь предложить Золотое Яблоко
- Если вопрос про баллы/карту лояльности конкурента — объясни, что можешь рассказать только про программу лояльности Золотого Яблока

Пример:
— «А в Летуаль доставка бесплатная, почему у вас нет?»
— «Могу рассказать про условия доставки в Золотом Яблоке: [ответ из документов] ✨»
"""

# ─── Утилиты ──────────────────────────────────────────────────────────────────

def nw(text: str) -> str:
    """Нормализация пробелов."""
    return re.sub(r"\s+", " ", text).strip()


# ─── IAM-токен ────────────────────────────────────────────────────────────────

class IAMTokenManager:
    """
    Хранит актуальный IAM-токен и обновляет его каждые 11 часов.
    Если задан YANDEX_IAM_TOKEN — использует его без обновления.
    Если задан YANDEX_OAUTH_TOKEN — получает и обновляет токен автоматически.
    """

    IAM_TTL = 11 * 3600  # 11 часов

    def __init__(self):
        self._token: str | None = YANDEX_IAM
        self._updated_at: float = 0.0

        if not self._token and not YANDEX_OAUTH:
            raise EnvironmentError(
                "Укажите YANDEX_IAM_TOKEN или YANDEX_OAUTH_TOKEN в .env"
            )

    def get(self) -> str:
        if YANDEX_OAUTH and (time.time() - self._updated_at > self.IAM_TTL):
            self._refresh()
        return self._token  # type: ignore[return-value]

    def _refresh(self):
        log.info("Обновление IAM-токена...")
        r = requests.post(
            "https://iam.api.cloud.yandex.net/iam/v1/tokens",
            json={"yandexPassportOauthToken": YANDEX_OAUTH},
            timeout=10,
        )
        r.raise_for_status()
        self._token = r.json()["iamToken"]
        self._updated_at = time.time()
        log.info("IAM-токен обновлён.")


iam = IAMTokenManager()


# ─── Парсинг FAQ ──────────────────────────────────────────────────────────────

def parse_faq(html: str) -> list[dict]:
    soup = bs4.BeautifulSoup(html, "html.parser")
    documents = []
    for i, section_tag in enumerate(soup.findAll("h2")):
        section = nw(section_tag.text)
        for j, q_tag in enumerate(section_tag.parent.findAll("h3")):
            answer_div = q_tag.parent.findChild("div")
            if not answer_div:
                continue
            answer = nw(answer_div.text)
            question = nw(q_tag.text)
            if not question.endswith("?"):
                question += "?"
            doc_text = f"## {section}: {question}\n\n{answer}"
            documents.append(
                {"id": f"{i}-{j}", "text": doc_text, "theme": section}
            )
    log.info("Распознано %d документов из FAQ.", len(documents))
    return documents


def load_faq_html() -> str:
    path = Path(FAQ_HTML_PATH)
    if path.exists():
        log.info("Читаем FAQ из файла: %s", path)
        return path.read_text(encoding="utf-8")
    log.info("Файл не найден — скачиваем FAQ с %s", FAQ_URL)
    r = requests.get(FAQ_URL, timeout=30)
    r.raise_for_status()
    return r.text


# ─── RAG-движок ───────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        log.info("Загрузка эмбеддинг-модели %s ...", MODEL_NAME)
        self.embedder = SentenceTransformer(MODEL_NAME)

        html = load_faq_html()
        self.documents = parse_faq(html)
        if not self.documents:
            raise RuntimeError("FAQ пуст — проверьте HTML-файл или URL.")

        log.info("Вычисление эмбеддингов для %d документов...", len(self.documents))
        texts = [d["text"] for d in self.documents]
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=True,
        )

        self.knn = NearestNeighbors(n_neighbors=TOP_K, metric="cosine")
        self.knn.fit(embeddings)
        self.embeddings = embeddings
        log.info("RAG-движок готов.")

    def retrieve(self, query: str) -> list[str]:
        q_emb = self.embedder.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )
        _, indices = self.knn.kneighbors(q_emb.reshape(1, -1))
        return [self.documents[i]["text"] for i in indices.flatten()]


# ─── YandexGPT ────────────────────────────────────────────────────────────────

YANDEX_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


def yandex_gpt(query: str, docs: list[str], history: list[dict]) -> str:
    """
    history — список {"role": "user"|"assistant", "text": "..."} прошлых реплик.
    Текущий запрос добавляется последним вместе с найденными документами.
    """
    docs_text = "\n\n".join(docs)
    messages = [{"role": "system", "text": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "text": (
        f"Запрос пользователя:\n{query}\n\n"
        f"Найденные документы:\n{docs_text}"
    )})
    payload = {
        "modelUri":          f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_GPT_MODEL}",
        "completionOptions": {
            "stream":      False,
            "temperature": TEMPERATURE,
            "maxTokens":   str(MAX_TOKENS),
        },
        "messages": messages,
    }
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {iam.get()}",
    }
    r = requests.post(YANDEX_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["result"]["alternatives"][0]["message"]["text"]


# ─── Telegram-хэндлеры ────────────────────────────────────────────────────────

rag: RAGEngine  # инициализируется в main()

# История диалога: { user_id: [{"role": ..., "text": ...}, ...] }
# Хранится в памяти — при перезапуске бота сбрасывается для всех
history_store: dict[int, list[dict]] = {}

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))  # пар user/assistant


def get_history(user_id: int) -> list[dict]:
    return history_store.setdefault(user_id, [])


def push_history(user_id: int, query: str, answer: str):
    h = get_history(user_id)
    h.append({"role": "user",      "text": query})
    h.append({"role": "assistant", "text": answer})
    # Ограничиваем глубину: оставляем последние MAX_HISTORY_TURNS пар
    if len(h) > MAX_HISTORY_TURNS * 2:
        history_store[user_id] = h[-(MAX_HISTORY_TURNS * 2):]


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    history_store.pop(update.effective_user.id, None)
    await update.message.reply_text(
        "👋 Привет! Я бот поддержки Золотого Яблока.\n"
        "Задайте любой вопрос — постараюсь помочь!\n\n"
        "Команды:\n"
        "/new — начать новое обращение (сбросить историю)\n"
        "/help — примеры вопросов"
    )


async def cmd_new(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    history_store.pop(update.effective_user.id, None)
    await update.message.reply_text(
        "🔄 История диалога сброшена. Можете задать новый вопрос!"
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Просто напишите вопрос, например:\n"
        "• Как отследить мой заказ?\n"
        "• Можно ли вернуть товар?\n"
        "• Сколько стоит доставка?\n\n"
        "/new — начать новое обращение"
    )


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        return

    user_id = update.effective_user.id
    log.info("Запрос [%s]: %s", user_id, query[:80])

    await ctx.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    history = get_history(user_id)

    try:
        docs = await asyncio.get_event_loop().run_in_executor(
            None, rag.retrieve, query
        )
        answer = await asyncio.get_event_loop().run_in_executor(
            None, yandex_gpt, query, docs, history
        )
        push_history(user_id, query, answer)
    except requests.HTTPError as e:
        log.error("YandexGPT HTTP ошибка: %s", e)
        answer = "⚠️ Сервис временно недоступен. Попробуйте позже."
    except Exception as e:
        log.exception("Неожиданная ошибка: %s", e)
        answer = "⚠️ Произошла ошибка. Пожалуйста, попробуйте позже."

    await update.message.reply_text(answer)


# ─── Точка входа ──────────────────────────────────────────────────────────────

def main():
    global rag

    log.info("Инициализация RAG-движка...")
    rag = RAGEngine()

    log.info("Запуск Telegram-бота...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("new",   cmd_new))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Бот запущен. Ожидание сообщений...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
