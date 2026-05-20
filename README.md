# Бот поддержки Золотого Яблока 🛍️

RAG-ассистент на базе **bge-m3** + **YandexGPT** в виде Telegram-бота.

---

## Структура файлов

```
bachelor_diploma/
├── bot.py            # Весь код бота
├── requirements.txt
├── .env              # Конфиг
├── faq.html          # Сохранённый HTML со страницы goldapple.ru/faq
├── assistant.ipynb   # Расчет и построение бенчмарка
├── index.html        # Калькулятор для расчета Cost per Ticket
└── README.md
```

---

## Как запустить - 1. Подготовить окружение на сервере

```bash
# Клонируйте или скопируйте папку goldapple_bot на сервер
cd goldapple_bot

# Python 3.10+
python3 --version

# Создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установите зависимости
pip install -r requirements.txt
```

> ⚠️ Первый запуск скачает модель **BAAI/bge-m3** (~570 МБ).  
> Нужно ~2 ГБ RAM для загрузки модели + ~500 МБ для эмбеддингов FAQ.

---

## 2. Создать/заполнить `.env`

```bash
nano .env   # или vim .env
```

Обязательно заполните:

| Переменная | Что вставить |
|---|---|
| `TELEGRAM_TOKEN` | Токен от BotFather |
| `YANDEX_FOLDER_ID` | ID каталога из Yandex Cloud |
| `YANDEX_OAUTH_TOKEN` | OAuth-токен Яндекса |

Положите `faq.html` в папку рядом с `bot.py`, или укажите `FAQ_HTML_PATH` — иначе бот попытается скачать страницу сам (может не работать из-за защиты сайта).

---

## 3. Запустить бота

### Разово (для проверки)
```bash
source venv/bin/activate
python bot.py
```

### Через systemd (автозапуск и перезапуск при падении)

Создайте файл сервиса:

```bash
sudo nano /etc/systemd/system/goldapple_bot.service
```

Содержимое:

```ini
[Unit]
Description=Gold Apple Support Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/goldapple_bot
ExecStart=/home/ubuntu/goldapple_bot/venv/bin/python bot.py
Restart=on-failure
RestartSec=10
EnvironmentFile=/home/ubuntu/goldapple_bot/.env

[Install]
WantedBy=multi-user.target
```

Активируйте и запустите:

```bash
sudo systemctl daemon-reload
sudo systemctl enable goldapple_bot
sudo systemctl start goldapple_bot

# Проверить статус
sudo systemctl status goldapple_bot

# Смотреть логи в реальном времени
sudo journalctl -u goldapple_bot -f
```

---

## Переменные окружения (полный список)

| Переменная | Обязательна | По умолчанию | Описание |
|---|---|---|---|
| `TELEGRAM_TOKEN` | ✅ | — | Токен бота |
| `YANDEX_FOLDER_ID` | ✅ | — | ID каталога YC |
| `YANDEX_OAUTH_TOKEN` | ✅* | — | OAuth-токен (*или IAM) |
| `YANDEX_IAM_TOKEN` | ✅* | — | IAM-токен (*или OAuth) |
| `FAQ_HTML_PATH` | | `faq.html` | Путь к HTML-файлу FAQ |
| `FAQ_URL` | | `https://goldapple.ru/faq` | URL для скачивания FAQ |
| `TOP_K` | | `3` | Кол-во документов в контексте |
| `TEMPERATURE` | | `0.4` | Температура генерации |
| `MAX_TOKENS` | | `2000` | Макс. токенов в ответе |
| `YANDEX_GPT_MODEL` | | `yandexgpt/rc` | Модель YandexGPT |
| `EMBED_MODEL` | | `BAAI/bge-m3` | Эмбеддинг-модель |

---

## Частые проблемы

**`401 Unauthorized` от YandexGPT**  
→ IAM-токен протух. Перезапустите бота, или убедитесь что задан `YANDEX_OAUTH_TOKEN` (тогда токен обновляется сам).

**`FAQ пуст — проверьте HTML-файл`**  
→ Убедитесь что `faq.html` — полная страница с загруженным JavaScript (нужен rendered HTML, не просто `curl`). Сохраните страницу через браузер: `Ctrl+S` → «Веб-страница, полностью».

**Модель качается слишком долго**  
→ Замените в `.env`: `EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~120 МБ, чуть хуже качество).

**Бот не отвечает после деплоя**  
→ Проверьте логи: `sudo journalctl -u goldapple_bot -n 50`
