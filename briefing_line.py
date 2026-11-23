import asyncio
import html
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import edge_tts
import feedparser
import requests
from google import genai

LINE_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_USER_ID = os.environ["LINE_USER_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
AUDIO_PUBLIC_URL = os.environ.get("AUDIO_PUBLIC_URL", "")

RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,NVDA,TSLA&region=US&lang=en-US",
    "https://news.google.com/rss/search?q=%E5%8F%B0%E8%82%A1+%E7%BE%8E%E8%82%A1+%E5%8D%8A%E5%B0%8E%E9%AB%94+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def log(msg: str) -> None:
    now = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def clean_html_text(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_headlines(limit: int = 10, summary_max_len:
