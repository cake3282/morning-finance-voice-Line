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

LINE_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
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


def fetch_headlines(limit: int = 10, summary_max_len: int = 150) -> str:
    log("開始抓取 RSS 新聞...")
    items = []
    seen_titles = set()

    for url in RSS_FEEDS:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            feed = feedparser.parse(resp.content)
        except Exception as e:
            log(f"抓取 RSS 失敗 ({url}): {e}")
            continue

        for entry in feed.entries[:8]:
            title = clean_html_text(entry.get("title", ""))
            pub_date = clean_html_text(entry.get("published", ""))
            raw_summary = entry.get("summary") or entry.get("description") or ""
            summary = clean_html_text(raw_summary)

            if summary and summary.lower() == title.lower():
                summary = ""

            if len(summary) > summary_max_len:
                summary = summary[:summary_max_len].rstrip() + "..."

            if title and title not in seen_titles:
                seen_titles.add(title)
                line = f"- 標題：{title}"
                if pub_date:
                    line += f"（時間：{pub_date}）"
                if summary:
                    line += f"\n  摘要：{summary}"
                items.append(line)

            if len(items) >= limit:
                break
        if len(items) >= limit:
            break

    log(f"RSS 抓取完成，共 {len(items)} 則。")
    return "\n".join(items)


def clean_script_for_weekend(text: str, is_weekend: bool) -> str:
    if not is_weekend:
        return text
    cleaned = re.sub(r"(今日|今天)(台股|美股|股市)?(即將|準備)?(開盤|盤前)", "本週市場", text)
    return cleaned


def call_gemini_rest(model_name: str, prompt: str, timeout_sec: int = 45) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    params = {"key": GEMINI_API_KEY}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    resp = requests.post(url, params=params, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError(f"沒有回傳 candidates: {data}")

    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise ValueError(f"回傳文字為空: {data}")
    return text


def generate_script(news_digest: str, now_tw: datetime) -> str:
    log("開始呼叫 Gemini 生成晨報稿...")
    weekday_map = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]
    today_str = now_tw.strftime("%Y年%m月%d日")
    weekday_str = weekday_map[now_tw.weekday()]
    is_weekend = now_tw.weekday() >= 5

    if is_weekend:
        mode_instruction = (
            f"今天是 {today_str}（{weekday_str}），屬於週末休市日，台股與美股今天不開盤。"
            "請以「本週市場焦點回顧」與「下週觀察重點」為主。"
            "嚴格禁止說「今天開盤」、「盤前觀察」或把過去日期當成今天。"
        )
    else:
        mode_instruction = (
            f"今天是 {today_str}（{weekday_str}），屬於交易日早晨。"
            "請聚焦隔夜美股與科技/半導體焦點、台股盤前觀察與重要政經或匯率訊號。"
        )

    prompt = f"""
你是每天早上七點播報的財經主播。現在時間是台灣時間 {today_str}（{weekday_str}）。
{mode_instruction}

請根據以下財經新聞標題與摘要，寫一份約 550 到 750 字（朗讀約 2.5 到 3.5 分鐘）的繁體中文晨報口語稿。
要求：
1. 依據標題與摘要的事實脈絡撰寫，不要自行腦補未提到的數字或事件。
2. 專為「語音朗讀」撰寫：使用自然口語、完整句子，不要用 Markdown 星號、條列符號或難唸的縮寫堆砌；股票代號或英文專有名詞請附上好唸的中文稱呼（例如特斯拉、輝達、標普五百、那斯達克）。
3. 開頭問候早安並清楚說出今天是 {today_str} {weekday_str}，結尾提醒投資人留意風險與市場波動。

近期財經新聞與摘要：
{news_digest}
"""
    # 修正模型名稱，優先使用穩定的 2.0-flash 與 1.5-flash
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                log(f"嘗試模型 {model_name}（第 {attempt + 1} 次）...")
                raw_text = call_gemini_rest(model_name, prompt, timeout_sec=45)
                log(f"模型 {model_name} 成功產稿！")
                return clean_script_for_weekend(raw_text, is_weekend)
            except Exception as e:
                log(f"模型 {model_name} 失敗: {type(e).__name__}: {e}")
                time.sleep(5)

    raise RuntimeError("無法從 Gemini API 取得回覆，請稍後重試。")


async def _save_tts(text: str, mp3_path: str) -> None:
    communicate = edge_tts.Communicate(text=text, voice="zh-TW-HsiaoChenNeural")
    await communicate.save(mp3_path)


async def text_to_speech(text: str, mp3_path: str = "briefing.mp3") -> None:
    log("開始合成語音 (edge-tts)...")
    for attempt in range(2):
        try:
            await asyncio.wait_for(_save_tts(text, mp3_path), timeout=45)
            log("語音合成完成！")
            return
        except Exception as e:
            log(f"edge-tts 第 {attempt + 1} 次失敗或逾時: {e}")
            await asyncio.sleep(3)
    raise RuntimeError("edge-tts 語音合成逾時或失敗")


def send_line(text: str, audio_url: str = "") -> None:
    log("開始廣播至 LINE...")
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_TOKEN}",
    }

    # 精準控制為 2 則訊息：1 則文字 + 1 則語音
    messages = [{"type": "text", "text": text[:5000]}]

    if audio_url:
        messages.append({
            "type": "audio",
            "originalContentUrl": audio_url,
            "duration": 180000,
        })

    payload = {"messages": messages}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    log("LINE 訊息廣播成功！共發送 2 則訊息。")


def main() -> None:
    now_tw = datetime.now(ZoneInfo("Asia/Taipei"))
    news_digest = fetch_headlines()
    script = generate_script(news_digest, now_tw)
    
    # 存檔供除錯或備份
    with open("briefing.txt", "w", encoding="utf-8") as f:
        f.write(script)

    asyncio.run(text_to_speech(script))
    
    # 若在流程中已傳入音訊網址則直接廣播
    if AUDIO_PUBLIC_URL:
        send_line(script, AUDIO_PUBLIC_URL)
    else:
        log("提示：未設定 AUDIO_PUBLIC_URL，將在 GitHub Release 步驟後由 Workflow 進行推播。")


if __name__ == "__main__":
    main()
