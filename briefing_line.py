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

US_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^IXIC,NVDA,TSLA,AAPL,MSFT,AVGO,MU,MRVL,AMD&region=US&lang=en-US",
]

TW_RSS_FEEDS = [
    "https://news.google.com/rss/search?q=%E5%8F%B0%E8%82%A1+%E5%8D%8A%E5%B0%8E%E9%AB%94+%E6%A9%9F%E5%99%A8%E4%BA%BA+%E7%B6%93%E6%BF%9F+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
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


def fetch_category_headlines(feeds: list, limit: int = 5, summary_max_len: int = 150) -> list:
    items = []
    seen_titles = set()

    for url in feeds:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            feed = feedparser.parse(resp.content)
        except Exception as e:
            log(f"抓取 RSS 失敗 ({url}): {e}")
            continue

        for entry in feed.entries:
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
    return items


def fetch_balanced_headlines() -> str:
    log("開始抓取 RSS 新聞（美股 3 則、台股 7 則）...")
    us_items = fetch_category_headlines(US_RSS_FEEDS, limit=3)
    tw_items = fetch_category_headlines(TW_RSS_FEEDS, limit=7)

    digest = "【美股重點新聞】\n" + "\n".join(us_items) + "\n\n【台股重點新聞】\n" + "\n".join(tw_items)
    log(f"新聞抓取完成：美股 {len(us_items)} 則，台股 {len(tw_items)} 則。")
    return digest


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
            "請以「本週台美股市場回顧」與「下週觀察重點」為主，內部篇幅分配約美股 30%、台股 70%。"
            "嚴格禁止說「今天開盤」、「盤前觀察」或把過去日期當成今天。"
        )
    else:
        mode_instruction = (
            f"今天是 {today_str}（{weekday_str}），屬於交易日早晨。"
            "內部篇幅分配約美股 30%、台股 70%：前半段簡要帶過隔夜美股與科技焦點；"
            "後半段著重台股盤前觀察、產業趨勢與可留意的產業方向及代表性個股。"
        )

    prompt = f"""
你是每天早上七點播報的財經主播。現在時間是台灣時間 {today_str}（{weekday_str}）。
{mode_instruction}

請根據以下財經新聞標題與摘要，寫一份約 700 到 800 字（朗讀時間約 3 分鐘）的繁體中文晨報口語稿。
撰寫風格與結構要求：
1. **開場與結尾精簡**：開頭一句話問候早安並點出今天是 {today_str} {weekday_str}，直接進入市場焦點，不要冗長開場；結尾簡短提醒留意波動與風險、祝投資順利即可。
2. **內部篇幅比例與台股重點**：內部撰寫時美股占約三成篇幅、台股占約七成篇幅，但**播報稿中嚴禁提到「百分之三十」「百分之七十」「三成」「七成」或「比例」「比重」等字眼**，直接自然銜接美股與台股內容即可。美股部分圍繞隔夜股市、人工智慧晶片或重點科技企業動態；台股部分結合盤前氣氛與產業訊號，並務實說明接下來可關注的產業方向（例如半導體先進製程與封測、AI伺服器與散熱、機器人與自動化設備等）以及市場常留意的代表性個股（如台積電、聯發科、鴻海、廣達、台達電或機器人與自動化相關供應鏈），語氣客觀務實，不講不切實際的保證或誇大預測。
3. **尾聲熱門 ETF 參考**：在播報尾聲（結尾前），順暢帶入幾支市場討論度高的台股 ETF 當作配置參考（例如高股息型的 00919 群益台灣精選高息、0056 元大高股息，或主動/主題型如 00406A 等熱門標的），說明投資人在市場震盪時可依個人風險偏好作為現金流或市值布局的參考，簡短 2 到 3 句話帶過即可。
4. **事實依據與口語化**：新聞事件與數字請依據提供的標題與摘要脈絡撰寫，勿自行捏造新聞數據；使用自然口語與完整句子，避免 Markdown 星號、條列符號或難唸縮寫，英文與數字代號請附上好唸的中文口訣或完整稱呼（例如輝達、特斯拉、博通、美光、標普五百、那斯達克、零零九一九、零零五六）。

近期財經新聞與摘要：
{news_digest}
"""
    models_to_try = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]

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
    log("開始合成語音 (edge-tts zh-TW-HsiaoChenNeural)...")
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
    news_digest = fetch_balanced_headlines()
    script = generate_script(news_digest, now_tw)

    with open("briefing.txt", "w", encoding="utf-8") as f:
        f.write(script)

    asyncio.run(text_to_speech(script))

    if AUDIO_PUBLIC_URL:
        send_line(script, AUDIO_PUBLIC_URL)
    else:
        log("提示：未設定 AUDIO_PUBLIC_URL，將在 GitHub Release 步驟後由 Workflow 進行推播。")


if __name__ == "__main__":
    main()
