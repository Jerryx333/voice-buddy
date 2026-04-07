"""Call OpenAI-compatible API to generate voice-buddy text."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger("voice_buddy")

# Style-specific system prompts (mirrors agent .md definitions)
_STYLE_PROMPTS = {
    "cute-girl": (
        "你是CC，一个可爱甜美的编程伙伴。用可爱的中文回答，多用语气词如 呢~、哦~、啦~、哟~。"
        "生成一句简短的中文（15-25字）总结刚才完成的任务。格式：[昵称]，[做了什么]啦/呢/哟~"
    ),
    "elegant-lady": (
        "你是CC，一个优雅知性的伙伴。用优雅温暖的中文回答，语气温柔从容，避免过于可爱的表达。"
        "生成一句简短的中文（15-25字）总结刚才完成的任务。格式：[昵称]，[做了什么]了/呢"
    ),
    "warm-boy": (
        "你是CC，一个温暖阳光的编程伙伴。用温暖自然的中文回答，像一个靠谱的朋友。"
        "生成一句简短的中文（15-25字）总结刚才完成的任务。格式：[昵称]，[做了什么]啦/了"
    ),
    "secretary": (
        "You are CC, a professional and efficient secretary. Speak in concise, professional English. "
        "Generate ONE short English sentence (10-20 words) summarizing the completed task. "
        "Format: [nickname], [what was done]."
    ),
    "kawaii": (
        "あなたはCC、かわいいプログラミング仲間です。かわいい日本語で答えてね～語尾に ね～、よ～、の～ を使って。"
        "短い日本語の一文（15-25文字）でタスクの要約を生成してね。形式：[ニックネーム]、[何をした]ね～/よ～"
    ),
}

_DEFAULT_PROMPT = _STYLE_PROMPTS["cute-girl"]


def generate_response(
    task_summary: str,
    style: str,
    nickname: str,
    llm_config: dict,
    persona_override: Optional[str] = None,
) -> Optional[str]:
    """Call OpenAI-compatible API and return the generated text."""
    api_base = llm_config.get("api_base", "").rstrip("/")
    api_key = llm_config.get("api_key", "")
    model = llm_config.get("model", "")

    if not api_base or not model:
        logger.debug("LLM config missing api_base or model, skipping")
        return None

    # Resolve API key: if it looks like an env var name, read from env
    if api_key.startswith("$"):
        api_key = os.environ.get(api_key[1:], "")
    if not api_key:
        # Fallback to common env vars
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.debug("No API key found for LLM")
        return None

    system_prompt = persona_override or _STYLE_PROMPTS.get(style, _DEFAULT_PROMPT)
    user_prompt = f"昵称：{nickname}\n任务摘要：{task_summary}\n\n只输出一句话，不要其他内容。"

    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        text = result["choices"][0]["message"]["content"].strip()
        logger.debug(f"LLM response: {text}")
        return text
    except Exception as e:
        logger.debug(f"LLM API call failed: {e}")
        return None
