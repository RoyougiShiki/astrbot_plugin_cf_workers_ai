"""Cloudflare Workers AI Provider 插件

为 AstrBot 提供 Cloudflare Workers AI 的 STT 和 TTS Provider。
在 initialize() 中动态注册，避免插件重载时的重复注册冲突。

注册的 Provider 类型：
  - cf_whisper_api:   Cloudflare Whisper 语音转文字
  - cf_melotts_api:   Cloudflare MeloTTS 文字转语音（中英日韩等）
  - cf_aura_api:      Cloudflare Deepgram Aura 文字转语音（英语）
"""

from __future__ import annotations

import base64
import os
import uuid

import aiohttp

from astrbot.api import AstrBotConfig, logger
from astrbot.api.star import Context, Star, register
from astrbot.core.provider.entities import ProviderMetaData, ProviderType
from astrbot.core.provider.provider import STTProvider, TTSProvider
from astrbot.core.provider.register import provider_cls_map, provider_registry
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path


# ─── Provider 类定义（不使用装饰器，在 initialize 中手动注册）───

class ProviderCFWhisperAPI(STTProvider):
    """Cloudflare Workers AI Whisper 语音转文字"""

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        self.account_id = provider_config.get("cf_account_id", "")
        self.api_token = provider_config.get("api_key", "")
        self.model = provider_config.get("model", "@cf/openai/whisper")
        self.proxy = provider_config.get("proxy", "")

    async def get_text(self, audio_url: str) -> str:
        if audio_url.startswith("http"):
            audio_path = await self._download(audio_url)
        else:
            audio_path = audio_url

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        api_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"
            f"/ai/run/{self.model}"
        )
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {"audio": list(audio_bytes)}

        proxy_url = self.proxy if self.proxy else None
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url, json=payload, headers=headers, proxy=proxy_url,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"CF Whisper 错误 ({resp.status}): {text[:200]}")
                data = await resp.json()

        if not data.get("success"):
            errors = data.get("errors", [])
            msg = errors[0].get("message", "Unknown") if errors else "Unknown"
            raise RuntimeError(f"CF Whisper 失败: {msg}")

        return data.get("result", {}).get("text", "")

    async def _download(self, url: str) -> str:
        temp_dir = get_astrbot_temp_path()
        path = os.path.join(temp_dir, f"cf_whisper_{uuid.uuid4().hex[:8]}.wav")
        proxy_url = self.proxy if self.proxy else None
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, proxy=proxy_url,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                with open(path, "wb") as f:
                    f.write(await resp.read())
        return path

    async def terminate(self):
        pass


class ProviderCFMeloTTSAPI(TTSProvider):
    """Cloudflare Workers AI MeloTTS 文字转语音"""

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        self.account_id = provider_config.get("cf_account_id", "")
        self.api_token = provider_config.get("api_key", "")
        self.model = provider_config.get("model", "@cf/myshell-ai/melotts")
        self.language = provider_config.get("language", "zh")
        self.proxy = provider_config.get("proxy", "")

    async def get_audio(self, text: str) -> str:
        api_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"
            f"/ai/run/{self.model}"
        )
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {"prompt": text, "language": self.language}

        proxy_url = self.proxy if self.proxy else None
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url, json=payload, headers=headers, proxy=proxy_url,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    raise RuntimeError(f"MeloTTS 错误 ({resp.status}): {err[:200]}")
                data = await resp.json()

        if not data.get("success"):
            errors = data.get("errors", [])
            msg = errors[0].get("message", "Unknown") if errors else "Unknown"
            raise RuntimeError(f"MeloTTS 失败: {msg}")

        audio_b64 = data.get("result", {}).get("audio", "")
        if not audio_b64:
            raise RuntimeError("MeloTTS 返回空音频")

        audio_bytes = base64.b64decode(audio_b64)
        temp_dir = get_astrbot_temp_path()
        path = os.path.join(temp_dir, f"cf_melotts_{uuid.uuid4().hex[:8]}.wav")
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path

    async def terminate(self):
        pass


class ProviderCFAuraAPI(TTSProvider):
    """Cloudflare Workers AI Deepgram Aura 文字转语音"""

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        self.account_id = provider_config.get("cf_account_id", "")
        self.api_token = provider_config.get("api_key", "")
        self.model = provider_config.get("model", "@cf/deepgram/aura-1")
        self.proxy = provider_config.get("proxy", "")

    async def get_audio(self, text: str) -> str:
        api_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"
            f"/ai/run/{self.model}"
        )
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {"text": text}

        proxy_url = self.proxy if self.proxy else None
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url, json=payload, headers=headers, proxy=proxy_url,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    raise RuntimeError(f"Aura 错误 ({resp.status}): {err[:200]}")
                audio_bytes = await resp.read()

        if not audio_bytes:
            raise RuntimeError("Aura 返回空音频")

        temp_dir = get_astrbot_temp_path()
        path = os.path.join(temp_dir, f"cf_aura_{uuid.uuid4().hex[:8]}.mp3")
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path

    async def terminate(self):
        pass


# ─── Provider 注册表 ────────────────────────────────────

PROVIDERS = [
    {
        "type_name": "cf_whisper_api",
        "desc": "Cloudflare Whisper API",
        "provider_type": ProviderType.SPEECH_TO_TEXT,
        "cls": ProviderCFWhisperAPI,
        "default_config_tmpl": {
            "id": "cf_whisper",
            "type": "cf_whisper_api",
            "provider_type": "speech_to_text",
            "enable": False,
            "api_key": "",
            "cf_account_id": "",
            "model": "@cf/openai/whisper",
            "proxy": "",
        },
    },
    {
        "type_name": "cf_melotts_api",
        "desc": "Cloudflare MeloTTS API",
        "provider_type": ProviderType.TEXT_TO_SPEECH,
        "cls": ProviderCFMeloTTSAPI,
        "default_config_tmpl": {
            "id": "cf_melotts",
            "type": "cf_melotts_api",
            "provider_type": "text_to_speech",
            "enable": False,
            "api_key": "",
            "cf_account_id": "",
            "model": "@cf/myshell-ai/melotts",
            "language": "zh",
            "proxy": "",
        },
    },
    {
        "type_name": "cf_aura_api",
        "desc": "Cloudflare Aura TTS API",
        "provider_type": ProviderType.TEXT_TO_SPEECH,
        "cls": ProviderCFAuraAPI,
        "default_config_tmpl": {
            "id": "cf_aura",
            "type": "cf_aura_api",
            "provider_type": "text_to_speech",
            "enable": False,
            "api_key": "",
            "cf_account_id": "",
            "model": "@cf/deepgram/aura-1",
            "proxy": "",
        },
    },
]


def _register_providers():
    """手动注册 Provider（支持重复注册时覆盖）"""
    for p in PROVIDERS:
        type_name = p["type_name"]
        tmpl = p["default_config_tmpl"]
        if tmpl:
            tmpl.setdefault("type", type_name)
            tmpl.setdefault("enable", False)
            tmpl.setdefault("id", type_name)

        pm = ProviderMetaData(
            id="default",
            model=None,
            type=type_name,
            desc=p["desc"],
            provider_type=p["provider_type"],
            cls_type=p["cls"],
            default_config_tmpl=tmpl,
        )

        # 覆盖旧注册（如果存在）
        if type_name in provider_cls_map:
            old_pm = provider_cls_map[type_name]
            if old_pm in provider_registry:
                provider_registry.remove(old_pm)

        provider_registry.append(pm)
        provider_cls_map[type_name] = pm
        logger.debug(f"CF Workers AI: Provider {type_name} 已注册")


# ─── 插件主体 ─────────────────────────────────────────────

@register(
    "astrbot_plugin_cf_workers_ai",
    "RoyougiShiki",
    "Cloudflare Workers AI Provider - ASR(Whisper) + TTS(MeloTTS/Aura)",
    "0.1.0",
)
class CFWorkersAIPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)

    async def initialize(self):
        _register_providers()
        logger.info("CF Workers AI Provider 已加载: cf_whisper_api (ASR), cf_melotts_api (TTS), cf_aura_api (TTS)")

    async def terminate(self):
        logger.info("CF Workers AI Provider 已卸载")
