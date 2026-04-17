# astrbot_plugin_cf_workers_ai

Cloudflare Workers AI Provider 插件，让 AstrBot 原生支持 Cloudflare 的 ASR 和 TTS 模型。

## 功能

| Provider 类型 | 说明 | 模型 |
|---|---|---|
| `cf_whisper_api` | 语音转文字 (ASR) | `@cf/openai/whisper` |
| `cf_melotts_api` | 文字转语音 (TTS) | `@cf/myshell-ai/melotts` |
| `cf_aura_api` | 文字转语音 (TTS) | `@cf/deepgram/aura-1` |

## 使用方法

1. 安装插件
2. 打开 WebUI → 模型提供商 → 新增
3. 选择"语音转文字"或"文字转语音"选项卡
4. 选择 `Cloudflare Whisper API` / `Cloudflare MeloTTS API` / `Cloudflare Aura TTS API`
5. 填写配置：
   - **API Key**: Cloudflare API Token（`cfut_` 开头）
   - **CF Account ID**: Cloudflare Account ID
   - **Model**: 模型名称（默认已填）
   - **Language**（仅 MeloTTS）: 语言代码（zh/en/ja/ko）
   - **Proxy**: 代理地址（可选）
6. 在 AI 配置 → 模型 中选择已添加的 STT/TTS 模型

## 为什么需要这个插件？

Cloudflare Workers AI 不提供 OpenAI 兼容的 Audio 端点（`/v1/audio/transcriptions` 和 `/v1/audio/speech`），
所以 AstrBot 原生的 `Whisper(API)` 和 `OpenAI TTS(API)` Provider 无法对接 Cloudflare。
本插件使用 Cloudflare 原生 API 端点（`/ai/run/`）来调用模型。