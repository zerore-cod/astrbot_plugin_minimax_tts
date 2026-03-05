# MiniMax TTS 插件（minimax_tts）

[![Version](https://img.shields.io/badge/version-0.10-blue.svg)](https://github.com/zerore-cod/astrbot_plugin_minimax_tts)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

MiniMax TTS 插件：尽量还原 MiniMax 的克隆声线，支持多种情绪声线切换、会话策略、分段/概率输出与按需语音调用。

## 功能概览

- 支持双 TTS 服务商：`minimax` / `siliconflow`
- 支持多种情绪声线切换（由会话情绪/标记/手动设置驱动）
- 多策略控制：
  - 自动语音输出 `voice_output`
  - 文字+语音同时输出 `text_voice_output`
  - 分段语音输出 `segmented_output`
  - 概率语音输出 `probability_output`
- 支持按需触发语音输出：
  - 命令：`tts_say`
  - LLM 工具：`tts_speak(text: str)`

## 快速开始

1. 在 AstrBot 插件目录安装并启用本插件。
2. 安装 `ffmpeg`（系统可直接调用即可）。
3. 在插件配置面板填写 TTS 参数（推荐先用 MiniMax 跑通）。
4. 在群聊或私聊发送：
   - `/sid` 获取当前会话 UMO
   - `tts_status` 查看当前插件状态
   - `tts_test` 或 `tts_say` 测试语音输出

## 常用命令

| 命令 | 说明 |
|------|------|
| `tts_help` | 指令说明 |
| `tts_on` / `tts_off` | 开关当前会话自动语音 |
| `tts_all_on` / `tts_all_off` | 开关全局自动语音（保留按需语音） |
| `tts_prob_on` / `tts_prob_off` | 开关概率策略 |
| `tts_prob <0~1>` | 设置概率（例如 `tts_prob 0.35`） |
| `tts_emotion <...>` | 设置情绪（`auto` 表示不传 emotion 字段） |
| `tts_payload_preview [文本]` | 预览 MiniMax 请求体 |
| `tts_say [文本]` | 立即合成并发送语音 |
| `tts_test [文本]` | `tts_say` 别名 |

LLM 工具（函数调用）：

- `tts_speak(text: str)`：在用户明确要求语音时由模型调用输出语音。

## 配置提示

- UMO：在聊天中发送 `/sid` 获取，黑白名单字段填写 UMO。
- MiniMax：接口使用 `https://api.minimaxi.com/v1/t2a_v2`。
- `pronunciation_dict`：支持 JSON 或简写（可多行），例如：

```text
处理/(chu3)(li3)
```

## 项目信息

- 作者：zerore-cod
- 仓库：<https://github.com/zerore-cod/astrbot_plugin_minimax_tts>
- 协议：MIT
