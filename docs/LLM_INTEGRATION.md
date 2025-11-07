# LLM 调用与中转 API 集成方案

## 1. 平台信息
- **终端点**：`https://api.gpt.ge/v1/chat/completions`
- **鉴权方式**：`Authorization: Bearer <token>`，token 存储于 `.env` 中的 `V3_API_KEY`。
- **模型**：默认使用 `gpt-4o-mini`，可根据需要在请求体中调整。

## 2. 通用请求结构
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." }
  ],
  "max_tokens": 1688,
  "temperature": 0.5,
  "stream": false
}
```

## 3. Python 封装示例
```python
import os
import httpx
from typing import Iterable, Mapping

API_URL = "https://api.gpt.ge/v1/chat/completions"
API_KEY = os.getenv("V3_API_KEY")

class GPTGateway:
    """封装中转站 Chat Completion API"""

    def __init__(self, *, timeout: float = 60.0):
        if not API_KEY:
            raise RuntimeError("V3_API_KEY is not set")
        self._client = httpx.AsyncClient(
            base_url=API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=timeout
        )

    async def chat(self, messages: Iterable[Mapping[str, str]], *,
                   model: str = "gpt-4o-mini", max_tokens: int = 1688,
                   temperature: float = 0.5, stream: bool = False) -> dict:
        payload = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        response = await self._client.post("", json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()
```

## 4. AutoGen 集成要点
- 在 `ModelClient` 实现中调用 `GPTGateway.chat`，确保与 AutoGen 的异步接口兼容。
- 对流式需求，可将 `stream=True` 并处理增量返回（需在路由层实现）。
- 记录请求与响应摘要（避免敏感信息泄露）。
- 在测试环境中使用 Mock，防止频繁调用真实接口。

## 5. 错误处理
- 统一捕获 `httpx.HTTPStatusError`，打印状态码与错误信息。
- 当返回 401/403 时，提示检查 `V3_API_KEY`。
- 对 5xx 错误应用指数退避重试，并记录日志。

## 6. 配置与安全
- `.env` 文件仅在本地存放，禁止提交。
- 在部署环境中通过安全的 Secret 管理服务注入 `V3_API_KEY`。
- `GPTGateway` 应提供健康检查方法，便于监控连接状态。

