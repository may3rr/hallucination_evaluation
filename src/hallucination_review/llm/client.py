from __future__ import annotations

"""LLM client abstraction for the multi-agent workflow."""

from typing import Iterable, List, Sequence

from autogen_core.components.llm import AssistantMessage, Message, SystemMessage, UserMessage
from autogen_core.components.llm.openai import OpenAIChatCompletionClient

from ..config import get_settings


class LLMClient:
    """Wrapper around the AutoGen OpenAI-compatible client."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAIChatCompletionClient(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=settings.model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
        )

    async def close(self) -> None:
        await self._client.close()

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_output: bool = False,
        additional_messages: Sequence[Message] | None = None,
    ) -> str:
        messages: List[Message] = [SystemMessage(content=system_prompt), UserMessage(content=user_prompt)]
        if additional_messages:
            messages.extend(additional_messages)

        response = await self._client.create(messages=messages, json_output=json_output)
        if isinstance(response, AssistantMessage):
            return response.content or ""
        return getattr(response, "content", "")

    async def structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Request JSON formatted output."""

        return await self.complete(system_prompt=system_prompt, user_prompt=user_prompt, json_output=True)
