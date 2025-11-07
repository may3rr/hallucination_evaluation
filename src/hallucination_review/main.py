from __future__ import annotations

"""Command-line entry point for generating literature reviews."""

import argparse
import asyncio
import uuid

from autogen_core import TopicId

from .models.messages import TopicRequest
from .runtime.factory import TOPIC_TYPE, build_runtime, shutdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a literature review using the multi-agent system.")
    parser.add_argument("topic", help="Topic for the literature review")
    parser.add_argument("--instructions", help="Additional instructions", default=None)
    return parser.parse_args()


async def main_async(topic: str, instructions: str | None) -> None:
    bundle = await build_runtime()
    runtime = bundle.runtime
    runtime.start()

    request_id = str(uuid.uuid4())
    topic_id = TopicId(type=TOPIC_TYPE, source=request_id)

    await runtime.publish_message(
        TopicRequest(request_id=request_id, topic=topic, instructions=instructions),
        topic_id=topic_id,
    )

    await runtime.stop_when_idle()
    await shutdown(bundle)


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args.topic, args.instructions))


if __name__ == "__main__":
    main()
