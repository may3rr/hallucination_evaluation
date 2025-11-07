from __future__ import annotations

"""Runtime factory for assembling the multi-agent system."""

from dataclasses import dataclass

from autogen_core import SingleThreadedAgentRuntime, TypeSubscription

from ..agents.coordinator import CoordinatorAgent
from ..agents.outline import OutlineAgent
from ..agents.reviewer import ReviewAgent
from ..agents.search import SearchAgent
from ..agents.writer import WritingAgent
from ..arxiv.client import ArxivClient
from ..llm.client import LLMClient
from ..models.papers import PaperRepository


TOPIC_TYPE = "literature_review"


@dataclass
class RuntimeBundle:
    runtime: SingleThreadedAgentRuntime
    llm: LLMClient
    arxiv: ArxivClient


async def build_runtime() -> RuntimeBundle:
    runtime = SingleThreadedAgentRuntime()
    llm = LLMClient()
    arxiv = ArxivClient()
    papers = PaperRepository()

    await CoordinatorAgent.register(runtime, "CoordinatorAgent", lambda: CoordinatorAgent(llm=llm, paper_repo=papers))
    await OutlineAgent.register(runtime, "OutlineAgent", lambda: OutlineAgent(llm=llm, paper_repo=papers))
    await SearchAgent.register(runtime, "SearchAgent", lambda: SearchAgent(llm=llm, paper_repo=papers, arxiv_client=arxiv))
    await WritingAgent.register(runtime, "WritingAgent", lambda: WritingAgent(llm=llm, paper_repo=papers))
    await ReviewAgent.register(runtime, "ReviewAgent", lambda: ReviewAgent(llm=llm, paper_repo=papers))

    for agent_type in [
        "CoordinatorAgent",
        "OutlineAgent",
        "SearchAgent",
        "WritingAgent",
        "ReviewAgent",
    ]:
        await runtime.add_subscription(TypeSubscription(topic_type=TOPIC_TYPE, agent_type=agent_type))

    return RuntimeBundle(runtime=runtime, llm=llm, arxiv=arxiv)


async def shutdown(bundle: RuntimeBundle) -> None:
    await bundle.llm.close()
    await bundle.arxiv.close()
    await bundle.runtime.close()
