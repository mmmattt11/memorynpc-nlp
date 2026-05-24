"""Baseline agent for comparing MemoryNPC against a plain LLM role prompt.

This file exists to make the baseline explicit. The baseline can roleplay Elara,
but it does not have durable memory extraction, vector retrieval, trust state, or
a validation trace. That contrast is the core reason MemoryNPC is an NLP system
built on top of an LLM rather than just a chatbot prompt.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class RoleOnlyNPCBaseline:
    """A plain role-prompt LLM baseline with no external memory or tools."""

    def __init__(self, chat_model: str = "gpt-4o-mini", temperature: float = 0.4) -> None:
        load_dotenv(override=True)
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is missing.")

        self.chat_model_name = os.getenv("OPENAI_MODEL", chat_model)
        self.llm = ChatOpenAI(model=self.chat_model_name, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Elara, a cautious, practical village blacksmith in a fantasy village. "
                    "Answer in character. You do not have an external memory system.",
                ),
                ("human", "Player input: {player_input}\n\nElara's response:"),
            ]
        )
        self.chain = self.prompt | self.llm

    def generate_response(self, player_input: str) -> str:
        """Generate one baseline response from only the current player message."""
        return self.chain.invoke({"player_input": player_input}).content.strip()


BASELINE_CAPABILITY_COMPARISON = [
    {
        "capability": "Long-term memory",
        "role_only_baseline": "No external store; only current prompt context.",
        "memorynpc": "Stores durable memories in FAISS and retrieves top-k relevant memories.",
    },
    {
        "capability": "Relationship state",
        "role_only_baseline": "Tone is implicit and may drift.",
        "memorynpc": "Uses deterministic trust score and trust bands.",
    },
    {
        "capability": "Inspectability",
        "role_only_baseline": "No trace of intermediate decisions.",
        "memorynpc": "Logs intent, extracted memory, retrieval, trust delta, and response.",
    },
    {
        "capability": "Validation",
        "role_only_baseline": "Mostly judged by final response quality.",
        "memorynpc": "Internal components can be tested separately.",
    },
]
