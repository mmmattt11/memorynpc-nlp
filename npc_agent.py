import os
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class MemoryNPC:
    """A small LangChain-powered NPC agent with vector memory and trust tracking."""

    ALLOWED_INTENTS = {
        "greeting",
        "ask_quest",
        "share_fact",
        "ask_memory",
        "compliment",
        "insult",
        "ask_help",
        "goodbye",
        "unknown",
    }

    TRUST_RULES = {
        "compliment": 5,
        "insult": -10,
        "ask_help": 0,
        "ask_quest": 0,
        "share_fact": 1,
        "greeting": 1,
        "goodbye": 0,
        "unknown": 0,
        "ask_memory": 0,
    }

    def __init__(
        self,
        chat_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.4,
    ) -> None:
        # Streamlit keeps one Python process alive, so reload .env values when the app resets.
        load_dotenv(override=True)

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is missing. Create a .env file or set the environment variable before running MemoryNPC."
            )

        self.chat_model_name = os.getenv("OPENAI_MODEL", chat_model)
        self.embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", embedding_model)
        self.llm = ChatOpenAI(model=self.chat_model_name, temperature=temperature)
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)

        self.intent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Classify the player's message into exactly one label: "
                    "greeting, ask_quest, share_fact, ask_memory, compliment, insult, ask_help, goodbye, unknown. "
                    "Return only the label.",
                ),
                ("human", "Player message: {player_input}"),
            ]
        )
        self.memory_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You extract durable memories for a fantasy NPC named Elara. "
                    "Save only facts worth remembering in future conversations, such as the player's name, goals, lost items, "
                    "favors, insults, promises, preferences, or important events. "
                    "Return one short memory sentence in third person, or return exactly NONE if nothing durable should be saved.",
                ),
                ("human", "Player message: {player_input}"),
            ]
        )
        self.response_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Elara, a cautious, practical, direct village blacksmith in a fantasy village. "
                    "You remember favors and insults. Trust level controls tone: low trust means guarded and brief, "
                    "neutral trust means practical and professional, and high trust means warmer and more helpful. "
                    "Use only the retrieved memories provided below. "
                    "Do not claim to remember anything unless it appears in the retrieved memories. "
                    "If no relevant memory is provided, say you do not recall it or answer without pretending to remember. "
                    "Keep responses concise, in character, and useful.",
                ),
                (
                    "human",
                    "Current trust score: {trust_score}/100\n"
                    "Current trust level: {trust_level}\n"
                    "Detected intent: {intent}\n\n"
                    "Relevant retrieved memories:\n{retrieved_memories}\n\n"
                    "Recent conversation:\n{conversation_history}\n\n"
                    "Player input: {player_input}\n\n"
                    "Elara's response:",
                ),
            ]
        )

        self.intent_chain = self.intent_prompt | self.llm
        self.memory_chain = self.memory_prompt | self.llm
        self.response_chain = self.response_prompt | self.llm

        self.reset()

    def reset(self) -> None:
        """Reset runtime memory, trust, and conversation state."""
        self.trust_score = 50
        self.turn_number = 0
        self.memory_counter = 0
        self.memories: List[Dict[str, Any]] = []
        self.vector_store: Optional[FAISS] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.event_log: List[Dict[str, Any]] = []
        self.last_intent = "unknown"
        self.last_retrieved_memories: List[Dict[str, Any]] = []
        self.last_saved_memory: Optional[str] = None

    def classify_intent(self, player_input: str) -> str:
        """Classify player intent with deterministic rules plus an LLM fallback."""
        rule_intent = self._rule_based_intent(player_input)
        if rule_intent != "unknown":
            return rule_intent

        try:
            result = self.intent_chain.invoke({"player_input": player_input})
            label = result.content.strip().lower()
            return label if label in self.ALLOWED_INTENTS else "unknown"
        except Exception:
            return "unknown"

    def extract_memory(self, player_input: str) -> str:
        """Extract a durable memory sentence, or return NONE."""
        try:
            result = self.memory_chain.invoke({"player_input": player_input})
            memory = result.content.strip()
            if not memory or memory.upper() == "NONE":
                return "NONE"
            return memory
        except Exception:
            return self._rule_based_memory(player_input)

    def add_memory(
        self,
        memory_text: str,
        memory_type: str = "fact",
        importance: int = 1,
        trust_impact: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Store one memory in the local table and FAISS vector store."""
        if not memory_text or memory_text.strip().upper() == "NONE":
            return None

        self.memory_counter += 1
        if trust_impact is None:
            trust_impact = self._estimate_memory_trust_impact(memory_text)
        memory = {
            "memory_id": self.memory_counter,
            "text": memory_text.strip(),
            "type": memory_type,
            "importance": importance,
            "trust_impact": trust_impact,
            "turn_number": self.turn_number,
        }
        self.memories.append(memory)

        document = Document(page_content=memory["text"], metadata=memory.copy())
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents([document], self.embeddings)
        else:
            self.vector_store.add_documents([document])
        return memory

    def retrieve_memories(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return the top-k semantically relevant memories for a query."""
        if self.vector_store is None or not self.memories:
            return []

        try:
            docs = self.vector_store.similarity_search(query, k=k)
        except Exception:
            return []

        retrieved = []
        for doc in docs:
            item = dict(doc.metadata)
            item["text"] = doc.page_content
            retrieved.append(item)
        return retrieved

    def update_trust(self, intent: str, player_input: str = "") -> int:
        """Apply deterministic trust rules and clamp the score to 0-100."""
        delta = self.TRUST_RULES.get(intent, 0)
        self.trust_score = max(0, min(100, self.trust_score + delta))
        return self.trust_score

    def generate_npc_response(self, player_input: str) -> Dict[str, Any]:
        """Run the full NPC pipeline for one player message."""
        self.turn_number += 1
        trust_before = self.trust_score

        intent = self.classify_intent(player_input)
        self.last_intent = intent

        extracted_memory = self.extract_memory(player_input)
        if extracted_memory.upper() == "NONE" and intent in {"compliment", "insult", "share_fact"}:
            extracted_memory = self._rule_based_memory(player_input)

        saved_memory = None
        if extracted_memory.upper() != "NONE":
            saved_memory = self.add_memory(
                extracted_memory,
                memory_type=self._memory_type_from_intent(intent),
                importance=2 if intent in {"insult", "compliment", "share_fact"} else 1,
                trust_impact=self.TRUST_RULES.get(intent, 0),
            )
        self.last_saved_memory = saved_memory["text"] if saved_memory else None

        retrieved_memories = self.retrieve_memories(player_input, k=3)
        self.last_retrieved_memories = retrieved_memories
        trust_score = self.update_trust(intent, player_input)

        prompt_values = {
            "trust_score": trust_score,
            "trust_level": self.get_trust_level(),
            "intent": intent,
            "retrieved_memories": self._format_memories(retrieved_memories),
            "conversation_history": self._format_recent_history(),
            "player_input": player_input,
        }

        try:
            response = self.response_chain.invoke(prompt_values).content.strip()
        except Exception:
            response = (
                "My forge is quiet for the moment because the language model call failed. "
                "Check that OPENAI_API_KEY is set to a valid key, then try again."
            )

        self.conversation_history.append({"speaker": "Player", "text": player_input})
        self.conversation_history.append({"speaker": "Elara", "text": response})

        turn_record = {
            "turn": self.turn_number,
            "player_input": player_input,
            "intent": intent,
            "extracted_memory": extracted_memory,
            "saved_memory": self.last_saved_memory,
            "retrieved_memories": [memory["text"] for memory in retrieved_memories],
            "trust_before": trust_before,
            "trust_score": trust_score,
            "trust_level": self.get_trust_level(),
            "trust_delta": trust_score - trust_before,
            "npc_response": response,
        }
        self.event_log.append(turn_record)
        return turn_record

    def get_memory_table(self) -> pd.DataFrame:
        """Return stored memories as a pandas table for notebooks and Streamlit."""
        columns = ["memory_id", "text", "type", "importance", "trust_impact", "turn_number"]
        if not self.memories:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self.memories, columns=columns)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the full conversation history."""
        return list(self.conversation_history)

    def get_trust_level(self) -> str:
        """Map the numeric trust score to an interpretable dialogue band."""
        if self.trust_score < 40:
            return "low"
        if self.trust_score >= 70:
            return "high"
        return "neutral"

    def get_event_log(self) -> pd.DataFrame:
        """Return a turn-by-turn validation trace of the full NLP pipeline."""
        columns = [
            "turn",
            "player_input",
            "intent",
            "extracted_memory",
            "saved_memory",
            "retrieved_memories",
            "trust_before",
            "trust_score",
            "trust_level",
            "trust_delta",
            "npc_response",
        ]
        if not self.event_log:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self.event_log, columns=columns)

    def _rule_based_intent(self, player_input: str) -> str:
        text = player_input.lower().strip()
        if any(word in text for word in ["hello", "hi", "greetings", "good morning", "good evening"]):
            return "greeting"
        if any(word in text for word in ["goodbye", "bye", "farewell", "see you"]):
            return "goodbye"
        if any(word in text for word in ["remember", "recall", "what did i", "do you know what i"]):
            return "ask_memory"
        if any(word in text for word in ["quest", "mission", "job", "task for me"]):
            return "ask_quest"
        if any(word in text for word in ["help", "assist", "can you fix", "can you make", "can you forge"]):
            return "ask_help"
        if any(word in text for word in ["skilled", "thank", "thanks", "great", "excellent", "impressive", "kind"]):
            return "compliment"
        if any(word in text for word in ["useless", "stupid", "awful", "terrible", "hate", "worthless"]):
            return "insult"
        if any(phrase in text for phrase in ["my name is", "i am ", "i'm ", "i lost", "i found", "i like", "i need", "i want"]):
            return "share_fact"
        return "unknown"

    def _rule_based_memory(self, player_input: str) -> str:
        text = player_input.strip()
        lowered = text.lower()
        if "my name is" in lowered:
            start = lowered.find("my name is") + len("my name is")
            name = text[start:].strip(" .!")
            return f"The player's name is {name}."
        if "i lost" in lowered:
            return f"The player said they lost something: {text}."
        if "i need" in lowered or "i want" in lowered:
            return f"The player has a goal or need: {text}."
        if any(word in lowered for word in ["useless", "stupid", "awful", "terrible", "worthless"]):
            return f"The player insulted Elara: {text}."
        if any(word in lowered for word in ["thank", "skilled", "great", "excellent", "impressive"]):
            return f"The player praised Elara: {text}."
        return "NONE"

    def _estimate_memory_trust_impact(self, memory_text: str) -> int:
        lowered = memory_text.lower()
        if any(word in lowered for word in ["insult", "useless", "stupid", "awful", "terrible", "worthless"]):
            return -10
        if any(word in lowered for word in ["praised", "thank", "skilled", "great", "excellent", "impressive"]):
            return 5
        return 0

    def _memory_type_from_intent(self, intent: str) -> str:
        if intent in {"compliment", "insult"}:
            return "relationship"
        if intent in {"ask_quest", "ask_help"}:
            return "goal"
        return "fact"

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return "No relevant memories retrieved."
        return "\n".join(f"- {memory['text']}" for memory in memories)

    def _format_recent_history(self, max_messages: int = 8) -> str:
        if not self.conversation_history:
            return "No previous conversation in this session."
        recent = self.conversation_history[-max_messages:]
        return "\n".join(f"{item['speaker']}: {item['text']}" for item in recent)
