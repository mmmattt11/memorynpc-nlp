"""Shared NLP backend for the MemoryNPC project.

This file contains the actual assignment logic. The Streamlit app is only a
demo surface; this class is where the NLP pipeline is built:

player text -> intent detection -> memory extraction -> FAISS memory storage
-> semantic retrieval -> deterministic trust update -> grounded response.

The comments are intentionally detailed because the project is graded on design
motivation as well as working code. They explain both what each block does and
why it was implemented this way.
"""

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

    # A closed label set makes intent detection testable. Without this list, the
    # LLM could invent labels such as "question" or "angry_request", which would
    # make trust updates and validation inconsistent.
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

    # Trust is deliberately deterministic instead of letting the LLM decide
    # relationship changes. This makes the social state inspectable, repeatable,
    # and easy to validate in the notebook.
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

        # Fail early with a clear message. This is better than letting LangChain
        # fail later with a long API error that is harder for a student/demo user
        # to understand.
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is missing. Create a .env file or set the environment variable before running MemoryNPC."
            )

        # The defaults use cheap/fast OpenAI models because this is a prototype
        # and a university assignment. Environment overrides make it easy to swap
        # models without changing code.
        self.chat_model_name = os.getenv("OPENAI_MODEL", chat_model)
        self.embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", embedding_model)
        self.llm = ChatOpenAI(model=self.chat_model_name, temperature=temperature)
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)

        # Intent detection is a separate chain because it is an NLP interpretation
        # step. The final NPC response should not be responsible for silently
        # deciding intent and relationship effects at the same time.
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
        # Memory extraction is also separate from response generation. This avoids
        # saving every utterance and keeps the vector store focused on durable
        # information such as names, goals, favors, insults, and lost items.
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
        # The final response prompt receives only selected state: trust, intent,
        # recent conversation, and top-k retrieved memories. This is a controlled
        # RAG-style prompt, not a raw dump of every past message.
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

        # LangChain's pipe syntax turns prompt templates plus models into small,
        # reusable chains. Keeping three chains makes the architecture visible for
        # the report and easier to test piece by piece.
        self.intent_chain = self.intent_prompt | self.llm
        self.memory_chain = self.memory_prompt | self.llm
        self.response_chain = self.response_prompt | self.llm

        self.reset()

    def reset(self) -> None:
        """Reset runtime memory, trust, and conversation state."""
        # A starting score of 50 means Elara begins neutral, not hostile or overly
        # friendly. This gives compliments and insults room to move the score.
        self.trust_score = 50
        self.turn_number = 0
        self.memory_counter = 0
        # `memories` is the human-readable metadata table. FAISS stores the same
        # memory text as vectors for semantic search, but this list is easier to
        # inspect in the notebook and Streamlit sidebar.
        self.memories: List[Dict[str, Any]] = []
        # FAISS cannot search before it has documents, so the vector store starts
        # as None and is created when the first memory is saved.
        self.vector_store: Optional[FAISS] = None
        # Recent conversation is kept separately from long-term memory. This lets
        # the prompt include short-term dialogue context without polluting durable
        # memory with every line of chat.
        self.conversation_history: List[Dict[str, str]] = []
        # The event log is a validation artifact. It records the internal pipeline
        # decisions for every turn so the project can be evaluated later.
        self.event_log: List[Dict[str, Any]] = []
        self.last_intent = "unknown"
        self.last_retrieved_memories: List[Dict[str, Any]] = []
        self.last_saved_memory: Optional[str] = None

    def classify_intent(self, player_input: str) -> str:
        """Classify player intent with deterministic rules plus an LLM fallback."""
        # Rules handle obvious cases cheaply and consistently. This improves
        # validation accuracy for common examples such as greetings, insults, and
        # memory questions.
        rule_intent = self._rule_based_intent(player_input)
        if rule_intent != "unknown":
            return rule_intent

        try:
            # The LLM fallback handles language that is not covered by the simple
            # rules. Its output is still restricted to ALLOWED_INTENTS so the rest
            # of the pipeline stays predictable.
            result = self.intent_chain.invoke({"player_input": player_input})
            label = result.content.strip().lower()
            return label if label in self.ALLOWED_INTENTS else "unknown"
        except Exception:
            # If the model call fails, unknown is safer than crashing the whole
            # app or making up a label.
            return "unknown"

    def extract_memory(self, player_input: str) -> str:
        """Extract a durable memory sentence, or return NONE."""
        try:
            # This prompt asks the model to compress useful player statements into
            # a short memory. That is an information extraction step, not dialogue.
            result = self.memory_chain.invoke({"player_input": player_input})
            memory = result.content.strip()
            if not memory or memory.upper() == "NONE":
                return "NONE"
            return memory
        except Exception:
            # A tiny rule-based fallback keeps the app usable if the extraction
            # model call fails during a demo.
            return self._rule_based_memory(player_input)

    def add_memory(
        self,
        memory_text: str,
        memory_type: str = "fact",
        importance: int = 1,
        trust_impact: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Store one memory in the local table and FAISS vector store."""
        # "NONE" is the explicit signal from the extraction chain that a message is
        # not worth storing. This prevents filler text from becoming memory noise.
        if not memory_text or memory_text.strip().upper() == "NONE":
            return None

        self.memory_counter += 1
        # Normal conversation memories often have no trust effect. For memories
        # created from a detected intent, generate_npc_response passes the exact
        # deterministic trust impact so the metadata matches the score update.
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

        # LangChain Document metadata keeps the memory text connected to its
        # validation fields. When FAISS returns a document, we can show the same
        # memory_id, type, importance, trust_impact, and turn number in the UI.
        document = Document(page_content=memory["text"], metadata=memory.copy())
        # FAISS is initialized lazily because an empty FAISS store is awkward to
        # query. Creating it on the first document avoids no-document crashes.
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents([document], self.embeddings)
        else:
            self.vector_store.add_documents([document])
        return memory

    def retrieve_memories(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return the top-k semantically relevant memories for a query."""
        # Empty memory is a valid early-conversation state. Returning [] lets the
        # response prompt say "no relevant memories" instead of crashing.
        if self.vector_store is None or not self.memories:
            return []

        try:
            # Top-k retrieval keeps prompts small and focused. This is a common
            # RAG choice: enough context to ground the answer, not so much that the
            # model is distracted by irrelevant memories.
            docs = self.vector_store.similarity_search(query, k=k)
        except Exception:
            # Retrieval can fail if embeddings/API/network fail. Returning no
            # memories is safer than giving the LLM unverified context.
            return []

        retrieved = []
        for doc in docs:
            # Convert LangChain Documents back into normal dictionaries so pandas,
            # Streamlit, and validation code can display them easily.
            item = dict(doc.metadata)
            item["text"] = doc.page_content
            retrieved.append(item)
        return retrieved

    def update_trust(self, intent: str, player_input: str = "") -> int:
        """Apply deterministic trust rules and clamp the score to 0-100."""
        # `player_input` is kept in the signature for future extensions, but the
        # current assignment version uses intent only so the score is reproducible.
        delta = self.TRUST_RULES.get(intent, 0)
        # Clamping prevents impossible relationship values such as -20 or 130.
        self.trust_score = max(0, min(100, self.trust_score + delta))
        return self.trust_score

    def generate_npc_response(self, player_input: str) -> Dict[str, Any]:
        """Run the full NPC pipeline for one player message."""
        # The method intentionally follows the architecture diagram step by step.
        # The returned dictionary becomes both the app pipeline details and the
        # validation event log.
        self.turn_number += 1
        trust_before = self.trust_score

        # 1. Interpret the player's message as an intent label.
        intent = self.classify_intent(player_input)
        self.last_intent = intent

        # 2. Extract durable memory, if the utterance contains anything worth
        # remembering. If the model declines but the intent clearly matters for
        # relationship testing, use the rule fallback to keep validation visible.
        extracted_memory = self.extract_memory(player_input)
        if extracted_memory.upper() == "NONE" and intent in {"compliment", "insult", "share_fact"}:
            extracted_memory = self._rule_based_memory(player_input)

        saved_memory = None
        if extracted_memory.upper() != "NONE":
            # 3. Store the memory in both the readable table and FAISS. The
            # trust_impact is tied to the deterministic intent rule so the memory
            # metadata explains why trust changed.
            saved_memory = self.add_memory(
                extracted_memory,
                memory_type=self._memory_type_from_intent(intent),
                importance=2 if intent in {"insult", "compliment", "share_fact"} else 1,
                trust_impact=self.TRUST_RULES.get(intent, 0),
            )
        self.last_saved_memory = saved_memory["text"] if saved_memory else None

        # 4. Retrieve relevant long-term memories for the current message.
        retrieved_memories = self.retrieve_memories(player_input, k=3)
        self.last_retrieved_memories = retrieved_memories
        # 5. Update trust after intent detection. Doing this deterministically
        # makes social state validation independent of LLM randomness.
        trust_score = self.update_trust(intent, player_input)

        # 6. Build the final prompt inputs. Only selected evidence and state are
        # injected, which makes the generated answer more grounded and inspectable.
        prompt_values = {
            "trust_score": trust_score,
            "trust_level": self.get_trust_level(),
            "intent": intent,
            "retrieved_memories": self._format_memories(retrieved_memories),
            "conversation_history": self._format_recent_history(),
            "player_input": player_input,
        }

        try:
            # 7. Generate Elara's final natural-language response.
            response = self.response_chain.invoke(prompt_values).content.strip()
        except Exception:
            # Do not expose raw API errors or key fragments in the UI. The message
            # is enough for the user to fix configuration without leaking secrets.
            response = (
                "My forge is quiet for the moment because the language model call failed. "
                "Check that OPENAI_API_KEY is set to a valid key, then try again."
            )

        # Keep a short visible transcript for prompt context and app display. This
        # is separate from event_log because it is dialogue, not validation data.
        self.conversation_history.append({"speaker": "Player", "text": player_input})
        self.conversation_history.append({"speaker": "Elara", "text": response})

        # The turn record is the most important validation artifact. It stores the
        # internal pipeline decisions that explain the final response.
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
        # Returning an empty DataFrame with fixed columns keeps Streamlit and the
        # notebook from failing before the first memory exists.
        columns = ["memory_id", "text", "type", "importance", "trust_impact", "turn_number"]
        if not self.memories:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(self.memories, columns=columns)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the full conversation history."""
        return list(self.conversation_history)

    def get_trust_level(self) -> str:
        """Map the numeric trust score to an interpretable dialogue band."""
        # These thresholds are simple by design. They give the LLM an interpretable
        # tone instruction without pretending to model complex human relationships.
        if self.trust_score < 40:
            return "low"
        if self.trust_score >= 70:
            return "high"
        return "neutral"

    def get_event_log(self) -> pd.DataFrame:
        """Return a turn-by-turn validation trace of the full NLP pipeline."""
        # This schema mirrors the assignment validation questions: What was the
        # input? How was it interpreted? What memory was saved/retrieved? How did
        # trust change? What answer was generated?
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
        """Fast fallback intent classifier for common, easy-to-test phrases."""
        text = player_input.lower().strip()
        # Order matters: memory questions should be caught before broad fact
        # sharing, and insults/compliments should be caught before unknown.
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
        """Small fallback extractor for durable facts used in tests and demos."""
        text = player_input.strip()
        lowered = text.lower()
        if "my name is" in lowered:
            # Preserve the user's original capitalization for names.
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
        """Infer trust impact when add_memory is called directly in tests."""
        lowered = memory_text.lower()
        if any(word in lowered for word in ["insult", "useless", "stupid", "awful", "terrible", "worthless"]):
            return -10
        if any(word in lowered for word in ["praised", "thank", "skilled", "great", "excellent", "impressive"]):
            return 5
        return 0

    def _memory_type_from_intent(self, intent: str) -> str:
        """Store memory type metadata so tables are easier to interpret."""
        if intent in {"compliment", "insult"}:
            return "relationship"
        if intent in {"ask_quest", "ask_help"}:
            return "goal"
        return "fact"

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories as prompt text."""
        if not memories:
            return "No relevant memories retrieved."
        return "\n".join(f"- {memory['text']}" for memory in memories)

    def _format_recent_history(self, max_messages: int = 8) -> str:
        """Format only recent dialogue to keep the prompt short."""
        if not self.conversation_history:
            return "No previous conversation in this session."
        # Long-term facts should come from FAISS retrieval. The recent transcript
        # is only short-term context, so the prompt uses the last few messages.
        recent = self.conversation_history[-max_messages:]
        return "\n".join(f"{item['speaker']}: {item['text']}" for item in recent)
