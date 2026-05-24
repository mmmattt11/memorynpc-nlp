"""Validation suite for the MemoryNPC NLP assignment.

Each metric is tested with at least 20 cases and written to JSON/CSV result
files so the results can be inspected outside the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evaluation_helpers import automatic_response_checks, keyword_retrieve, retrieval_success
from npc_agent import MemoryNPC


RESULT_DIR = Path("results/validation")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


INTENT_CASES = [
    ("Hello Elara", "greeting"),
    ("Hi there blacksmith", "greeting"),
    ("Good morning", "greeting"),
    ("Greetings from the north road", "greeting"),
    ("Goodbye for now", "goodbye"),
    ("Farewell Elara", "goodbye"),
    ("See you later", "goodbye"),
    ("Can you give me a quest?", "ask_quest"),
    ("Do you have a mission for me?", "ask_quest"),
    ("Any job for me today?", "ask_quest"),
    ("My name is Matt", "share_fact"),
    ("I lost a silver dagger", "share_fact"),
    ("I want to find my brother", "share_fact"),
    ("I like heavy axes", "share_fact"),
    ("Do you remember my name?", "ask_memory"),
    ("What did I lose earlier?", "ask_memory"),
    ("Can you recall what I told you?", "ask_memory"),
    ("You are a skilled blacksmith", "compliment"),
    ("Thanks, that was great", "compliment"),
    ("That repair was excellent", "compliment"),
    ("You are useless", "insult"),
    ("That was a terrible repair", "insult"),
    ("I hate your work", "insult"),
    ("Can you help me fix my shield?", "ask_help"),
    ("Please assist me with this blade", "ask_help"),
    ("Can you forge a hinge?", "ask_help"),
    ("The sky is cloudy", "unknown"),
    ("Maybe the moon is made of iron", "unknown"),
    ("The rain sounds loud", "unknown"),
    ("Clouds pass over the village", "unknown"),
]


MEMORY_EXTRACTION_CASES = [
    ("My name is Matt", True),
    ("My name is Lena", True),
    ("I lost a sword near the forest", True),
    ("I lost a copper ring at the bridge", True),
    ("I want to find my brother", True),
    ("I need my shield repaired before sunset", True),
    ("I like heavy axes", True),
    ("I found a strange map", True),
    ("You are a skilled blacksmith", True),
    ("You are useless", True),
    ("I promised to help the miller", True),
    ("I am afraid of cave spiders", True),
    ("I need a quiet road north", True),
    ("I want a dagger made of silver", True),
    ("I found an old key in the chapel", True),
    ("I helped the baker yesterday", True),
    ("Hello there", False),
    ("Goodbye", False),
    ("The weather is gray today", False),
    ("The moon looks bright", False),
    ("Clouds are moving slowly", False),
    ("It is noisy today", False),
]


RETRIEVAL_MEMORIES = [
    ("name_matt", "The player's name is Matt.", "Matt"),
    ("lost_sword", "The player lost a sword near the forest.", "sword"),
    ("miller_wheel", "The player helped the village miller repair a broken wheel.", "miller"),
    ("likes_axes", "The player prefers heavy axes over light daggers.", "axes"),
    ("brother_goal", "The player wants to find their missing brother.", "brother"),
    ("insult_useless", "The player insulted Elara and called her useless.", "useless"),
    ("praise_skilled", "The player praised Elara as a skilled blacksmith.", "skilled"),
    ("lost_ring", "The player lost a copper ring at the old bridge.", "ring"),
    ("shield_repair", "The player needs a shield repaired before sunset.", "shield"),
    ("spider_fear", "The player is afraid of giant cave spiders.", "spiders"),
    ("old_map", "The player found an old map under the chapel floor.", "map"),
    ("baker_favor", "The player promised to help the baker carry flour.", "baker"),
    ("silver_dagger", "The player wants a silver dagger for night patrol.", "dagger"),
    ("quiet_roads", "The player prefers quiet roads over crowded markets.", "roads"),
    ("north_gate", "The player plans to leave through the north gate.", "north"),
    ("broken_hinge", "The player needs a broken stable hinge forged.", "hinge"),
    ("lost_horse", "The player lost a brown horse near the river.", "horse"),
    ("blue_cloak", "The player wears a blue cloak with torn sleeves.", "cloak"),
    ("old_key", "The player found an old iron key in the chapel.", "key"),
    ("torch_advice", "Elara advised the player to bring a torch into the forest.", "torch"),
    ("wolf_fear", "The player is afraid of wolves near the ridge.", "wolves"),
    ("merchant_debt", "The player owes a merchant three silver coins.", "merchant"),
    ("broken_axe", "The player's axe head cracked during training.", "axe"),
    ("sister_goal", "The player is searching for their missing sister.", "sister"),
    ("forest_path", "The player knows a hidden path behind the forest shrine.", "path"),
]

RETRIEVAL_QUERIES = [
    ("What is my name?", "Matt"),
    ("Which weapon did I misplace near the trees?", "sword"),
    ("Who did I help with a broken wheel?", "miller"),
    ("What kind of weapon do I prefer?", "axes"),
    ("Who am I searching for in my family?", "brother"),
    ("Did I offend Elara before?", "useless"),
    ("Did I compliment the blacksmith?", "skilled"),
    ("What did I lose at the bridge?", "ring"),
    ("What equipment needs repair before sunset?", "shield"),
    ("What creature scares me in caves?", "spiders"),
    ("What did I discover below the chapel?", "map"),
    ("Who did I promise to help with flour?", "baker"),
    ("What blade do I want for patrol?", "dagger"),
    ("What type of roads do I prefer?", "roads"),
    ("Which gate do I plan to use?", "north"),
    ("What stable part needs forging?", "hinge"),
    ("What animal did I lose by the river?", "horse"),
    ("What color is my torn cloak?", "cloak"),
    ("What did I find in the chapel?", "key"),
    ("What did Elara tell me to bring into the woods?", "torch"),
    ("What animals near the ridge frighten me?", "wolves"),
    ("Who do I owe silver coins to?", "merchant"),
    ("What tool cracked during training?", "axe"),
    ("Which sibling am I looking for besides my brother?", "sister"),
    ("What route do I know behind the shrine?", "path"),
]


TRUST_CASES = [
    ("greeting", 50, 51),
    ("share_fact", 51, 52),
    ("compliment", 52, 57),
    ("insult", 57, 47),
    ("ask_help", 47, 47),
    ("ask_quest", 47, 47),
    ("ask_memory", 47, 47),
    ("goodbye", 47, 47),
    ("unknown", 47, 47),
    ("compliment", 47, 52),
    ("compliment", 52, 57),
    ("insult", 57, 47),
    ("insult", 47, 37),
    ("share_fact", 37, 38),
    ("greeting", 38, 39),
    ("ask_help", 39, 39),
    ("compliment", 39, 44),
    ("unknown", 44, 44),
    ("goodbye", 44, 44),
    ("insult", 44, 34),
    ("compliment", 34, 39),
    ("share_fact", 39, 40),
]

ADVICE_CASES = [
    ("I will follow your advice and bring a torch", "followed_advice", 3),
    ("I will bring a torch", "followed_advice", 3),
    ("I will bring light", "followed_advice", 3),
    ("I will not go alone", "followed_advice", 3),
    ("I will be careful", "followed_advice", 3),
    ("I will ask around", "followed_advice", 3),
    ("I will prepare first", "followed_advice", 3),
    ("I will repair it first", "followed_advice", 3),
    ("I will take your advice", "followed_advice", 3),
    ("I will do that", "followed_advice", 3),
    ("I will go alone anyway", "ignored_advice", -8),
    ("I will ignore that", "ignored_advice", -8),
    ("I will do the opposite", "ignored_advice", -8),
    ("I will be reckless", "ignored_advice", -8),
    ("I will go without help", "ignored_advice", -8),
    ("I will go without a torch", "ignored_advice", -8),
    ("I will go without light", "ignored_advice", -8),
    ("I do not care", "ignored_advice", -8),
    ("No need for caution", "ignored_advice", -8),
    ("I am not going to prepare", "ignored_advice", -8),
    ("I found a coin", "unrelated", 0),
    ("The road is quiet", "unrelated", 0),
]

FULL_PIPELINE_SCRIPT = [
    "Hello Elara",
    "My name is Matt",
    "I lost my sword near the forest",
    "Do you remember what I lost?",
    "You are a very skilled blacksmith",
    "Can you help me recover my sword?",
    "I will go alone anyway",
    "Can you help me recover my sword?",
    "I will follow your advice and bring a torch",
    "I want to find my brother",
    "Do you remember who I am looking for?",
    "You are useless",
    "I lost a copper ring at the old bridge",
    "What did I lose at the bridge?",
    "I need my shield repaired before sunset",
    "Can you help me fix my shield?",
    "I will prepare first",
    "The sky is cloudy",
    "Do you remember the name of my horse?",
    "Goodbye",
]


def pct(value: float) -> str:
    return f"{value:.0%}"


def evaluate_intent() -> pd.DataFrame:
    agent = MemoryNPC()
    rows = []
    for text, expected in INTENT_CASES:
        predicted = agent.classify_intent(text)
        rows.append(
            {
                "input": text,
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
            }
        )
    return pd.DataFrame(rows)


def evaluate_memory_extraction() -> pd.DataFrame:
    agent = MemoryNPC()
    rows = []
    for text, should_save in MEMORY_EXTRACTION_CASES:
        memory = agent.extract_memory(text)
        saved = memory.strip().upper() != "NONE"
        rows.append(
            {
                "input": text,
                "expected_saved": should_save,
                "extracted_memory": memory,
                "actual_saved": saved,
                "correct": saved == should_save,
            }
        )
    return pd.DataFrame(rows)


def seed_retrieval_agent() -> MemoryNPC:
    agent = MemoryNPC()
    for _, memory_text, _ in RETRIEVAL_MEMORIES:
        agent.add_memory(memory_text)
    return agent


def evaluate_retrieval() -> pd.DataFrame:
    agent = seed_retrieval_agent()
    rows = []
    for query, expected_keyword in RETRIEVAL_QUERIES:
        top_memories = [memory["text"] for memory in agent.retrieve_memories(query, k=3)]
        rows.append(
            {
                "query": query,
                "expected_keyword": expected_keyword,
                "top_3_memories": top_memories,
                "success": retrieval_success(top_memories, expected_keyword),
            }
        )
    return pd.DataFrame(rows)


def evaluate_keyword_baseline() -> pd.DataFrame:
    memory_texts = [memory_text for _, memory_text, _ in RETRIEVAL_MEMORIES]
    rows = []
    for query, expected_keyword in RETRIEVAL_QUERIES:
        top_memories = keyword_retrieve(query, memory_texts, k=3)
        rows.append(
            {
                "query": query,
                "expected_keyword": expected_keyword,
                "keyword_top_3": top_memories,
                "success": retrieval_success(top_memories, expected_keyword),
            }
        )
    return pd.DataFrame(rows)


def evaluate_trust_rules() -> pd.DataFrame:
    agent = MemoryNPC()
    rows = []
    for intent, expected_before, expected_after in TRUST_CASES:
        before = agent.trust_score
        after = agent.update_trust(intent)
        rows.append(
            {
                "intent": intent,
                "expected_before": expected_before,
                "actual_before": before,
                "expected_after": expected_after,
                "actual_after": after,
                "correct": before == expected_before and after == expected_after,
            }
        )
    return pd.DataFrame(rows)


def evaluate_advice_behavior() -> pd.DataFrame:
    rows = []
    for text, expected_status, expected_delta in ADVICE_CASES:
        agent = MemoryNPC()
        agent.pending_advice = {
            "topic": "lost_item_recovery",
            "summary": "Recover the lost item carefully; bring light, ask around, and do not go alone.",
        }
        result = agent._evaluate_advice_following(text)
        rows.append(
            {
                "input": text,
                "expected_status": expected_status,
                "actual_status": result["advice_status"],
                "expected_delta": expected_delta,
                "actual_delta": result["advice_delta"],
                "correct": result["advice_status"] == expected_status and result["advice_delta"] == expected_delta,
            }
        )
    return pd.DataFrame(rows)


def evaluate_persistence() -> pd.DataFrame:
    rows = []
    for index, (memory_id, memory_text, expected_keyword) in enumerate(RETRIEVAL_MEMORIES[:20], start=1):
        agent = MemoryNPC()
        agent.trust_score = 40 + index
        agent.turn_number = index
        agent.add_memory(memory_text)
        agent.conversation_history.append({"speaker": "Player", "text": f"seed {index}"})
        state_json = agent.export_state_json()

        restored = MemoryNPC()
        restored.import_state_json(state_json)
        retrieved = [memory["text"] for memory in restored.retrieve_memories(memory_text, k=1)]
        rows.append(
            {
                "case": index,
                "memory_id": memory_id,
                "expected_keyword": expected_keyword,
                "trust_preserved": restored.trust_score == agent.trust_score,
                "turn_preserved": restored.turn_number == agent.turn_number,
                "memory_count_preserved": len(restored.memories) == len(agent.memories),
                "retrieval_after_load": retrieved,
                "retrieval_success": retrieval_success(retrieved, expected_keyword),
            }
        )
    df = pd.DataFrame(rows)
    df["correct"] = (
        df["trust_preserved"]
        & df["turn_preserved"]
        & df["memory_count_preserved"]
        & df["retrieval_success"]
    )
    return df


def evaluate_full_pipeline_trace() -> Dict[str, Any]:
    agent = MemoryNPC()
    rows = []
    for message in FULL_PIPELINE_SCRIPT:
        event = agent.generate_npc_response(message)
        checks = automatic_response_checks(event)
        rows.append({**event, **checks})
    trace_df = pd.DataFrame(rows)
    return {"agent": agent, "trace": trace_df}


def dataframe_rate(df: pd.DataFrame, column: str = "correct") -> float:
    return float(df[column].mean()) if len(df) else 0.0


def run_validation() -> Dict[str, Any]:
    intent_df = evaluate_intent()
    memory_df = evaluate_memory_extraction()
    retrieval_df = evaluate_retrieval()
    keyword_df = evaluate_keyword_baseline()
    trust_df = evaluate_trust_rules()
    advice_df = evaluate_advice_behavior()
    persistence_df = evaluate_persistence()
    full_pipeline = evaluate_full_pipeline_trace()
    trace_df = full_pipeline["trace"]

    metric_tables = {
        "intent": intent_df,
        "memory_extraction": memory_df,
        "faiss_retrieval": retrieval_df,
        "keyword_baseline": keyword_df,
        "trust_rules": trust_df,
        "advice_behavior": advice_df,
        "persistence": persistence_df,
        "full_pipeline_trace": trace_df,
    }

    summary_rows = [
        {
            "metric": "Intent classification",
            "cases": len(intent_df),
            "success_rate": dataframe_rate(intent_df),
            "target": ">= 80%",
        },
        {
            "metric": "Memory extraction selectivity",
            "cases": len(memory_df),
            "success_rate": dataframe_rate(memory_df),
            "target": ">= 80%",
        },
        {
            "metric": "FAISS semantic retrieval",
            "cases": len(retrieval_df),
            "success_rate": dataframe_rate(retrieval_df, "success"),
            "target": ">= 80%",
        },
        {
            "metric": "Keyword retrieval baseline",
            "cases": len(keyword_df),
            "success_rate": dataframe_rate(keyword_df, "success"),
            "target": "comparison baseline",
        },
        {
            "metric": "Trust rule validation",
            "cases": len(trust_df),
            "success_rate": dataframe_rate(trust_df),
            "target": "100% deterministic",
        },
        {
            "metric": "Advice behavior validation",
            "cases": len(advice_df),
            "success_rate": dataframe_rate(advice_df),
            "target": "100% deterministic",
        },
        {
            "metric": "Persistence roundtrip",
            "cases": len(persistence_df),
            "success_rate": dataframe_rate(persistence_df),
            "target": "100% deterministic",
        },
        {
            "metric": "Full-pipeline trace completeness",
            "cases": len(trace_df),
            "success_rate": float(trace_df["trace_complete"].mean()),
            "target": "100%",
        },
        {
            "metric": "Full-pipeline trust arithmetic",
            "cases": len(trace_df),
            "success_rate": float(trace_df["trust_delta_matches"].mean()),
            "target": "100%",
        },
        {
            "metric": "Full-pipeline automatic response checks",
            "cases": len(trace_df),
            "success_rate": float(trace_df["all_automatic_checks_pass"].mean()),
            "target": ">= 80%",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df["success_rate_percent"] = summary_df["success_rate"].map(pct)

    for name, df in metric_tables.items():
        df.to_csv(RESULT_DIR / f"{name}.csv", index=False)

    result = {
        "summary": summary_df,
        "tables": metric_tables,
        "result_dir": str(RESULT_DIR),
    }

    json_payload = {
        "summary": summary_df.to_dict(orient="records"),
        "tables": {
            name: json.loads(df.to_json(orient="records"))
            for name, df in metric_tables.items()
        },
    }
    (RESULT_DIR / "validation_results.json").write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )
    summary_df.to_csv(RESULT_DIR / "summary.csv", index=False)
    return result


if __name__ == "__main__":
    validation = run_validation()
    print(validation["summary"].to_string(index=False))
    print(f"\nWrote results to {validation['result_dir']}")
