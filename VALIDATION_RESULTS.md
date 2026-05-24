# MemoryNPC Validation Results

Validation was executed from the project notebooks on May 24, 2026 using the real `MemoryNPC` backend.

## Executed Notebooks

- `notebook.ipynb`: main report notebook, executed end to end.
- `validation_lab.ipynb`: technical validation lab, executed end to end.

## Main Metrics

| Metric | Result |
|---|---:|
| Main notebook intent accuracy | 100% |
| Main notebook FAISS memory retrieval success | 100% |
| Main notebook keyword retrieval baseline | 70% |
| Validation lab intent accuracy | 100% |
| Validation lab FAISS retrieval success | 100% |
| Validation lab keyword baseline | 50% |
| Long conversation scenario pass rate | 100% |

## Long Conversation Evaluation

The validation lab runs three longer scripted conversations:

| Scenario | Turns | Stored memories | Final trust | Advice followed | Advice ignored | Passed |
|---|---:|---:|---:|---:|---:|---|
| cooperative_memory_player | 12 | 6 | 65 | 2 | 0 | yes |
| defiant_low_trust_player | 11 | 3 | 0 | 0 | 3 | yes |
| mixed_memory_stress_player | 14 | 5 | 66 | 2 | 0 | yes |

Each scenario checks whether:

- all turns are logged in the validation trace;
- at least three durable memories are stored;
- expected memory retrieval checks pass;
- advice following or advice ignoring changes trust as expected;
- final trust moves in the expected direction.

## Baseline

The keyword baseline uses simple word overlap. It is intentionally weaker than FAISS because it does not understand semantic similarity. The validation results support the design choice to use embeddings and FAISS:

- Validation lab FAISS retrieval: 100%
- Validation lab keyword baseline: 50%
- Main notebook FAISS retrieval: 100%
- Main notebook keyword baseline: 70%

## Validation-Driven Fixes

Running the notebooks revealed an intent-classification edge case: substring matching treated words like `shield` as if they contained the greeting `hi`. The intent fallback was updated to use whole-word matching. Obvious filler statements about weather, clouds, sky, and the moon are now kept as `unknown` instead of being over-interpreted by the LLM fallback.

## Conclusion

The final validation supports the assignment claim that MemoryNPC is more than a normal chatbot. The system exposes and tests intermediate NLP behavior: intent classification, memory extraction, semantic retrieval, trust tracking, advice-following state, response generation, and trace completeness.
