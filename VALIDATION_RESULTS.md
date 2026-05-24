# MemoryNPC Validation Results

Validation was executed from the project notebooks on May 24, 2026 using the real `MemoryNPC` backend.

The scores below are controlled results on manually designed validation cases. They are evidence that the system meets the assignment target, not a claim that the agent will be perfect on arbitrary player language.

## Executed Notebooks

- `memorynpc.ipynb`: main report notebook.
- `validation.ipynb`: technical validation notebook with at least 20 cases per major metric.

## Main Metrics

| Metric | Result |
|---|---:|
| Main notebook intent accuracy | 100% |
| Main notebook FAISS memory retrieval success | 100% |
| Main notebook keyword retrieval baseline | 70% |
| Validation notebook intent accuracy | 100% |
| Validation notebook FAISS retrieval success | 100% |
| Validation notebook keyword baseline | 50% |
| Long conversation scenario pass rate | 100% |

## Extended 20x Validation

I added a separate validation suite after reviewing the project from an NLP evaluation perspective. The suite tests every major metric at least 20 times and stores the result files in `results/validation/`.

| Metric | Cases | Result |
|---|---:|---:|
| Intent classification | 30 | 100% |
| Memory extraction selectivity | 22 | 100% |
| FAISS semantic retrieval | 25 | 100% |
| Keyword retrieval baseline | 25 | 88% |
| Trust rule validation | 22 | 100% |
| Advice behavior validation | 22 | 100% |
| Persistence roundtrip | 20 | 100% |
| Full-pipeline trace completeness | 20 | 100% |
| Full-pipeline trust arithmetic | 20 | 100% |
| Full-pipeline automatic response checks | 20 | 100% |

The first extended run found a direct memory-extraction weakness: `extract_memory()` returned `NONE` for some durable relationship and goal utterances. I fixed this by adding the narrow rule-based fallback inside `extract_memory()` when the LLM extractor returns `NONE`. The final extended run above reflects the corrected backend.

## Long Conversation Evaluation

The validation notebook runs three longer scripted conversations:

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

The project now distinguishes two baselines:

- `baseline_agent.py`: a role-only LLM baseline. This can roleplay Elara, but it has no external memory, trust state, persistence, or validation trace.
- Keyword retrieval baseline: a simple word-overlap retriever. This is the quantitative baseline compared against FAISS.

The keyword baseline is intentionally weaker than FAISS because it does not understand semantic similarity. The validation results support the design choice to use embeddings and FAISS:

- Validation notebook FAISS retrieval: 100%
- Validation notebook keyword baseline: 50%
- Main notebook FAISS retrieval: 100%
- Main notebook keyword baseline: 70%

## Validation-Driven Fixes

Running the notebooks revealed an intent-classification edge case: substring matching treated words like `shield` as if they contained the greeting `hi`. The intent fallback was updated to use whole-word matching. Obvious filler statements about weather, clouds, sky, and the moon are now kept as `unknown` instead of being over-interpreted by the LLM fallback.

## Conclusion

The final validation supports the assignment claim that MemoryNPC is more than a normal chatbot. The system exposes and tests intermediate NLP behavior: intent classification, memory extraction, semantic retrieval, trust tracking, advice-following state, response generation, persistence, and trace completeness.
