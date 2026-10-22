# Stage-2 persona data (optional)

`*.jsonl` files, one record per line:

```json
{"persona": {"age": "34", "occupation": "nurse", "political": 30, "openness": 70, "commStyle": "warm"},
 "scenario": {"domain": "political", "text": "Should the city expand bus service?"},
 "response": "I'd support it...", "source": "my-survey-2026"}
```

Every record is validated with the same rules as the UI (`twin/personas.py`, `twin/scenarios.py`).
Invalid records are skipped and counted on the Learning page. They are never silently fixed.
Records that use a benchmark scenario text are dropped so the benchmark stays held out.
