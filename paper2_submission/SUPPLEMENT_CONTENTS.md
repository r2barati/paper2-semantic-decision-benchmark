# Anonymous supplement contents

Recommended minimal package:

- anonymized source code needed for offline verification;
- dependency files and environment instructions;
- frozen manifests, template identifiers, seed lists, and configuration hashes;
- deterministic analysis scripts and generated summary tables needed to map
  manuscript numbers to frozen artifacts;
- tests and the clean offline reproduction README;
- selected small frozen summaries, not the full LLM cache unless required.

Exclude:

- `results/phase9b_robustness/divergent_medium_lead.yaml`;
- `.llm_cache/` and raw provider credentials;
- `.env`, Git metadata, absolute paths, usernames, or author-identifying URLs;
- unnecessary large raw caches and nonessential historical snapshots.

The supplement is a plan only; no external upload is performed in this phase.

