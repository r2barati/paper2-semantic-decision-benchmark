# V3 FINAL freeze manifest — benchmark development CLOSED 2026-09-14
# Any change after this point is a protocol violation, not a revision.

corpus_sha256 (2026-09-14, 64-node build):
  data/v3/corpus.jsonl: fb6228f32db37b72
  data/v3/queries.jsonl: 096ce514c19f94c6
  data/v3/qrels/dev.tsv: 98e0fb45a16599d4
  data/v3/qrels/test.tsv: 5e1192274ca007cf
  data/v3/splits/splits.json: 4e4d6b346a3f845e
  configs/v3/qwen_instruction.txt: 3438a63a479703e0
integrity_tests: tests/test_v3full.py + tests/test_v3proto.py — 16 passed at freeze
frozen_config_pins:
  retrieval: configs/v3/RETRIEVAL_FREEZE_FINAL.md   # BM25 -> Qwen3-dense-instruct -> RRF k=120 -> rerank top-30
  metric: configs/v3/metrics.md                     # trec_eval nDCG everywhere
  instruction: configs/v3/qwen_instruction.txt
  models: configs/v3/models.yaml                    # gpt-4o-2024-11-20 + Qwen3-8B-AWQ@4da05a8e
  prompts: configs/v3/prompts_freeze.json           # C1 66e6890b, C3 5966cadd, tau 0.5
  tuning_scope: configs/v3/tuning_scope.yaml        # SPENT (only k=3/5 reporting remains)
  dataset_rules: configs/v3/dataset_full.yaml       # 64-node amendment included
test_seal:
  test_qrels_seen_by: nobody (kernels hard-abort on test.tsv; local test metrics: none)
  test_metrics_computed: false
  seal_lift_condition: main frozen experiment analysis ONLY
human_validation: MANDATORY SUBMISSION GATE (no external annotators available in-build;
  blind 60-pair agent audit 0.82/0.90 + independent-grader kappa 0.874 on record)
