# GPU smoke gate report — EXP-P2-REASONING-AGENTIC-v1

Overall: PASS

| Gate | Result |
|---|---|
| G1_zero_parse_failures | PASS |
| G2_zero_schema_failures | PASS |
| G3_prob_vectors_sum_to_one | PASS |
| G0_prompt_freeze | PASS |
| G4_retrieval_budgets | PASS |
| G5_leakage_seal | PASS |
| G6_infer_once_per_warning | PASS |
| G7_controller_parity | PASS |
| G8_deterministic_replay | PASS |
| G9_pathwise_hook | PASS |
| G10_checkpoint_resume | PASS |

Fetched beliefs: 30 (6 warnings x 5 arms).
LLM calls: 42. Failures: 0.
GPU: {'torch_cuda': True, 'device_name': 'Tesla T4', 'device_count': 2, 'total_mem_GB': 15.64}.
Resolved: vLLM 0.11.0, transformers 4.57.6, served={'served_models': ['Qwen/Qwen3-8B-AWQ']}.
Prompt SHAs match frozen DEV: True.
Worst same-seed replay diff: 0.00e+00.
