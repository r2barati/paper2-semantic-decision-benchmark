# Kaggle GPU smoke observation

- **Observed:** 2026-09-28, as reported by the operator to Codex.
- **Account:** `rezabarati2`.
- **Hardware:** 2× NVIDIA T4, 15 GB VRAM each.
- **CUDA:** 12.8 as reported by the smoke test.
- **Quota:** approximately 30 GPU-hours per week.
- **Evidence class:** operator-reported infrastructure smoke, not a Sem2Act V5 scientific job and not a fresh V5 account/quota/input/runtime preflight.
- **Policy:** before an authorized V5 Kaggle pilot, OpenCode must freshly verify the active account, staged input access, positive quota, exact locked PyTorch/CUDA runtime, and one-T4 job shape. The V5 launcher requests one T4 despite the reported two-device entitlement.
- **Secret handling:** no token, credential value, or credential file was included in this record.
