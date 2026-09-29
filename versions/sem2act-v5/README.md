# Sem2Act v5

This is an isolated confirmation overlay derived from sem2act-v4. The v4
working version is not modified by v5 execution.

The frozen package has four evidence blocks:

1. a 240-query untouched lockbox with 20 paired seeds;
2. official Llama 3.1 and Mistral 7B v0.3 consumer families on the headline subset;
3. CausalOptimizer and BeliefBaseStock controller replay plus the explicitly
   evidence-insensitive WorstCaseBaseStock safety control;
4. entropy-selective consumption and a persistence/reset horizon diagnostic.

The primary persistence arm is Qwen3-8B, Rerank versus NoInfo, C1,
CausalOptimizer, with evidence released at t=12. Its signed estimands are
defined in protocol/confirmation.yaml and used by the deterministic replay.

Execution is fail-closed. A provenance validation failure invalidates the
lockbox. A model preflight failure excludes that model without substitution.
After protocol_hash is frozen, null, mixed, contradictory, and failed results
are evidence and do not trigger redesign.

The local execution chain is:

1. fetch and verify the private qrel-free reranker output;
2. assemble and stage the three qrel-free model bundles;
3. run the Qwen primary job and the Llama/Mistral cross-family jobs only after
   their private dataset versions and model revisions are recorded;
4. install only verifier-accepted outputs;
5. assemble beliefs, run the deterministic CPU replay, and execute the
   one-pass analysis;
6. freeze manifests before editing the isolated v5 paper snapshot.

The earlier GPU Kaggle and Lightning attempts are retained as infrastructure
history. The current execution amendments are scoped by stage: Moon has a
reranker-only CPU lock with a gated 16-thread profile; Kaggle v3 and Colab v1
have separate GPU runtime locks. Their exact account, image, and STOP rules
are in the repository RUNBOOK and COMPUTE routing index. The frozen scientific
protocol is unchanged.

The broader CPU consumer backend remains blocked before runtime freeze.
Qwen AWQ-to-GGUF conversion failed, and the pinned Llama source was not
mounted. Mistral source assembly is verified, but no CPU consumer result stage
is authorized. Do not substitute model, quantization, or source. The fixed
non-lockbox fixture is 'fixtures/runtime_smoke_v1.json'.

The Kaggle CPU preflight requires a user secret named 'HF_TOKEN'; its value is
never stored in the repository or canary package. The canary package embeds
only the hash-checked qrel-free config and smoke fixture because Kaggle script
runtimes may omit auxiliary package files. A missing secret, model conversion
failure, schema failure, or nondeterministic repeat is recorded as an excluded
preflight failure and blocks the CPU runtime freeze.
