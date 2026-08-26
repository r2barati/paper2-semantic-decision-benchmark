# Related-work audit

**Scope.** Targeted audit completed 2026-08-26 using official proceedings or
publisher records where available. The map below records 34 relevant primary
or authoritative sources; it is a guide for author verification, not a claim
that Paper 2 is first in any area.

## Literature map

| Family | Primary sources screened | What they establish | Boundary for Paper 2 |
|---|---|---|---|
| Decision-focused / predict-then-optimize | Donti et al. (NeurIPS 2017); Elmachtoub et al. (ICML 2020); Elmachtoub & Grigas (Management Science 2022); Mandi et al. (ICML 2022); Rychener et al. (AISTATS 2023); Yamao et al. (UAI 2026) | Downstream decision loss can be preferable to prediction loss for optimization inputs; robust variants address uncertainty and shift | Paper 2 does not introduce a decision-focused loss or optimizer; it studies semantic beliefs entering a fixed sequential controller and reports a controlled accuracy/value dissociation |
| Value-aware / decision-aware dynamics | Farahmand et al. (AISTATS 2017); Lambert et al. (L4DC 2020); Grimm et al. (NeurIPS 2020); Grimm et al. (NeurIPS 2021); Nair et al. (ICML 2020); Lovatto et al. (NeurIPS workshop 2020); Shin et al. (L4DC 2024); Voelcker et al. (ICML 2025) | Dynamics need not be globally predictive if only downstream value matters; model objectives can be mismatched | Paper 2 is not model-based RL and does not claim a new value-aware dynamics objective; these works are direct reasons to avoid that claim |
| Calibration and decisions | Guo et al. (ICML 2017); Zhao et al. (NeurIPS 2021); Gopalan et al. (COLT 2022); Deshpande et al. (AISTATS 2024); Qiao & Zhao (COLT 2025); Hartline et al. (COLT 2026); DeGroot & Fienberg (1983) | Calibration and decision-specific calibration can affect downstream decisions; recent theory studies truthfulness and decision-theoretic calibration | Paper 2 provides an operational empirical instance under a fixed controller; it does not introduce decision calibration theory |
| Value of information / sequential decisions | Blackwell (1953); Howard (1966); Puterman (1994); Arumugam & Van Roy (NeurIPS 2021); Russo & Van Roy (JMLR 2016) | Information value is defined through decisions, and sequential agents may choose what to learn | Paper 2 uses the vocabulary of this literature and must not claim semantic information value as a new principle |
| Operational and inventory RL | Hubbs et al. (OR-Gym 2020); Balaji et al. (ORL 2020); Boute et al. (EJOR 2022); Zhang et al. (AISTATS 2023); Gammelli et al. (L4DC 2023); Farias et al. (ICML 2025); Liu et al. (AISTATS 2025) | Inventory and supply-chain decision systems have controlled RL benchmarks, multi-echelon models, and sample-efficiency analyses | Paper 1 supplies environment lineage; Paper 2 adds the semantic-interpreter/controller separation and consequential sensing analysis |
| NLP, text events, and operational language | Lewis et al. (ACL 2020); Guu et al. (ICML 2020); event extraction and temporal information extraction benchmarks; Quan et al. (2024); contemporary supply-chain LLM studies (2025) | Text can provide structured or retrieved evidence for downstream systems, but most evaluations stop at extraction, QA, forecasting, or task success | Paper 2 evaluates cached classical/LLM interpreters by paired sequential utility, but its language is synthetic and its LLM sample is limited |
| Learning to defer / downstream decision makers | Madras et al. (NeurIPS 2018); Mozannar & Sontag (ICML 2020); Zhao et al. (NeurIPS 2021); Kleinberg et al. (2018) | Upstream predictions can be coupled to a downstream decision-maker and selective action | Paper 2 shares coupling logic but fixes the controller and varies semantic sensing; it is not a defer-to-human method |

The sources above include more than 30 relevant records. The bibliography in
the staged manuscript intentionally cites only the sources needed in the main
text; the author should expand and correct the publication bibliography before
submission, especially for older value-of-information and textual-event work.

## Exact claim boundary

The exact distinction is not “semantic information has value,” “accuracy is not
decision quality,” or “calibration matters.” Those are established ideas in
decision theory, decision-focused learning, and calibration. The bounded claim
is empirical and protocol-level:

> In a fixed, executable sequential operational contract, changing only the
> probabilistic semantic interpreter can produce materially different paired
> utility despite modest or reversed changes in classification accuracy; the
> effect is mediated by calibration, controller response, event mechanism, and
> structural transfer boundaries.

This is a controlled empirical characterization, not a new universal metric,
new decision rule, or causal claim about all operational systems.

## Threats explicitly checked

- **Decision-focused learning:** direct downstream optimization is prior art;
  Paper 2 must present the framework and evidence as an empirical study.
- **Value-aware model learning:** prediction/value mismatch is prior art;
  Paper 2 does not learn a world model or value-aware transition loss.
- **Decision calibration:** downstream decision-maker dependence is prior art;
  Paper 2's calibration result is a benchmark-specific operational finding.
- **Value of information:** decision-dependent information value is prior art;
  OIV/SIVR are framework diagnostics, not new information theory.
- **Inventory RL benchmarks:** the environment is a controlled testbed, not
  the central novelty.
- **LLM operations:** LLM evidence is secondary and cannot support superiority,
  production, or general supply-chain claims.

## Author verification checklist

Before submission, verify title, venue, DOI, and page metadata for every cited
work; add the most relevant 2024--2026 follow-ups; search forward citations from
Donti, Farahmand, Zhao, Boute, and the recent operational-LLM papers; and keep
the wording “in this benchmark” wherever a claim is not supported outside the
frozen environments.

## Verified primary-record links

[Farahmand et al. (AISTATS 2017)](https://proceedings.mlr.press/v54/farahmand17a.html),
[Lambert et al. (L4DC 2020)](https://proceedings.mlr.press/v120/lambert20a.html),
[Grimm et al. (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html),
[Elmachtoub et al. (ICML 2020)](https://proceedings.mlr.press/v119/elmachtoub20a.html),
[Arumugam and Van Roy (NeurIPS 2021)](https://papers.nips.cc/paper_files/paper/2021/hash/517da335fd0ec2f4a25ea139d5494163-Abstract.html),
[Zhao et al. (NeurIPS 2021)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/bbc92a647199b832ec90d7cf57074e9e-Abstract.html),
[Mandi et al. (ICML 2022)](https://proceedings.mlr.press/v162/mandi22a.html),
[Shin et al. (L4DC 2024)](https://proceedings.mlr.press/v242/shin24a.html),
[Voelcker et al. (ICML 2025)](https://proceedings.mlr.press/v267/voelcker25a.html),
[Yamao et al. (UAI 2026)](https://proceedings.mlr.press/v337/yamao26a.html),
[Liu et al. (AISTATS 2025)](https://proceedings.mlr.press/v258/liu25e.html), and
[Farias et al. (ICML 2025)](https://proceedings.mlr.press/v267/farias25a.html).
