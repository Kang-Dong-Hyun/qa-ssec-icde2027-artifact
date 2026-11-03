# QA-SSEC: Evaluating Budgeted Telemetry Views

Companion materials for **Evaluating Budgeted Telemetry Views: Separating Record Coverage, Joint Witness Preservation, and Query Fidelity**.

This repository contains analysis code and frozen, public-benchmark-derived outputs for comparing record coverage, joint witness preservation, and deterministic query fidelity. It includes exact finite-panel monotone-comparison certificates and a solver-independent certificate checker.

## Run the stored-output verification

Python **3.8 or later**, standard library only:

```console
python -I -B VERIFY_PUBLIC.py
```

Windows PowerShell wrapper (optional):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\RUN_VERIFY_WINDOWS.ps1
```

The verifier checks the complete file manifest, verifies all **32 directional certificates**, executes the saved-output analyses, and checks **14 regenerated JSON/CSV files byte for byte** against their archived references. Results and execution logs go to `reproduced/`, which is excluded from version control. Run again to validate and reuse a completed verification rather than repeat the calculation.

It makes no network requests, installs no dependencies, and performs no model training/inference, tokenization, packing, or new source-query evaluation. The public-source compiler is not executed by this command.

## Repository map

| Directory | Contents |
|---|---|
| `certificate/` | Exact finite-panel comparison code, immutable witness inputs, 32 directed certificates, summary |
| `objectives/` | Empirical threshold profiles over the recorded witness outcomes |
| `composition/` | Identity-aware AND/OR diagnostics and complete subset tables |
| `mechanism/` | Covariance and COUNT/FRACTION mechanism analyses |
| `count_floor/` | Raw COUNT capacity-floor and residual-omission analysis |
| `fixed_readout/` | Equal-record-count results and paired readout |
| `inputs/` | Development, repeat-transfer and fixed-count saved result archives |
| `source_data/` | Public RCAEval adapter-compiler reference, source-file hashes and protocol |

Nested ZIPs are preserved byte for byte because their digests are used by the analysis contracts. They contain CSV/JSON records, not executable installers or equipment documentation.

## Source provenance

- Public benchmark: [RCAEval](https://huggingface.co/datasets/phamquiluan/RCAEval).
- Recorded dataset revision: `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`.
- Recorded `cases.parquet` SHA256: `c49a288920dbba2e8e724679a14636d5c7eb2b45426bba14007ef79a6c0ab1bb`.
- Development source-file digest reference: `source_data/PINNED_PUBLIC_150_SOURCE_DIGESTS.csv` (150 files / 75 cases).
- Repeat-transfer source digests: `SOURCE_ELIGIBILITY.csv` inside `inputs/NEXT_CASE_TRANSFER_RESULT.zip`.

The repository stores derived evaluations and provenance records, not original Parquet telemetry or model weights. Hash-like case keys refer to benchmark cases; hashing is not represented as an anonymity guarantee. Repeated cases, labels and post-hoc combinations are not additional independent observations.

## Reproduction scope

This release reproduces **stored-output analyses and certificate verification**. It does not itself provide a single executable pipeline from raw telemetry through every original selection/query run, and does not reproduce the manuscript's archived model-training diagnostic. The public adapter compiler is a source reference, not a substitute for those missing execution paths. See `ARTIFACT_SCOPE.md` for the exact coverage.

## Release integrity and attribution

`FILE_MANIFEST_SHA256.csv` lists every release file except itself. `.gitattributes` disables line-ending conversion so Windows Git settings do not change pinned bytes. `.gitignore` admits only the reviewed file paths; generated outputs and accidental additions stay ignored. `RELEASE_AUDIT.json` records the content review and verification scope.

Benchmark attribution and licensing information are in `THIRD_PARTY_NOTICES.md`. No new MIT, Apache, or other reuse license for the authors' code is assigned by the packaging step; publication permission and the choice of a reuse license are separate.
