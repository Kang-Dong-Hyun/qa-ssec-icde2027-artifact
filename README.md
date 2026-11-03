# QA-SSEC: Evaluating Budgeted Telemetry Views

Companion materials for **Evaluating Budgeted Telemetry Views: Separating Record
Coverage, Joint Witness Preservation, and Query Fidelity**.

This release candidate extends v1.0.0 with the original historical D4 inputs,
B01/B02 test predictions, fold assignments, tokenizer snapshot, and the original
parameterized training reference. The reported research results are unchanged.

## Verify saved results and original prediction records

Python **3.8+**, standard library only:

```text
python -I -B VERIFY_PUBLIC.py
```

On Windows, the included wrapper is optional:

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\RUN_VERIFY_WINDOWS.ps1
```

The verifier checks the release manifest, all 32 directional proof certificates,
14 original regenerated analysis files, and two historical input/prediction
verification outputs. The latter link 40 original B01/B02 test files (600
repeated-setting rows) to the targets, test-fold assignments, 160 fold-level test
metrics, 12 paired summaries, and Table X. Results go to `reproduced/`, excluded
from Git. A completed invocation can be reused after integrity checks.

Verification uses no network or package installation. It does not run a model,
train, tokenize, pack, or evaluate new raw-source queries. The training reference
is opt-in and is not imported by the verifier.

## Repository map

| Directory | Contents |
|---|---|
| `certificate/` | Exact finite-panel comparisons and 32 solver-independent proof objects |
| `objectives/` | Complete empirical attainment profiles |
| `composition/` | Named-case AND/OR and subset diagnostics |
| `mechanism/` | Covariance and same-predicate COUNT/FRACTION analyses |
| `count_floor/` | Raw COUNT capacity-floor and residual-omission accounting |
| `fixed_readout/` | Equal-count results and paired readout |
| `inputs/` | Original development, repeat-transfer, and fixed-count results |
| `source_data/` | Original public development compiler reference and source digests |
| `pipeline_reference/` | Frozen computation definitions, 300 source-file references, and verified case mapping |
| `historical/` | D4 original summaries, frozen inputs, test predictions, checkers, and opt-in training reference |
| `licenses/` | License text for the bundled model tokenizer/configuration |

Nested ZIP members are kept byte for byte where their digests are part of the
experiment record. Do not unpack them in place or rename them.

## Source provenance

Public benchmark: https://huggingface.co/datasets/phamquiluan/RCAEval

Pinned dataset revision: `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e`

Pinned `cases.parquet` SHA256:
`c49a288920dbba2e8e724679a14636d5c7eb2b45426bba14007ef79a6c0ab1bb`

Pinned model: `HuggingFaceTB/SmolLM2-135M`, revision
`28e66ca6931668447a3bac213f23d990ad3b0e2b`.

All 75 development case keys are now linked to the original record identifiers
and saved query results. The mapping uses SHA256(record_uid)[:16]; the original
record_uid itself is SHA256(public case name). These are separate identifiers.
The source cases, metrics and service names are from the public benchmark.

The historical inputs preserve all four original representation files. B03
contains already-frozen pseudonyms, not a secret key or its private generation
material. The accompanying checker reproduces the saved B01/B02 prediction
metrics without using any trained model. See `historical/README.md` for the
separately invoked original model-training path.

## Scope and attribution

`ARTIFACT_SCOPE.md` specifies the available verification and execution paths.
`THIRD_PARTY_NOTICES.md` records the upstream dataset/model notices. Model weights
are obtained separately at their pinned official revision. No new license is
assigned to repository-authored code by this packaging step.

The Word/PDF manuscript, author correspondence, internal collection reports,
private machine paths, company equipment documentation and response procedures
are not part of this repository candidate. Creating this local package does not
change the already published v1.0.0 tag or publish a new GitHub release.
