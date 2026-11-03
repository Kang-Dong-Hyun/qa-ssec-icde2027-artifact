# Artifact scope and result linkage

## Included and executable from the saved inputs

| Verification job | Reads | Recreated / checked output |
|---|---|---|
| Proof-only verification | `certificate/INPUT_EVIDENCE.zip`, `certificate/CERTIFICATES.json` | 32 directed proofs, without rerunning the matching optimizer |
| Certificate generation | `certificate/INPUT_EVIDENCE.zip` | `CERTIFICATES.json`, `SUMMARY.csv` |
| Mechanism readout | `mechanism/INPUT_RESULTS.zip` | `RESULTS.json`, `ANALYSIS_TABLES.csv` |
| Requirement profiles | `objectives/INPUT_EVIDENCE.zip` | `RESULTS.json`, `SEED_COUNTS.csv`, `THRESHOLD_PROFILE.csv` |
| Named-case composition | `composition/INPUT_EVIDENCE.zip` | `RESULTS.json`, `GROUP_PROFILE.csv`, `SUBSET_RESULTS.csv` |
| COUNT decomposition | `inputs/*.zip` | `COUNT_ERROR_COMPONENTS.csv`, `CALCULATION_RECORD.json` |
| Fixed-count readout | `fixed_readout/07_SOURCE_RESULTS.zip` | `04_ANALYSIS_SUMMARY.json`, `03_COMPARISON_TABLES.csv` |

The last six jobs reproduce 14 stored outputs. They do not generate new source observations or increase the independent sample size.

## Source reference

`source_data/PUBLIC_ORIGIN_COMPILER_V1.py` describes the 75-case development adapter. Its `build` mode takes the pinned public index, the separately downloaded 150 public source files and the digest manifest. NumPy, pandas and a Parquet reader are required for that separate mode. Those dependencies are not required by `VERIFY_PUBLIC.py`. Its optional `verify` mode expects an author-side frozen reference that is not included here.

## Outside this release

- A complete raw-data acquisition, tokenizer, packing and original-query execution pipeline.
- Original telemetry/Parquet files, source model weights and archived model-training reproduction.
- Manuscript Word/PDF drafts and author correspondence.

The upload URL establishes access to this release, not completion of the manuscript's entire raw-data or model-retraining chain. The submitted manuscript's artifact statement must reflect this coverage.
