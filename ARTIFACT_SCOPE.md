# Artifact scope and execution paths

## Saved-result verification (executed during packaging)

| Path | Provided material | Check |
|---|---|---|
| D1--D3 stored outputs | Original result ZIPs in `inputs/` | Existing input digests |
| Finite witness comparisons | Boolean input panels, certificates and checker | 32 directed proofs |
| Mechanism, objectives, composition, COUNT floor and equal-count readout | Original programs and saved input/results | 14 original outputs, byte for byte |
| Historical aggregates D4 | 80 fold rows, 16 representation/cap aggregates and 12 paired summaries | 128 mean/SD values and paired arithmetic |
| Historical prediction records | 40 B01/B02 test CSVs, 600 rows, targets and the original fold registry | 160 test metrics, 12 paired comparisons, all four Table X rows |
| Input linkage | Original adapter/representations, public case map and frozen tokenizer hashes | 75 original case-key joins and B01/B02 value identity |

The verification entry point reads frozen data and does not train, infer,
tokenize, pack or execute new raw-source queries. All experiment row counts
retain their original repeated-setting interpretation.

## Historical training path (provided, not rerun during packaging)

`historical/TRAIN_TRACKB_REFERENCE.py` is the original training code, unchanged.
The frozen input ZIP supplies the four representation files and original
five-fold registry expected by its CLI. The exact tokenizer/configuration files,
model repository revision, model-weight SHA256 and recorded dependency versions
are included. `historical/README.md` documents the opt-in invocation using a new
scratch output directory. Model weights are obtained from the official pinned
repository rather than redistributed here.

D4 B03/B04 fold summaries and frozen representations are supplied. Per-case
prediction files recovered for this release cover B01/B02, the arms in Table X.
No new model-training run or cross-machine numerical training parity is claimed
by the saved-output verification.

## Raw telemetry / primary selection path

The public development adapter compiler in `source_data/` expects the separately
obtained 150 development source files, a pinned index and digests. The supplied
public index and recovered adapter permit the original development identities
to be connected without inventing substitute keys. `pipeline_reference/` provides
the frozen packer, ordering, seven-query definitions and all 300 original source
file references for the two panels.

The release does not yet provide a single, validated raw-source-to-all-primary-
results orchestrator for both panels. Reference functions and pinned inputs must
not be described as a tested end-to-end primary pipeline. Original raw telemetry
files and model weights are externally obtained. This is a release-coverage
statement, not an additional research experiment or a change to reported results.

## Publication state

This local candidate has not itself been uploaded by the packaging scripts.
Keep the published v1.0.0 unchanged; a subsequent release can identify this
expanded snapshot. The manuscript's URL and contents statement must identify
the version that has actually been made available.
