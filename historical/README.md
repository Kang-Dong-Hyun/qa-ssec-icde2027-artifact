# Historical experiment D4

D4 connects Table X to original frozen input files and prediction-level outputs.

## Files and evidence path

- `TRACKB_PER_FOLD.csv`: the original 80 fold-level result rows.
- `TRACKB_ALL_ARMS.csv`: the original 16 representation/cap aggregates.
- `TRACKB_EXACT_COMPACT.csv`: the B01/B02 subset.
- `TRACKB_PAIRED_COUNTS.csv`: 12 paired service/fault/joint comparisons.
- `FROZEN_INPUTS.zip`: 54 original data files plus one derived, verified case map.
  It contains a 75-case/8,299-metric development adapter; targets and the original
  375-row five-fold registry; 75 frozen representation inputs for each of B01--B04;
  the pinned tokenizer/configuration files and public metadata index; and 40
  B01/B02 test-prediction CSVs comprising 600 repeated-setting rows.
- `INPUT_MANIFEST.csv`: SHA256 and byte count for every archive member.
- `CHECK_HISTORICAL.py`: fold/aggregate arithmetic and paired-count checks.
- `CHECK_RECOVERED_INPUTS.py`: source/UID joins, frozen representation identity,
  all 40 test folds, correctness flags, 160 stored test metrics, 12 paired
  comparisons and Table X, using the original predictions without model execution.
- `TRAIN_TRACKB_REFERENCE.py`: the original training implementation, preserved
  byte for byte. It is never executed by the verification entry point.
- `PINNED_MODEL_AND_ENV.json` and `RECORDED_ENVIRONMENT.txt`: recorded versions,
  model revision and weight/tokenizer hashes.

The `record_uid` is SHA256(public case name). The 16-character development key
used by the later selection study is SHA256(record_uid)[:16]. All 75 links are
checked against the stored development query archive. The two hashes have
different roles and must not be substituted for each other.

## Verification without training

From the repository root:

```text
python -I -B VERIFY_PUBLIC.py
```

For the historical input/prediction layer only:

```text
python -I -B historical/CHECK_RECOVERED_INPUTS.py
```

These commands read frozen files. They do not infer, train, retokenize, run
new source queries, download or install packages.

## Explicit original training path (optional, not a verification command)

The original runner uses 4 representations x 4 chunk caps x 5 folds = 80 jobs,
seed 20260915, one epoch, and the parameterized paths shown below. It requires
CUDA/BF16 and the recorded Python dependencies. The frozen B03 pseudonyms are
provided as original input bytes; no secret key is included or required to
train from these inputs. B03/B04 per-case prediction files are not supplied;
their original fold-level summaries are provided.

1. Expand the verified inputs to a new scratch directory:

```text
python -I -B historical/PREPARE_INPUTS.py --output scratch/historical_inputs
```

2. Obtain `model.safetensors` from the official model repository at the exact
revision in `PINNED_MODEL_AND_ENV.json`, and place it alongside the extracted
`model/config.json` and tokenizer files. The reference weight SHA256 is:

```text
80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
```

Model repository: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
Revision: `28e66ca6931668447a3bac213f23d990ad3b0e2b`

3. Only when deliberately running a historical retraining reproduction, invoke:

```text
python historical/TRAIN_TRACKB_REFERENCE.py --model-root scratch/historical_inputs/model --representation-root scratch/historical_inputs/representations --split-registry scratch/historical_inputs/development/folds.csv --output-root scratch/historical_retrain_new
```

Use a new output directory. The original runner can remove an incomplete job
subdirectory inside its chosen output root before restarting that job. Never
point it at a source directory or an old experiment.

This packaging work checked the input and saved-output path; it did not perform
model retraining. Recorded dependency versions are not a full transitive lockfile.
The source/model acquisition and raw telemetry regeneration scope is recorded
once in the repository's `ARTIFACT_SCOPE.md`.
