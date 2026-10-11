# Result Validation

## Scope and provenance

The experiment uses CUDA for training. The checks below use CPU evaluation to verify the saved checkpoints and results in `outputs/`. They do not retrain models, regenerate samples, or replace the reported results.

The original GPU model and training-environment package versions are not recorded. The verification environment used PyTorch `2.14.1+cpu`, NumPy `2.5.4`, and pandas `3.0.6`; these are not the recorded training-environment versions.

To avoid the top-level training/data-export side effects in `main.py`, the verification command extracted only the relevant definitions (`tokenize`, `RecurrentLM`, `evaluate`, and `count_params`) using Python's AST. Checkpoints were loaded with `map_location="cpu"` and `weights_only=True`. No import or full execution of `main.py` was used for checkpoint verification.

## Dataset and vocabulary checks

- Parsed `main.py` successfully as Python source.
- Reprocessed the supplied `sherlock_holmes.txt` using the implemented Gutenberg-marker extraction and tokenizer.
- Confirmed the 562,202-character body and 129,084 model tokens match `outputs/dataset_stats.json`.
- Confirmed `outputs/vocab.json` equals `<unk>` followed by the 3,999 most frequent tokens of the **full corpus**. This independently confirms the vocabulary-selection leakage noted in the README; vocabulary was not fitted solely to training tokens.
- Reconstructed the chronological test partition and its **6,435** context–target pairs, using 20-token windows.

## Saved-checkpoint evaluation

All four checkpoint files load into their corresponding model architectures. The table compares the original saved GPU metrics against a fresh CPU evaluation on the reconstructed test windows. Values below retain more precision than the report tables to make differences visible.

| Model | Saved loss | CPU loss | Saved perplexity | CPU perplexity | Saved top-1 | CPU top-1 | Top-5 agreement |
|---|---:|---:|---:|---:|---:|---:|---|
| SimpleRNN | 4.590062106 | 4.590062089 | 98.50054745 | 98.50054581 | 0.20994561 | 0.20994561 | Exact |
| BiRNN | 4.696706226 | 4.696706193 | 109.58562708 | 109.58562345 | 0.19269619 | 0.19269619 | Exact |
| LSTM | 4.476481687 | 4.476481343 | 87.92478090 | 87.92475067 | 0.22517483 | 0.22501943 | Exact |
| GRU | 4.481540835 | 4.481540733 | 88.37073251 | 88.37072350 | 0.22144522 | 0.22144522 | Exact |

**Important discrepancy:** an initial strict comparison using `rel_tol=1e-5` and `abs_tol=1e-5` failed for LSTM top-1 accuracy. CPU evaluation records **one fewer correct prediction out of 6,435**, a difference of approximately **0.01554 percentage points**. A second evaluation reproduced that difference. CPU/GPU numerical behavior is a plausible explanation, but the original GPU runtime was not reproduced, so the cause is not established.

All losses and perplexities agree closely enough to preserve their reported rounded values. Top-1 results for SimpleRNN, BiRNN, and GRU and top-5 results for all four models agree with the saved values. The README intentionally reports the **original training-run measurements**, not a mixture of original and local measurements. Exact cross-device reproduction of every prediction is not claimed.

## Generation-artifact checks

The saved per-sample CSV and summary were checked without generating new text:

- Exactly **720** metric records are present.
- Each model–temperature–prompt combination contains **20** records.
- Recomputing the means of all six generation metrics by model and temperature, then rounding to three decimals, reproduces all **12** rows in `outputs/generation_summary.csv`.
- Qualitative observations in the README cite the displayed text from `outputs/generations.md`; they do not assert that all 720 complete continuations were reviewed. Only 72 continuations are displayed in that file.

## Timing consistency

The saved JSON and CSV use total training times of **17.2, 12.3, 22.6, and 15.4 seconds**, respectively, for SimpleRNN, BiRNN, LSTM, and GRU. The supplied console transcript instead displays **17.8, 12.5, 22.4, and 15.4 seconds**.

The reason for the discrepancy is unknown. The report uses JSON/CSV timings consistently and leaves the original artifacts and console transcript unchanged. These timings belong to the GPU training run; CPU checkpoint evaluation does not validate or recreate those training times.

## Report consistency checks

- Verified the README's test-result tables against `outputs/results.csv` and `outputs/training_results.json`.
- Checked best-checkpoint epochs against the validation-loss histories.
- Recomputed the three temperature-level averages and all 12 heuristic generation scores from `outputs/generation_summary.csv`.
- Verified each model's highest-scoring tested temperature against the computed scores.
- Checked that all relative README links resolve to existing files or directories.
- Editor diagnostics for `main.py` report no errors or warnings.

## Limits of validation

- No full retraining was performed during verification.
- The original training software/hardware environment was not recreated.
- The corpus-wide vocabulary limitation remains part of the reported experiment; correcting it requires retraining and new results.
- Reading selected generated continuations is a qualitative review, not a blinded human study or a semantic-quality benchmark.

