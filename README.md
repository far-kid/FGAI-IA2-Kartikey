# Sherlock Holmes Autoregressive Text Generation

Foundations of Generative AI (ICT4403), Assignment 2.

## Objective

Develop and compare four word-level autoregressive text-generation systems: Simple RNN, Bidirectional RNN, LSTM, and GRU. Each system predicts the next token from a fixed context and generates a continuation by repeatedly sampling and appending that prediction.

The assignment requires dataset characterization and preprocessing, comparable training conditions, quantitative evaluation, qualitative review, and generation from identical seeds at at least three temperatures. The final conclusion must justify the most effective model using measured performance and actual generated text, not architectural expectations alone. See `ia2.md` for the assignment requirements and `SUBMISSION.md` for the submission checklist.

## Implementation and results status

**Actual measured results are pending.** No training or evaluation was run to prepare this documentation. Dataset counts, parameter counts, losses, perplexities, accuracies, timings, generated examples, rankings, and a best-model conclusion must come from a completed run. No numerical results or winning model are invented here.

The documentation describes the intended completed workflow. Three changes are being implemented separately by the parent task:

- Fit the vocabulary on the training partition only. The inspected `main.py` currently counts tokens from the entire corpus before splitting; this must be corrected before reporting leakage-controlled results.
- Implement `python main.py --evaluate-only` to reuse saved artifacts without retraining. This option is not implemented in the inspected version.
- Generate `report.md` from `main.py`. The inspected version writes `outputs/summary_for_readme.md`, but does not yet generate `report.md`.

Verify these changes are complete before following the evaluation-only workflow or submitting results.

## Dataset

- **Title:** *The Adventures of Sherlock Holmes*
- **Author:** Arthur Conan Doyle
- **Source:** Project Gutenberg, ebook **1661**
- **Dataset page:** https://www.gutenberg.org/ebooks/1661
- **Primary plain-text URL:** https://www.gutenberg.org/cache/epub/1661/pg1661.txt
- **Fallback plain-text URL:** https://www.gutenberg.org/files/1661/1661-0.txt
- **Local file:** `sherlock_holmes.txt`

This English collection of detective stories contains narrative prose, dialogue, recurring characters, punctuation, and paragraph boundaries. It is suitable for studying next-token prediction and short narrative continuations, but is a single-author, single-book corpus rather than a representative sample of general English. Historical language and social biases in the text may be reflected in generated output.

The script downloads the text if the local file is absent, reads UTF-8 with optional byte-order mark, and removes the Gutenberg header and footer using the ebook boundary markers. Inspect the extracted body: marker removal does not necessarily remove the title, contents, or all front matter. If markers are missing, the current extraction retains the corresponding untrimmed portion.

Dataset characteristics are printed and saved by the program. Report the observed counts from the actual downloaded file; do not assume that different Gutenberg text revisions are identical. Preserve the source link and check Project Gutenberg's terms and applicable local copyright rules when redistributing the text.

## Preprocessing and sequence construction

The intended leakage-controlled workflow is:

1. Strip the Gutenberg wrapper and normalize line endings.
2. Normalize curly quotation marks and apostrophes; replace em dashes and double hyphens with a hyphen token.
3. Lowercase the text.
4. Convert blank-line paragraph breaks to `<p>` and ordinary hard-wrapped newlines to spaces.
5. Tokenize alphabetic words with internal apostrophes and the punctuation `. , ! ? ; : " - ( )`, preserving `<p>`. Digits and other unmatched characters are omitted by the tokenizer.
6. Split the token stream chronologically into **90% training, 5% validation, and 5% test**. Boundaries use integer truncation at 90% and 95% of the token count.
7. **Build the vocabulary using only training-token frequencies.** Reserve index 0 for `<unk>` and retain up to 3,999 frequent training tokens, for a maximum vocabulary size of 4,000. Use this same mapping for every partition and every model. Validation/test tokens absent from the training vocabulary map to `<unk>`; never fit or expand vocabulary using held-out text.
8. Construct windows independently within each partition, so no input–target pair crosses a split boundary.

For sequence length `L = 20`, each pair is:

```text
input  X[i] = tokens[i : i + 20]
target y[i] = tokens[i + 20]
```

The stride is 1 for training, validation, and test. A partition with `N` tokens therefore supplies `N - 20` examples when `N > 20`. Only training batches are shuffled. Adjacent windows overlap heavily within each split; this is intentional and should not be mistaken for independent examples.

Although the assignment calls this next-word prediction, punctuation and paragraph markers are also tokens. Lowercasing loses capitalization, and vocabulary truncation collapses rare words into `<unk>`. Report vocabulary size and unknown-token coverage, ideally separately for each partition. Unknown targets can make held-out metrics easier to satisfy without recovering the original rare word.

