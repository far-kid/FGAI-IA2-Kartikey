## Results (test split)

| Model | Parameters | Epochs run | Train time (s) | Sec/epoch | Final train loss | Best val loss | Test loss | Test perplexity | Top-1 acc | Top-5 acc | Val-train gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SimpleRNN | 796832 | 9 | 17.2 | 1.9 | 3.689 | 4.544 | 4.59 | 98.5 | 0.2099 | 0.4345 | 0.514 |
| BiRNN | 1333664 | 6 | 12.3 | 2.1 | 3.42 | 4.676 | 4.697 | 109.6 | 0.1927 | 0.4155 | 0.422 |
| LSTM | 871328 | 10 | 22.6 | 2.3 | 3.637 | 4.447 | 4.476 | 87.9 | 0.2252 | 0.4544 | 0.628 |
| GRU | 846496 | 8 | 15.4 | 1.9 | 3.549 | 4.46 | 4.482 | 88.4 | 0.2214 | 0.4434 | 0.447 |


## Generation statistics

| temperature | model | distinct1 | distinct2 | rep_trigram | imm_repeat | bigram_valid | trigram_copy |
|---|---|---|---|---|---|---|---|
| 0.5 | SimpleRNN | 0.669 | 0.903 | 0.034 | 0.042 | 0.966 | 0.759 |
| 0.5 | BiRNN | 0.603 | 0.821 | 0.083 | 0.058 | 0.962 | 0.766 |
| 0.5 | LSTM | 0.655 | 0.894 | 0.028 | 0.034 | 0.971 | 0.769 |
| 0.5 | GRU | 0.664 | 0.898 | 0.028 | 0.042 | 0.949 | 0.733 |
| 1.0 | SimpleRNN | 0.804 | 0.979 | 0.004 | 0.021 | 0.754 | 0.336 |
| 1.0 | BiRNN | 0.796 | 0.976 | 0.005 | 0.02 | 0.709 | 0.298 |
| 1.0 | LSTM | 0.815 | 0.977 | 0.005 | 0.016 | 0.724 | 0.296 |
| 1.0 | GRU | 0.794 | 0.972 | 0.005 | 0.02 | 0.694 | 0.309 |
| 1.5 | SimpleRNN | 0.917 | 0.998 | 0.0 | 0.002 | 0.392 | 0.063 |
| 1.5 | BiRNN | 0.917 | 0.999 | 0.0 | 0.003 | 0.342 | 0.041 |
| 1.5 | LSTM | 0.907 | 0.998 | 0.0 | 0.002 | 0.411 | 0.067 |
| 1.5 | GRU | 0.92 | 0.997 | 0.0 | 0.002 | 0.344 | 0.054 |


## Generation score and best temperature per model

| model | 0.5 | 1.0 | 1.5 | best_T | best_score |
|---|---|---|---|---|---|
| SimpleRNN | 0.902 | 0.848 | 0.563 | 0.5 | 0.902 |
| BiRNN | 0.812 | 0.817 | 0.51 | 1.0 | 0.817 |
| LSTM | 0.905 | 0.828 | 0.582 | 0.5 | 0.905 |
| GRU | 0.897 | 0.806 | 0.512 | 0.5 | 0.897 |


## Ranks

| model | perplexity_rank | gen_score_rank | time_rank | combined(ppl+gen) |
|---|---|---|---|---|
| SimpleRNN | 3.0 | 2.0 | 3.0 | 5.0 |
| BiRNN | 4.0 | 4.0 | 1.0 | 8.0 |
| LSTM | 1.0 | 1.0 | 4.0 | 2.0 |
| GRU | 2.0 | 3.0 | 2.0 | 5.0 |


Best overall model (perplexity + generation score): **LSTM**
