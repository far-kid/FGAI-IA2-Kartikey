device: cuda
Downloaded from https://www.gutenberg.org/cache/epub/1661/pg1661.txt

--- Dataset characteristics ---
Characters : 562,202
Lines      : 11,922
Paragraphs : 2,546
Words (whitespace split): 104,506
Sample:
one of his own high-power lenses, would not
be more disturbing than a strong emotion in a nature such as his. And
yet there was but one woman to him, and that woman was the late Irene
Adler, of dubious and questionable memory.

I had seen little of Holmes lately. My marriage had drifted us away
from each other. My own complete happiness, and the home-centred
interests which rise up around the man

--- After preprocessing ---
{'characters': 562202, 'tokens': 129084, 'unique_tokens': 7955, 'vocab_size': 4000, 'unk_percent': 3.4}
Top-10 tokens: [(',', 7659), ('.', 6208), ('the', 5630), ('"', 5089), ('and', 3018), ('i', 3003), ('to', 2744), ('of', 2655), ('a', 2643), ('<p>', 2545)]
Tokens seen only once: 3521

--- Input-target sequences ---
train tokens= 116,175  X(116155, 20)  y(116155,)
val   tokens=   6,454  X(6434, 20)  y(6434,)
test  tokens=   6,455  X(6435, 20)  y(6435,)

Example input  X: the adventures of sherlock holmes <p> by arthur <unk> <unk> <p> contents <p> i . a scandal in bohemia ii
Example target y: .

=== SimpleRNN | parameters: 796,832 | device: cuda ===
[SimpleRNN] epoch 01 | train loss 5.329 | val loss 4.874 | val ppl 130.9 | val acc 0.173 | 2.7s
[SimpleRNN] epoch 02 | train loss 4.685 | val loss 4.684 | val ppl 108.2 | val acc 0.192 | 1.8s
[SimpleRNN] epoch 03 | train loss 4.441 | val loss 4.602 | val ppl 99.7 | val acc 0.199 | 1.9s
[SimpleRNN] epoch 04 | train loss 4.272 | val loss 4.557 | val ppl 95.3 | val acc 0.205 | 1.9s
[SimpleRNN] epoch 05 | train loss 4.141 | val loss 4.551 | val ppl 94.7 | val acc 0.208 | 1.9s
[SimpleRNN] epoch 06 | train loss 4.029 | val loss 4.544 | val ppl 94.0 | val acc 0.215 | 1.9s
[SimpleRNN] epoch 07 | train loss 3.933 | val loss 4.563 | val ppl 95.9 | val acc 0.213 | 1.9s
[SimpleRNN] epoch 08 | train loss 3.847 | val loss 4.569 | val ppl 96.5 | val acc 0.213 | 1.9s
[SimpleRNN] epoch 09 | train loss 3.689 | val loss 4.551 | val ppl 94.7 | val acc 0.216 | 1.9s
[SimpleRNN] early stopping at epoch 9
[SimpleRNN] TEST | loss 4.590 | perplexity 98.5 | top-1 0.210 | top-5 0.434 | train time 17.8s

=== BiRNN | parameters: 1,333,664 | device: cuda ===
[BiRNN] epoch 01 | train loss 5.253 | val loss 4.860 | val ppl 129.0 | val acc 0.176 | 2.1s
[BiRNN] epoch 02 | train loss 4.591 | val loss 4.700 | val ppl 110.0 | val acc 0.188 | 2.1s
[BiRNN] epoch 03 | train loss 4.253 | val loss 4.676 | val ppl 107.3 | val acc 0.197 | 2.1s
[BiRNN] epoch 04 | train loss 3.979 | val loss 4.686 | val ppl 108.4 | val acc 0.198 | 2.1s
[BiRNN] epoch 05 | train loss 3.745 | val loss 4.727 | val ppl 113.0 | val acc 0.199 | 2.0s
[BiRNN] epoch 06 | train loss 3.420 | val loss 4.723 | val ppl 112.6 | val acc 0.204 | 2.1s
[BiRNN] early stopping at epoch 6
[BiRNN] TEST | loss 4.697 | perplexity 109.6 | top-1 0.193 | top-5 0.416 | train time 12.5s

=== LSTM | parameters: 871,328 | device: cuda ===
[LSTM] epoch 01 | train loss 5.406 | val loss 4.978 | val ppl 145.2 | val acc 0.163 | 2.2s
[LSTM] epoch 02 | train loss 4.763 | val loss 4.725 | val ppl 112.8 | val acc 0.188 | 2.2s
[LSTM] epoch 03 | train loss 4.509 | val loss 4.604 | val ppl 99.9 | val acc 0.203 | 2.2s
[LSTM] epoch 04 | train loss 4.323 | val loss 4.528 | val ppl 92.6 | val acc 0.211 | 2.2s
[LSTM] epoch 05 | train loss 4.173 | val loss 4.488 | val ppl 88.9 | val acc 0.216 | 2.2s
[LSTM] epoch 06 | train loss 4.044 | val loss 4.458 | val ppl 86.3 | val acc 0.216 | 2.2s
[LSTM] epoch 07 | train loss 3.928 | val loss 4.457 | val ppl 86.2 | val acc 0.227 | 2.2s
[LSTM] epoch 08 | train loss 3.820 | val loss 4.447 | val ppl 85.4 | val acc 0.224 | 2.2s
[LSTM] epoch 09 | train loss 3.722 | val loss 4.464 | val ppl 86.8 | val acc 0.226 | 2.3s
[LSTM] epoch 10 | train loss 3.637 | val loss 4.475 | val ppl 87.8 | val acc 0.224 | 2.3s
[LSTM] TEST | loss 4.476 | perplexity 87.9 | top-1 0.225 | top-5 0.454 | train time 22.4s

=== GRU | parameters: 846,496 | device: cuda ===
[GRU] epoch 01 | train loss 5.320 | val loss 4.853 | val ppl 128.1 | val acc 0.179 | 1.9s
[GRU] epoch 02 | train loss 4.638 | val loss 4.625 | val ppl 102.0 | val acc 0.196 | 1.9s
[GRU] epoch 03 | train loss 4.371 | val loss 4.523 | val ppl 92.1 | val acc 0.209 | 1.9s
[GRU] epoch 04 | train loss 4.176 | val loss 4.470 | val ppl 87.4 | val acc 0.214 | 1.9s
[GRU] epoch 05 | train loss 4.013 | val loss 4.460 | val ppl 86.5 | val acc 0.218 | 1.9s
[GRU] epoch 06 | train loss 3.873 | val loss 4.465 | val ppl 86.9 | val acc 0.219 | 1.9s
[GRU] epoch 07 | train loss 3.748 | val loss 4.484 | val ppl 88.6 | val acc 0.223 | 2.0s
[GRU] epoch 08 | train loss 3.549 | val loss 4.478 | val ppl 88.0 | val acc 0.226 | 1.9s
[GRU] early stopping at epoch 8
[GRU] TEST | loss 4.482 | perplexity 88.4 | top-1 0.221 | top-5 0.443 | train time 15.4s

 === Summary of training ===
SimpleRNN  params=  796,832  epochs= 9  time=   17.8s  test_loss=4.590  ppl=   98.5  top1=0.210  top5=0.434
BiRNN      params=1,333,664  epochs= 6  time=   12.5s  test_loss=4.697  ppl=  109.6  top1=0.193  top5=0.416
LSTM       params=  871,328  epochs=10  time=   22.4s  test_loss=4.476  ppl=   87.9  top1=0.225  top5=0.454
GRU        params=  846,496  epochs= 8  time=   15.4s  test_loss=4.482  ppl=   88.4  top1=0.221  top5=0.443

=== Quantitative results (test split) ===
    Model  Parameters  Epochs run  Train time (s)  Sec/epoch  Final train loss  Best val loss  Test loss  Test perplexity  Top-1 acc  Top-5 acc  Val-train gap
SimpleRNN      796832           9            17.8        2.0             3.689          4.544      4.590             98.5     0.2099     0.4345          0.514
    BiRNN     1333664           6            12.5        2.1             3.420          4.676      4.697            109.6     0.1927     0.4155          0.422
     LSTM      871328          10            22.4        2.2             3.637          4.447      4.476             87.9     0.2252     0.4544          0.628
      GRU      846496           8            15.4        1.9             3.549          4.460      4.482             88.4     0.2214     0.4434          0.447

=== Ranking (from measured results) ===
Lowest test perplexity : LSTM < GRU < SimpleRNN < BiRNN
Highest top-1 accuracy : LSTM > GRU > SimpleRNN > BiRNN
Fastest total training : BiRNN < GRU < SimpleRNN < LSTM
Smallest val-train gap : BiRNN < GRU < SimpleRNN < LSTM

Best perplexity: LSTM (87.9); random-guess perplexity for this vocabulary would be 4000.

Saved generated samples to outputs/generations.md

=== Generation statistics (mean over seeds and samples) ===
 temperature     model  distinct1  distinct2  rep_trigram  imm_repeat  bigram_valid  trigram_copy
         0.5 SimpleRNN      0.669      0.903        0.034       0.042         0.966         0.759
         0.5     BiRNN      0.603      0.821        0.083       0.058         0.962         0.766
         0.5      LSTM      0.655      0.894        0.028       0.034         0.971         0.769
         0.5       GRU      0.664      0.898        0.028       0.042         0.949         0.733
         1.0 SimpleRNN      0.804      0.979        0.004       0.021         0.754         0.336
         1.0     BiRNN      0.796      0.976        0.005       0.020         0.709         0.298
         1.0      LSTM      0.815      0.977        0.005       0.016         0.724         0.296
         1.0       GRU      0.794      0.972        0.005       0.020         0.694         0.309
         1.5 SimpleRNN      0.917      0.998        0.000       0.002         0.392         0.063
         1.5     BiRNN      0.917      0.999        0.000       0.003         0.342         0.041
         1.5      LSTM      0.907      0.998        0.000       0.002         0.411         0.067
         1.5       GRU      0.920      0.997        0.000       0.002         0.344         0.054

=== Overall effect of temperature (mean over the four models) ===
             distinct1  distinct2  rep_trigram  imm_repeat  bigram_valid  trigram_copy
temperature
0.5              0.648      0.879        0.043       0.044         0.962         0.757
1.0              0.802      0.976        0.005       0.019         0.720         0.310
1.5              0.915      0.998        0.000       0.002         0.372         0.056

=== Generation score (higher is better) ===
temperature    0.5    1.0    1.5  best_T  best_score
model
SimpleRNN    0.902  0.848  0.563     0.5       0.902
BiRNN        0.812  0.817  0.510     1.0       0.817
LSTM         0.905  0.828  0.582     0.5       0.905
GRU          0.897  0.806  0.512     0.5       0.897

=== Ranks (1 = best) ===
           perplexity_rank  gen_score_rank  time_rank  combined(ppl+gen)
SimpleRNN              3.0             2.0        3.0                5.0
BiRNN                  4.0             4.0        1.0                8.0
LSTM                   1.0             1.0        4.0                2.0
GRU                    2.0             3.0        2.0                5.0

Best overall model by perplexity + generation score: LSTM
Saved outputs/summary_for_readme.md  (tables to paste into README.md)
