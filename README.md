# OPT

| Wiki2 PPL | 125m | 350m | 1.3b | 6.7b | 13b | 30b |
|:---------:|:----:|:--------:|:---------:|:--------:|:---------:|:----------:|
| OPTQ-INT3 Row-wise  | 46.11 | 31.35 | 19.42 | 12.66 | 11.16  | 10.17 |
| LUT-GEMM-INT3 Row-wise | 69.81 | 43.41 | 68.56 | 17.45 | 12.5 | 139.9 |
| Ours-INT3 Column-wise | 37.22 | 27.72 | 17.79 | 11.81 | 10.69 | |
| Ours-INT3 Block-wise (8 column) | 69.03 | 33.01 | 21.97 | 12.69 | 12.44 | 10.13 |
| Ours-INT3 Block-wise (cust. groups) | 52.57 | 28.04 | 19.3 | 12.12 | 11.28 | |

| End-to-end latency (ms) | 125m | 350m | 1.3b | 6.7b | 13b | 30b |
|:---------:|:----:|:--------:|:---------:|:--------:|:---------:|:----------:|
| OPTQ-INT3 Row-wise  | 11.38 | 21.3 | 22.3 | 29.7 | 36.6 | 44.2 |
| LUT-GEMM-INT3 Row-wise | 8.1 | 15.6 | 15.9 | 21.3 | 27.2 | 32.9 |
| Ours-INT3 Column-wise | 8.1 | 15.8 | 15.9 | 21.6 | 26.4 | 44.1 |
| Ours-INT3 Block-wise (cust. groups) | 8.4 | 15.8 | 16 | 21.1 | 26.7 | 33.2 |

To evaluate Wiki2 PPL, run the 'eval_opt.sh' script under 'script/optq', 'script/lut', 'script/columnwise', 'script/blockwise' 
for 'OPTQ-INT3 Row-wise, LUT-GEMM-INT3 Row-wise, Ours-INT3 Column-wise, Ours-INT3 Block-wise (cust. groups)' respectively. To use the 'Ours-INT3 Block-wise (8 column)'
method, set the argument 'groupsize' to -1 in the script under the 'script/blockwise' folder.

To benchmark end-to-end latency, run the 'benchmark_opt.sh' script under 'script/optq', 'script/lut', 'script/columnwise', 'script/blockwise' 
for 'OPTQ-INT3 Row-wise, LUT-GEMM-INT3 Row-wise, Ours-INT3 Column-wise, Ours-INT3 Block-wise (cust. groups)' respectively.
