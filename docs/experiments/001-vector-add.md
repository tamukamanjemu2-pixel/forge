# Experiment 001 — CUDA Vector Add Baseline

## Hardware

- GPU: NVIDIA Tesla T4
- Compute Capability: 7.5
- Iterations: 100

## Results

| N | Threads | Kernel Time (ms) | Bandwidth (GB/s) |
|---:|---:|---:|---:|
| 1,048,576 | 64 | 0.0510 | 246.85 |
| 1,048,576 | 128 | 0.0507 | 248.02 |
| 1,048,576 | 256 | 0.0506 | 248.43 |
| 1,048,576 | 512 | 0.0508 | 247.65 |
| 1,048,576 | 1024 | 0.0514 | 244.97 |
| 4,194,304 | 64 | 0.1941 | 259.30 |
| 4,194,304 | 128 | 0.1935 | 260.12 |
| 4,194,304 | 256 | 0.1932 | 260.55 |
| 4,194,304 | 512 | 0.1937 | 259.87 |
| 4,194,304 | 1024 | 0.1949 | 258.29 |
| 16,777,216 | 64 | 0.7687 | 261.90 |
| 16,777,216 | 128 | 0.7679 | 262.19 |
| 16,777,216 | 256 | 0.7748 | 259.83 |
| 16,777,216 | 512 | 0.7811 | 257.75 |
| 16,777,216 | 1024 | 0.7859 | 256.16 |
| 67,108,864 | 64 | 3.1025 | 259.56 |
| 67,108,864 | 128 | 3.0530 | 263.77 |
| 67,108,864 | 256 | 3.0625 | 262.95 |
| 67,108,864 | 512 | 3.0865 | 260.91 |
| 67,108,864 | 1024 | 3.1109 | 258.87 |

All configurations produced a maximum numerical error of zero.

## Conclusion

Vector addition reaches a stable bandwidth region of approximately 260 GB/s for sufficiently large inputs. Launch configurations between 64 and 256 threads per block perform similarly, while larger blocks provide no measurable advantage.

The CUDA backend and benchmark infrastructure are therefore sufficient to proceed to compute-intensive operators and runtime development.