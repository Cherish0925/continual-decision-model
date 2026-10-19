| Model | instrument | thermal | navigation |
|---|---|---|---|
| Recurrent policy | 0.3949 ± 0.0057 (2) | — | — |
| Predictions only | 0.3068 ± 0.0363 (3) | — | — |
| Trained to predict, reading no prediction | 0.2955 ± 0.0372 (2) | — | — |
| Trained through deployment | 0.1837 ± 0.0027 (3) | — | — |
| Same, one-tick workspace gradients | 0.1607 ± 0.0037 (3) | — | — |
| Same, no learning | 0.3252 ± 0.0057 (3) | — | — |
| Same, one meta step per window | 0.1899 ± 0.0062 (3) | — | — |
| Same, cheap targets only | 0.2292 ± 0.0198 (2) | — | — |
| Same, four adapter blocks | 0.1786 ± 0.0058 (2) | — | — |
| Same, following values from copies | 0.2407 ± 0.0000 (1) | — | — |
| Same, copies kept, values not followed | 0.1970 ± 0.0000 (1) | — | — |
| New activity head over the fixed model | 0.1515 ± 0.0018 (2) | — | — |
| Trained through deployment, three families | 0.2562 ± 0.0081 (2) | 0.1344 ± 0.0273 (2) | 0.3089 ± 0.0015 (2) |
| Predictions only, three families | 0.3417 ± 0.0028 (2) | 0.1275 ± 0.0266 (2) | 0.3818 ± 0.0021 (2) |
| Trained through deployment, vehicles only | 0.3556 ± 0.0235 (2) | — | 0.2580 ± 0.0059 (2) |
| Trained through deployment, thermal only | 0.6630 ± 0.0000 (1) | 0.0745 ± 0.0000 (1) | — |
