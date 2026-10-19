| Model | seeds | error | setpoint | largest output | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Recurrent policy | 2 | 0.4065 ± 0.0040 | 0.2248 ± 0.0033 | 0.5836 ± 0.0047 | -24.66 ± 0.21 | 60.0 ± 0.0 | — | — | 0.00 ± 0.00 | 12.0 ± 2.8 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Predictions only | 3 | 0.3254 ± 0.0318 | 0.1730 ± 0.0134 | 0.4739 ± 0.0498 | -19.78 ± 1.91 | 42.3 ± 14.3 | 0.3311 ± 0.0007 | 0.3311 ± 0.0007 | 0.00 ± 0.00 | 14.2 ± 1.9 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Trained to predict, reading no prediction | 2 | 0.3184 ± 0.0274 | 0.1686 ± 0.0099 | 0.4644 ± 0.0444 | -19.29 ± 1.59 | 27.5 ± 8.5 | 0.3327 ± 0.0025 | 0.3327 ± 0.0025 | 0.00 ± 0.00 | 7.0 ± 6.5 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Trained through deployment | 3 | 0.2110 ± 0.0037 | 0.1447 ± 0.0014 | 0.2757 ± 0.0084 | -13.19 ± 0.23 | 8.0 ± 0.0 | 0.3335 ± 0.0005 | 0.3473 ± 0.0049 | 0.78 ± 0.02 | 20.4 ± 0.9 | 27.3 ± 1.3 | 9.28 ± 0.02 |
| Trained through deployment, learning steps kept, updates discarded | 3 | 0.3187 ± 0.0135 | 0.1851 ± 0.0038 | 0.4489 ± 0.0235 | -19.70 ± 0.82 | 18.3 ± 2.2 | 0.3422 ± 0.0018 | 0.3422 ± 0.0018 | 0.77 ± 0.02 | 20.5 ± 0.7 | 27.4 ± 1.3 | 9.27 ± 0.02 |
| Same, one-tick workspace gradients | 3 | 0.2117 ± 0.0050 | 0.1416 ± 0.0016 | 0.2799 ± 0.0085 | -13.21 ± 0.29 | 7.0 ± 0.0 | 0.3324 ± 0.0005 | 0.3517 ± 0.0037 | 0.67 ± 0.08 | 19.3 ± 0.9 | 29.0 ± 2.3 | 8.99 ± 0.14 |
| Same, one-tick workspace gradients, learning steps kept, updates discarded | 3 | 0.3505 ± 0.0085 | 0.1926 ± 0.0044 | 0.5045 ± 0.0129 | -21.59 ± 0.51 | 24.7 ± 3.0 | 0.3408 ± 0.0005 | 0.3408 ± 0.0005 | 0.63 ± 0.07 | 19.8 ± 1.0 | 30.0 ± 1.8 | 8.92 ± 0.14 |
| Same, no learning | 3 | 0.3616 ± 0.0090 | 0.1966 ± 0.0098 | 0.5226 ± 0.0094 | -22.32 ± 0.55 | 41.7 ± 5.9 | 0.3327 ± 0.0006 | 0.3327 ± 0.0006 | 1.57 ± 0.12 | 18.1 ± 0.9 | 0.0 ± 0.0 | 10.23 ± 0.46 |
| Same, one meta step per window | 3 | 0.2283 ± 0.0045 | 0.1529 ± 0.0018 | 0.3017 ± 0.0098 | -14.21 ± 0.28 | 9.0 ± 0.0 | 0.3344 ± 0.0007 | 0.3344 ± 0.0002 | 0.76 ± 0.06 | 18.3 ± 1.7 | 26.1 ± 1.1 | 9.12 ± 0.16 |
| Same, one meta step per window, learning steps kept, updates discarded | 3 | 0.2565 ± 0.0205 | 0.1571 ± 0.0079 | 0.3534 ± 0.0329 | -15.92 ± 1.24 | 11.3 ± 1.9 | 0.3365 ± 0.0002 | 0.3365 ± 0.0002 | 0.77 ± 0.06 | 18.6 ± 1.6 | 25.8 ± 1.0 | 9.15 ± 0.15 |
| Same, cheap targets only | 2 | 0.2557 ± 0.0170 | 0.1757 ± 0.0089 | 0.3337 ± 0.0248 | -15.86 ± 1.03 | 11.0 ± 1.0 | 0.3355 ± 0.0009 | 0.3515 ± 0.0024 | 0.77 ± 0.05 | 17.7 ± 0.2 | 24.6 ± 1.8 | 9.06 ± 0.04 |
| Same, cheap targets only, learning steps kept, updates discarded | 2 | 0.3571 ± 0.0227 | 0.2072 ± 0.0105 | 0.5032 ± 0.0345 | -21.97 ± 1.36 | 27.0 ± 9.0 | 0.3457 ± 0.0030 | 0.3457 ± 0.0030 | 0.71 ± 0.02 | 17.9 ± 0.3 | 26.5 ± 0.1 | 8.98 ± 0.06 |
| Same, four adapter blocks | 2 | 0.2036 ± 0.0087 | 0.1528 ± 0.0009 | 0.2531 ± 0.0162 | -12.72 ± 0.55 | 8.5 ± 0.5 | 0.3335 ± 0.0010 | 0.3461 ± 0.0017 | 0.67 ± 0.00 | 19.7 ± 2.5 | 29.1 ± 1.3 | 9.00 ± 0.09 |
| Same, four adapter blocks, learning steps kept, updates discarded | 2 | 0.3124 ± 0.0047 | 0.1914 ± 0.0010 | 0.4303 ± 0.0083 | -19.30 ± 0.31 | 17.0 ± 2.0 | 0.3411 ± 0.0012 | 0.3411 ± 0.0012 | 0.63 ± 0.03 | 20.6 ± 2.0 | 29.6 ± 0.9 | 8.90 ± 0.17 |
| Same, following values from copies | 1 | 0.2811 ± 0.0000 | 0.1866 ± 0.0000 | 0.3732 ± 0.0000 | -17.44 ± 0.00 | 10.0 ± 0.0 | 0.3347 ± 0.0000 | 0.3439 ± 0.0000 | 0.68 ± 0.00 | 22.7 ± 0.0 | 31.6 ± 0.0 | 9.24 ± 0.00 |
| Same, following values from copies, learning steps kept, updates discarded | 1 | 0.4368 ± 0.0000 | 0.2582 ± 0.0000 | 0.6109 ± 0.0000 | -26.84 ± 0.00 | 46.0 ± 0.0 | 0.3438 ± 0.0000 | 0.3438 ± 0.0000 | 0.67 ± 0.00 | 22.5 ± 0.0 | 31.9 ± 0.0 | 9.21 ± 0.00 |
| Same, copies kept, values not followed | 1 | 0.2221 ± 0.0000 | 0.1535 ± 0.0000 | 0.2889 ± 0.0000 | -13.84 ± 0.00 | 8.0 ± 0.0 | 0.3363 ± 0.0000 | 0.3450 ± 0.0000 | 0.75 ± 0.00 | 18.2 ± 0.0 | 28.5 ± 0.0 | 9.27 ± 0.00 |
| Same, copies kept, values not followed, learning steps kept, updates discarded | 1 | 0.3658 ± 0.0000 | 0.2152 ± 0.0000 | 0.5125 ± 0.0000 | -22.53 ± 0.00 | 26.0 ± 0.0 | 0.3463 ± 0.0000 | 0.3463 ± 0.0000 | 0.79 ± 0.00 | 18.9 ± 0.0 | 28.1 ± 0.0 | 9.38 ± 0.00 |
| New activity head over the fixed model | 2 | 0.1991 ± 0.0071 | 0.1360 ± 0.0032 | 0.2606 ± 0.0109 | -12.44 ± 0.42 | 7.5 ± 0.5 | 0.3321 ± 0.0003 | 0.3496 ± 0.0006 | 0.53 ± 0.09 | 20.8 ± 1.6 | 31.4 ± 1.6 | 8.66 ± 0.22 |
| New activity head over the fixed model, learning steps kept, updates discarded | 2 | 0.3626 ± 0.0001 | 0.2013 ± 0.0006 | 0.5197 ± 0.0009 | -22.32 ± 0.02 | 29.5 ± 0.5 | 0.3398 ± 0.0004 | 0.3398 ± 0.0004 | 0.50 ± 0.17 | 21.7 ± 1.6 | 33.1 ± 1.8 | 8.66 ± 0.50 |

Paired differences over seeds, own setting minus the other (negative: lower in the own setting):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| Trained through deployment | learning steps kept, updates discarded | -0.1077 ± 0.0105 (n=3) | +6.50 ± 0.63 |
| Same, one-tick workspace gradients | learning steps kept, updates discarded | -0.1389 ± 0.0128 (n=3) | +8.38 ± 0.77 |
| Same, one meta step per window | learning steps kept, updates discarded | -0.0283 ± 0.0165 (n=3) | +1.71 ± 1.00 |
| Same, cheap targets only | learning steps kept, updates discarded | -0.1013 ± 0.0057 (n=2) | +6.11 ± 0.33 |
| Same, four adapter blocks | learning steps kept, updates discarded | -0.1088 ± 0.0039 (n=2) | +6.57 ± 0.24 |
| Same, following values from copies | learning steps kept, updates discarded | -0.1557 ± 0.0000 (n=1) | +9.40 ± 0.00 |
| Same, copies kept, values not followed | learning steps kept, updates discarded | -0.1437 ± 0.0000 (n=1) | +8.69 ± 0.00 |
| New activity head over the fixed model | learning steps kept, updates discarded | -0.1635 ± 0.0070 (n=2) | +9.88 ± 0.40 |
