| Model | seeds | error | setpoint | largest output | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Recurrent policy | 2 | 0.3984 ± 0.0010 | 0.2251 ± 0.0008 | 0.5707 ± 0.0013 | -24.18 ± 0.04 | 60.0 ± 0.0 | — | — | 0.00 ± 0.00 | 12.0 ± 2.5 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Predictions only | 3 | 0.3304 ± 0.0263 | 0.1808 ± 0.0120 | 0.4792 ± 0.0406 | -20.10 ± 1.58 | 35.3 ± 11.1 | 0.2139 ± 0.0004 | 0.2139 ± 0.0004 | 0.00 ± 0.00 | 13.9 ± 1.9 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Trained to predict, reading no prediction | 2 | 0.3215 ± 0.0277 | 0.1751 ± 0.0107 | 0.4670 ± 0.0446 | -19.49 ± 1.61 | 24.0 ± 7.0 | 0.2152 ± 0.0014 | 0.2152 ± 0.0014 | 0.00 ± 0.00 | 7.2 ± 6.8 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Trained through deployment | 3 | 0.2472 ± 0.0040 | 0.1629 ± 0.0009 | 0.3310 ± 0.0083 | -15.39 ± 0.25 | 9.3 ± 0.3 | 0.2185 ± 0.0005 | 0.2249 ± 0.0011 | 0.76 ± 0.03 | 20.3 ± 0.8 | 27.5 ± 1.0 | 9.23 ± 0.02 |
| Trained through deployment, learning steps kept, updates discarded | 3 | 0.3101 ± 0.0082 | 0.1810 ± 0.0029 | 0.4386 ± 0.0138 | -19.18 ± 0.50 | 15.7 ± 1.3 | 0.2244 ± 0.0005 | 0.2244 ± 0.0005 | 0.75 ± 0.03 | 20.3 ± 0.7 | 27.9 ± 1.1 | 9.22 ± 0.05 |
| Same, one-tick workspace gradients | 3 | 0.2295 ± 0.0052 | 0.1492 ± 0.0014 | 0.3094 ± 0.0095 | -14.28 ± 0.30 | 7.0 ± 0.0 | 0.2167 ± 0.0005 | 0.2281 ± 0.0007 | 0.61 ± 0.08 | 18.9 ± 1.0 | 29.3 ± 2.6 | 8.80 ± 0.13 |
| Same, one-tick workspace gradients, learning steps kept, updates discarded | 3 | 0.3170 ± 0.0106 | 0.1792 ± 0.0037 | 0.4539 ± 0.0182 | -19.56 ± 0.63 | 18.0 ± 2.0 | 0.2239 ± 0.0007 | 0.2239 ± 0.0007 | 0.61 ± 0.07 | 18.9 ± 1.3 | 29.7 ± 2.1 | 8.83 ± 0.15 |
| Same, no learning | 3 | 0.3470 ± 0.0062 | 0.1955 ± 0.0061 | 0.4976 ± 0.0095 | -21.45 ± 0.40 | 30.0 ± 1.5 | 0.2163 ± 0.0004 | 0.2163 ± 0.0004 | 1.58 ± 0.13 | 18.0 ± 1.1 | 0.0 ± 0.0 | 10.27 ± 0.48 |
| Same, one meta step per window | 3 | 0.2521 ± 0.0064 | 0.1606 ± 0.0028 | 0.3430 ± 0.0100 | -15.66 ± 0.40 | 9.0 ± 0.6 | 0.2186 ± 0.0003 | 0.2175 ± 0.0002 | 0.76 ± 0.06 | 18.6 ± 1.6 | 25.8 ± 1.4 | 9.11 ± 0.13 |
| Same, one meta step per window, learning steps kept, updates discarded | 3 | 0.2794 ± 0.0120 | 0.1679 ± 0.0046 | 0.3902 ± 0.0195 | -17.31 ± 0.74 | 12.0 ± 1.5 | 0.2190 ± 0.0002 | 0.2190 ± 0.0002 | 0.75 ± 0.06 | 18.8 ± 1.6 | 26.2 ± 1.5 | 9.10 ± 0.13 |
| Same, cheap targets only | 2 | 0.2955 ± 0.0249 | 0.1965 ± 0.0113 | 0.3940 ± 0.0386 | -18.26 ± 1.51 | 13.5 ± 2.5 | 0.2214 ± 0.0008 | 0.2411 ± 0.0020 | 0.75 ± 0.03 | 17.2 ± 0.3 | 24.9 ± 0.4 | 8.99 ± 0.09 |
| Same, cheap targets only, learning steps kept, updates discarded | 2 | 0.3323 ± 0.0282 | 0.1994 ± 0.0134 | 0.4643 ± 0.0430 | -20.48 ± 1.70 | 20.0 ± 6.0 | 0.2285 ± 0.0007 | 0.2285 ± 0.0007 | 0.73 ± 0.02 | 17.0 ± 0.7 | 25.3 ± 0.1 | 8.97 ± 0.06 |
| Same, four adapter blocks | 2 | 0.2490 ± 0.0071 | 0.1740 ± 0.0047 | 0.3235 ± 0.0096 | -15.47 ± 0.46 | 9.5 ± 0.5 | 0.2181 ± 0.0004 | 0.2254 ± 0.0002 | 0.71 ± 0.04 | 19.3 ± 2.9 | 27.7 ± 2.4 | 9.05 ± 0.04 |
| Same, four adapter blocks, learning steps kept, updates discarded | 2 | 0.3007 ± 0.0166 | 0.1842 ± 0.0076 | 0.4165 ± 0.0257 | -18.59 ± 1.03 | 14.8 ± 3.2 | 0.2226 ± 0.0014 | 0.2226 ± 0.0014 | 0.67 ± 0.04 | 19.7 ± 2.6 | 28.1 ± 2.6 | 8.95 ± 0.04 |
| Same, following values from copies | 1 | 0.3157 ± 0.0000 | 0.2111 ± 0.0000 | 0.4197 ± 0.0000 | -19.54 ± 0.00 | 12.0 ± 0.0 | 0.2205 ± 0.0000 | 0.2267 ± 0.0000 | 0.63 ± 0.00 | 22.8 ± 0.0 | 32.2 ± 0.0 | 9.08 ± 0.00 |
| Same, following values from copies, learning steps kept, updates discarded | 1 | 0.4304 ± 0.0000 | 0.2592 ± 0.0000 | 0.6007 ± 0.0000 | -26.47 ± 0.00 | 38.0 ± 0.0 | 0.2286 ± 0.0000 | 0.2286 ± 0.0000 | 0.65 ± 0.00 | 22.6 ± 0.0 | 31.9 ± 0.0 | 9.17 ± 0.00 |
| Same, copies kept, values not followed | 1 | 0.2693 ± 0.0000 | 0.1786 ± 0.0000 | 0.3596 ± 0.0000 | -16.71 ± 0.00 | 9.0 ± 0.0 | 0.2218 ± 0.0000 | 0.2266 ± 0.0000 | 0.77 ± 0.00 | 18.7 ± 0.0 | 28.3 ± 0.0 | 9.33 ± 0.00 |
| Same, copies kept, values not followed, learning steps kept, updates discarded | 1 | 0.3682 ± 0.0000 | 0.2177 ± 0.0000 | 0.5178 ± 0.0000 | -22.69 ± 0.00 | 24.5 ± 0.0 | 0.2304 ± 0.0000 | 0.2304 ± 0.0000 | 0.80 ± 0.00 | 18.8 ± 0.0 | 28.0 ± 0.0 | 9.41 ± 0.00 |
| New activity head over the fixed model | 2 | 0.2172 ± 0.0033 | 0.1451 ± 0.0017 | 0.2890 ± 0.0049 | -13.55 ± 0.20 | 7.0 ± 0.0 | 0.2163 ± 0.0003 | 0.2277 ± 0.0002 | 0.53 ± 0.10 | 20.8 ± 1.4 | 30.8 ± 1.9 | 8.62 ± 0.24 |
| New activity head over the fixed model, learning steps kept, updates discarded | 2 | 0.3318 ± 0.0017 | 0.1903 ± 0.0002 | 0.4725 ± 0.0032 | -20.48 ± 0.09 | 22.0 ± 0.0 | 0.2227 ± 0.0001 | 0.2226 ± 0.0001 | 0.51 ± 0.14 | 21.8 ± 1.2 | 31.8 ± 1.6 | 8.63 ± 0.40 |

Paired differences over seeds, own setting minus the other (negative: lower in the own setting):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| Trained through deployment | learning steps kept, updates discarded | -0.0629 ± 0.0058 (n=3) | +3.80 ± 0.35 |
| Same, one-tick workspace gradients | learning steps kept, updates discarded | -0.0874 ± 0.0128 (n=3) | +5.28 ± 0.78 |
| Same, one meta step per window | learning steps kept, updates discarded | -0.0273 ± 0.0068 (n=3) | +1.65 ± 0.41 |
| Same, cheap targets only | learning steps kept, updates discarded | -0.0368 ± 0.0033 (n=2) | +2.22 ± 0.19 |
| Same, four adapter blocks | learning steps kept, updates discarded | -0.0517 ± 0.0095 (n=2) | +3.12 ± 0.57 |
| Same, following values from copies | learning steps kept, updates discarded | -0.1147 ± 0.0000 (n=1) | +6.93 ± 0.00 |
| Same, copies kept, values not followed | learning steps kept, updates discarded | -0.0988 ± 0.0000 (n=1) | +5.98 ± 0.00 |
| New activity head over the fixed model | learning steps kept, updates discarded | -0.1146 ± 0.0050 (n=2) | +6.93 ± 0.29 |
