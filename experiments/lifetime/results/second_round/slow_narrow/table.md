| Model | seeds | error | setpoint | largest output | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Recurrent policy | 2 | 0.5944 ± 0.0029 | 0.3506 ± 0.0007 | 0.8321 ± 0.0051 | -35.96 ± 0.16 | 60.0 ± 0.0 | — | — | 0.00 ± 0.00 | 7.5 ± 1.4 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Predictions only | 3 | 0.5366 ± 0.0159 | 0.3050 ± 0.0076 | 0.7624 ± 0.0241 | -32.49 ± 0.96 | 60.0 ± 0.0 | 0.2087 ± 0.0005 | 0.2087 ± 0.0005 | 0.00 ± 0.00 | 9.7 ± 1.5 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Trained to predict, reading no prediction | 2 | 0.5130 ± 0.0246 | 0.2921 ± 0.0131 | 0.7284 ± 0.0359 | -31.05 ± 1.41 | 60.0 ± 0.0 | 0.2098 ± 0.0015 | 0.2098 ± 0.0015 | 0.00 ± 0.00 | 8.2 ± 7.7 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Trained through deployment | 3 | 0.4090 ± 0.0063 | 0.2429 ± 0.0038 | 0.5709 ± 0.0097 | -25.14 ± 0.39 | 60.0 ± 0.0 | 0.2101 ± 0.0002 | 0.2272 ± 0.0011 | 0.74 ± 0.05 | 20.5 ± 0.6 | 27.9 ± 0.6 | 9.17 ± 0.15 |
| Trained through deployment, learning steps kept, updates discarded | 3 | 0.5392 ± 0.0036 | 0.3123 ± 0.0030 | 0.7604 ± 0.0042 | -33.00 ± 0.20 | 60.0 ± 0.0 | 0.2214 ± 0.0012 | 0.2214 ± 0.0012 | 0.76 ± 0.06 | 19.9 ± 1.0 | 27.5 ± 0.8 | 9.24 ± 0.18 |
| Same, one-tick workspace gradients | 3 | 0.4337 ± 0.0113 | 0.2675 ± 0.0046 | 0.5956 ± 0.0183 | -26.61 ± 0.66 | 60.0 ± 0.0 | 0.2083 ± 0.0003 | 0.2329 ± 0.0042 | 0.59 ± 0.07 | 19.8 ± 1.4 | 31.1 ± 1.4 | 8.85 ± 0.21 |
| Same, one-tick workspace gradients, learning steps kept, updates discarded | 3 | 0.5758 ± 0.0106 | 0.3424 ± 0.0073 | 0.8033 ± 0.0140 | -35.19 ± 0.63 | 60.0 ± 0.0 | 0.2209 ± 0.0021 | 0.2209 ± 0.0021 | 0.65 ± 0.11 | 19.2 ± 1.8 | 31.1 ± 1.4 | 9.08 ± 0.34 |
| Same, no learning | 3 | 0.5589 ± 0.0060 | 0.3252 ± 0.0072 | 0.7867 ± 0.0060 | -34.20 ± 0.35 | 60.0 ± 0.0 | 0.2098 ± 0.0004 | 0.2098 ± 0.0004 | 1.60 ± 0.11 | 14.1 ± 1.8 | 0.0 ± 0.0 | 10.32 ± 0.40 |
| Same, one meta step per window | 3 | 0.4453 ± 0.0137 | 0.2588 ± 0.0078 | 0.6271 ± 0.0195 | -27.31 ± 0.84 | 60.0 ± 0.0 | 0.2095 ± 0.0006 | 0.2097 ± 0.0004 | 0.72 ± 0.06 | 18.7 ± 1.4 | 27.3 ± 2.5 | 9.09 ± 0.07 |
| Same, one meta step per window, learning steps kept, updates discarded | 3 | 0.4781 ± 0.0169 | 0.2754 ± 0.0107 | 0.6756 ± 0.0233 | -29.29 ± 1.03 | 60.0 ± 0.0 | 0.2111 ± 0.0005 | 0.2111 ± 0.0005 | 0.71 ± 0.07 | 18.7 ± 1.5 | 27.9 ± 2.7 | 9.07 ± 0.06 |
| Same, cheap targets only | 2 | 0.4863 ± 0.0347 | 0.2862 ± 0.0294 | 0.6814 ± 0.0397 | -29.76 ± 2.08 | 60.0 ± 0.0 | 0.2125 ± 0.0005 | 0.2605 ± 0.0049 | 0.70 ± 0.10 | 17.0 ± 1.1 | 27.3 ± 4.6 | 8.99 ± 0.05 |
| Same, cheap targets only, learning steps kept, updates discarded | 2 | 0.5534 ± 0.0017 | 0.3221 ± 0.0036 | 0.7788 ± 0.0002 | -33.81 ± 0.07 | 60.0 ± 0.0 | 0.2283 ± 0.0016 | 0.2283 ± 0.0016 | 0.71 ± 0.13 | 16.6 ± 1.9 | 27.0 ± 5.0 | 8.99 ± 0.13 |
| Same, four adapter blocks | 2 | 0.3876 ± 0.0182 | 0.2348 ± 0.0108 | 0.5365 ± 0.0253 | -23.83 ± 1.13 | 53.5 ± 6.5 | 0.2102 ± 0.0014 | 0.2231 ± 0.0042 | 0.70 ± 0.00 | 19.6 ± 3.0 | 28.7 ± 1.2 | 9.08 ± 0.07 |
| Same, four adapter blocks, learning steps kept, updates discarded | 2 | 0.5039 ± 0.0095 | 0.2879 ± 0.0060 | 0.7145 ± 0.0129 | -30.86 ± 0.60 | 60.0 ± 0.0 | 0.2178 ± 0.0031 | 0.2178 ± 0.0031 | 0.67 ± 0.00 | 20.1 ± 2.6 | 29.3 ± 1.5 | 9.02 ± 0.10 |
| Same, following values from copies | 1 | 0.4799 ± 0.0000 | 0.2910 ± 0.0000 | 0.6640 ± 0.0000 | -29.44 ± 0.00 | 60.0 ± 0.0 | 0.2101 ± 0.0000 | 0.2238 ± 0.0000 | 0.60 ± 0.00 | 22.8 ± 0.0 | 32.8 ± 0.0 | 9.04 ± 0.00 |
| Same, following values from copies, learning steps kept, updates discarded | 1 | 0.6116 ± 0.0000 | 0.3717 ± 0.0000 | 0.8455 ± 0.0000 | -37.39 ± 0.00 | 60.0 ± 0.0 | 0.2219 ± 0.0000 | 0.2219 ± 0.0000 | 0.67 ± 0.00 | 22.2 ± 0.0 | 31.9 ± 0.0 | 9.24 ± 0.00 |
| Same, copies kept, values not followed | 1 | 0.4114 ± 0.0000 | 0.2478 ± 0.0000 | 0.5709 ± 0.0000 | -25.31 ± 0.00 | 60.0 ± 0.0 | 0.2116 ± 0.0000 | 0.2191 ± 0.0000 | 0.87 ± 0.00 | 19.8 ± 0.0 | 27.6 ± 0.0 | 9.67 ± 0.00 |
| Same, copies kept, values not followed, learning steps kept, updates discarded | 1 | 0.5683 ± 0.0000 | 0.3333 ± 0.0000 | 0.7974 ± 0.0000 | -34.79 ± 0.00 | 60.0 ± 0.0 | 0.2238 ± 0.0000 | 0.2238 ± 0.0000 | 0.98 ± 0.00 | 19.7 ± 0.0 | 25.9 ± 0.0 | 9.94 ± 0.00 |
| New activity head over the fixed model | 2 | 0.4222 ± 0.0056 | 0.2662 ± 0.0016 | 0.5743 ± 0.0095 | -25.92 ± 0.35 | 60.0 ± 0.0 | 0.2077 ± 0.0002 | 0.2268 ± 0.0007 | 0.50 ± 0.10 | 21.9 ± 1.2 | 31.5 ± 0.3 | 8.57 ± 0.38 |
| New activity head over the fixed model, learning steps kept, updates discarded | 2 | 0.5893 ± 0.0012 | 0.3553 ± 0.0011 | 0.8173 ± 0.0012 | -36.02 ± 0.10 | 60.0 ± 0.0 | 0.2172 ± 0.0003 | 0.2172 ± 0.0003 | 0.53 ± 0.16 | 22.7 ± 0.9 | 31.7 ± 1.7 | 8.70 ± 0.73 |

Paired differences over seeds, own setting minus the other (negative: lower in the own setting):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| Trained through deployment | learning steps kept, updates discarded | -0.1302 ± 0.0086 (n=3) | +7.86 ± 0.51 |
| Same, one-tick workspace gradients | learning steps kept, updates discarded | -0.1421 ± 0.0131 (n=3) | +8.59 ± 0.79 |
| Same, one meta step per window | learning steps kept, updates discarded | -0.0328 ± 0.0098 (n=3) | +1.98 ± 0.59 |
| Same, cheap targets only | learning steps kept, updates discarded | -0.0670 ± 0.0330 (n=2) | +4.04 ± 2.00 |
| Same, four adapter blocks | learning steps kept, updates discarded | -0.1163 ± 0.0087 (n=2) | +7.02 ± 0.53 |
| Same, following values from copies | learning steps kept, updates discarded | -0.1317 ± 0.0000 (n=1) | +7.96 ± 0.00 |
| Same, copies kept, values not followed | learning steps kept, updates discarded | -0.1569 ± 0.0000 (n=1) | +9.48 ± 0.00 |
| New activity head over the fixed model | learning steps kept, updates discarded | -0.1671 ± 0.0045 (n=2) | +10.10 ± 0.25 |
