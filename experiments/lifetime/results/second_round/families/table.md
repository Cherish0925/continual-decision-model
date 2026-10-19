| Model | seeds | error | setpoint | largest output | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Trained through deployment, three families | 2 | 0.3009 ± 0.0032 | 0.2640 ± 0.0011 | 0.3380 ± 0.0053 | -18.72 ± 0.19 | 13.5 ± 0.5 | 0.2166 ± 0.0001 | 0.2287 ± 0.0035 | 0.88 ± 0.03 | 19.1 ± 0.3 | 23.2 ± 0.6 | 9.37 ± 0.05 |
| Trained through deployment, three families, learning steps kept, updates discarded | 2 | 0.3955 ± 0.0020 | 0.2979 ± 0.0042 | 0.4935 ± 0.0082 | -24.46 ± 0.12 | 40.0 ± 3.0 | 0.2237 ± 0.0007 | 0.2237 ± 0.0007 | 0.92 ± 0.03 | 19.6 ± 0.1 | 22.7 ± 0.6 | 9.48 ± 0.05 |
| Trained through deployment, three families, refinements kept, workspace not updated | 2 | 0.3821 ± 0.0347 | 0.2884 ± 0.0172 | 0.4761 ± 0.0523 | -23.71 ± 2.27 | 32.0 ± 13.0 | 0.2183 ± 0.0013 | 0.2365 ± 0.0050 | 1.44 ± 0.92 | 17.3 ± 0.5 | 15.5 ± 7.7 | 10.89 ± 2.88 |
| Trained through deployment, three families, no computation between requests | 2 | 0.3147 ± 0.0035 | 0.2677 ± 0.0000 | 0.3620 ± 0.0070 | -19.45 ± 0.21 | 16.0 ± 1.0 | 0.2179 ± 0.0001 | 0.2276 ± 0.0026 | 0.53 ± 0.00 | 19.2 ± 0.2 | 14.3 ± 0.3 | 7.38 ± 0.01 |
| Trained through deployment, three families, no refinement | 2 | 0.3565 ± 0.0083 | 0.2754 ± 0.0036 | 0.4379 ± 0.0129 | -21.94 ± 0.52 | 22.0 ± 3.0 | 0.2178 ± 0.0007 | 0.2460 ± 0.0023 | 0.00 ± 0.00 | 16.8 ± 0.2 | 36.7 ± 4.8 | 7.06 ± 0.36 |
| Predictions only, three families | 2 | 0.3804 ± 0.0022 | 0.2680 ± 0.0150 | 0.4932 ± 0.0107 | -23.29 ± 0.11 | 60.0 ± 0.0 | 0.2151 ± 0.0006 | 0.2151 ± 0.0006 | 0.00 ± 0.00 | 19.8 ± 2.7 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Trained through deployment, vehicles only | 2 | 0.2557 ± 0.0067 | 0.2107 ± 0.0048 | 0.3008 ± 0.0086 | -15.98 ± 0.35 | 9.5 ± 0.5 | 0.2163 ± 0.0006 | 0.2264 ± 0.0003 | 0.97 ± 0.30 | 19.0 ± 1.4 | 21.5 ± 6.1 | 9.57 ± 0.68 |
| Trained through deployment, vehicles only, learning steps kept, updates discarded | 2 | 0.2943 ± 0.0160 | 0.2259 ± 0.0114 | 0.3631 ± 0.0205 | -18.32 ± 0.92 | 13.0 ± 2.0 | 0.2198 ± 0.0010 | 0.2198 ± 0.0010 | 0.93 ± 0.28 | 19.0 ± 1.4 | 22.1 ± 5.8 | 9.48 ± 0.61 |
| Trained through deployment, vehicles only, refinements kept, workspace not updated | 2 | 0.3612 ± 0.0797 | 0.2564 ± 0.0253 | 0.4665 ± 0.1342 | -22.49 ± 4.65 | 35.0 ± 25.0 | 0.2164 ± 0.0010 | 0.2286 ± 0.0026 | 1.83 ± 0.63 | 17.4 ± 7.2 | 8.4 ± 4.6 | 11.82 ± 2.01 |
| Trained through deployment, vehicles only, no computation between requests | 2 | 0.2599 ± 0.0081 | 0.2118 ± 0.0072 | 0.3082 ± 0.0089 | -16.12 ± 0.46 | 10.0 ± 1.0 | 0.2168 ± 0.0009 | 0.2257 ± 0.0001 | 0.54 ± 0.17 | 19.2 ± 1.2 | 13.6 ± 3.3 | 7.37 ± 0.39 |
| Trained through deployment, vehicles only, no refinement | 2 | 0.3552 ± 0.0784 | 0.2523 ± 0.0270 | 0.4587 ± 0.1301 | -21.84 ± 4.74 | 35.0 ± 25.0 | 0.2161 ± 0.0004 | 0.2298 ± 0.0007 | 0.00 ± 0.00 | 17.1 ± 7.5 | 29.0 ± 15.8 | 6.49 ± 1.18 |

Paired differences over seeds, own setting minus the other (negative: lower in the own setting):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| Trained through deployment, three families | learning steps kept, updates discarded | -0.0946 ± 0.0012 (n=2) | +5.75 ± 0.08 |
| Trained through deployment, three families | no refinement | -0.0556 ± 0.0114 (n=2) | +3.22 ± 0.71 |
| Trained through deployment, three families | no computation between requests | -0.0138 ± 0.0003 (n=2) | +0.74 ± 0.02 |
| Trained through deployment, three families | refinements kept, workspace not updated | -0.0812 ± 0.0379 (n=2) | +4.99 ± 2.46 |
| Trained through deployment, vehicles only | learning steps kept, updates discarded | -0.0387 ± 0.0093 (n=2) | +2.34 ± 0.57 |
| Trained through deployment, vehicles only | no refinement | -0.0996 ± 0.0717 (n=2) | +5.86 ± 4.39 |
| Trained through deployment, vehicles only | no computation between requests | -0.0042 ± 0.0014 (n=2) | +0.14 ± 0.10 |
| Trained through deployment, vehicles only | refinements kept, workspace not updated | -0.1056 ± 0.0730 (n=2) | +6.51 ± 4.30 |
