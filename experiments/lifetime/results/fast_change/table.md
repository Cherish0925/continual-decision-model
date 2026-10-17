| Model | seeds | error | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|
| Full setting | 5 | 0.1349 ± 0.0011 | -8.89 ± 0.08 | 3.0 ± 0.0 | 0.2193 ± 0.0006 | 0.2148 ± 0.0003 | 1.14 ± 0.09 | 19.9 ± 0.7 | 14.4 ± 1.6 | 8.97 ± 0.27 |
| Full setting, learning steps kept, updates discarded | 5 | 0.1334 ± 0.0005 | -8.80 ± 0.04 | 3.0 ± 0.0 | 0.2146 ± 0.0003 | 0.2146 ± 0.0003 | 1.14 ± 0.09 | 19.9 ± 0.7 | 14.3 ± 1.6 | 8.96 ± 0.28 |
| Full setting, learning switched off | 5 | 0.1340 ± 0.0011 | -8.63 ± 0.07 | 3.0 ± 0.0 | 0.2145 ± 0.0003 | 0.2145 ± 0.0003 | 1.53 ± 0.08 | 20.0 ± 0.8 | 0.0 ± 0.0 | 9.76 ± 0.29 |
| Fixed learning schedule | 5 | 0.1327 ± 0.0027 | -8.70 ± 0.16 | 3.0 ± 0.0 | 0.2180 ± 0.0004 | 0.2152 ± 0.0003 | 1.09 ± 0.04 | 18.9 ± 1.2 | 12.5 ± 0.0 | 8.70 ± 0.14 |
| Fixed learning schedule, learning steps kept, updates discarded | 5 | 0.1311 ± 0.0027 | -8.60 ± 0.16 | 3.0 ± 0.0 | 0.2151 ± 0.0003 | 0.2151 ± 0.0003 | 1.09 ± 0.04 | 18.9 ± 1.2 | 12.5 ± 0.0 | 8.70 ± 0.14 |
| Fixed learning schedule, learning switched off | 5 | 0.1328 ± 0.0027 | -8.52 ± 0.16 | 3.0 ± 0.0 | 0.2149 ± 0.0003 | 0.2149 ± 0.0003 | 1.44 ± 0.04 | 18.8 ± 1.2 | 0.0 ± 0.0 | 9.41 ± 0.15 |
| Workspace computation, no learning | 5 | 0.1348 ± 0.0015 | -8.64 ± 0.09 | 3.0 ± 0.0 | 0.2146 ± 0.0002 | 0.2146 ± 0.0002 | 1.49 ± 0.06 | 17.4 ± 0.6 | 0.0 ± 0.0 | 9.59 ± 0.21 |
| World model, fixed schedules | 5 | 0.1403 ± 0.0026 | -9.02 ± 0.15 | 3.2 ± 0.2 | 0.2192 ± 0.0002 | 0.2164 ± 0.0003 | 0.75 ± 0.00 | 12.5 ± 0.0 | 12.5 ± 0.0 | 7.46 ± 0.00 |
| World model, fixed schedules, learning steps kept, updates discarded | 5 | 0.1383 ± 0.0027 | -8.91 ± 0.16 | 3.0 ± 0.0 | 0.2162 ± 0.0002 | 0.2162 ± 0.0002 | 0.75 ± 0.00 | 12.5 ± 0.0 | 12.5 ± 0.0 | 7.46 ± 0.00 |
| Recurrent policy | 5 | 0.1659 ± 0.0070 | -10.24 ± 0.42 | 4.0 ± 0.0 | — | — | 0.00 ± 0.00 | 18.9 ± 0.8 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| Trained through its learning step | 5 | 0.1210 ± 0.0043 | -8.05 ± 0.26 | 3.0 ± 0.0 | 0.2144 ± 0.0004 | 0.2148 ± 0.0006 | 1.09 ± 0.07 | 21.0 ± 0.6 | 14.3 ± 1.6 | 8.79 ± 0.20 |
| Trained through its learning step, learning steps kept, updates discarded | 5 | 0.1218 ± 0.0048 | -8.10 ± 0.29 | 3.0 ± 0.0 | 0.2145 ± 0.0005 | 0.2145 ± 0.0005 | 1.09 ± 0.07 | 21.0 ± 0.6 | 14.4 ± 1.6 | 8.78 ± 0.21 |
| Trained through its learning step, learning switched off | 5 | 0.1219 ± 0.0053 | -7.90 ± 0.32 | 3.0 ± 0.0 | 0.2140 ± 0.0005 | 0.2140 ± 0.0005 | 1.53 ± 0.06 | 21.2 ± 0.6 | 0.0 ± 0.0 | 9.74 ± 0.22 |
| Predictions only | 5 | 0.1329 ± 0.0018 | -8.24 ± 0.10 | 3.0 ± 0.0 | 0.2134 ± 0.0001 | 0.2134 ± 0.0001 | 0.00 ± 0.00 | 19.0 ± 0.6 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| Trained to predict, reading no prediction | 5 | 0.1361 ± 0.0030 | -8.44 ± 0.18 | 3.0 ± 0.0 | 0.2130 ± 0.0002 | 0.2130 ± 0.0002 | 0.00 ± 0.00 | 19.7 ± 0.6 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| Trained through a learned step size | 3 | 0.1253 ± 0.0010 | -8.30 ± 0.05 | 3.0 ± 0.0 | 0.2181 ± 0.0010 | 0.2133 ± 0.0010 | 1.01 ± 0.13 | 21.4 ± 0.7 | 15.0 ± 1.9 | 8.52 ± 0.38 |
| Trained through a learned step size, learning steps kept, updates discarded | 3 | 0.1219 ± 0.0019 | -8.11 ± 0.10 | 3.0 ± 0.0 | 0.2128 ± 0.0008 | 0.2128 ± 0.0008 | 1.00 ± 0.13 | 21.4 ± 0.7 | 15.3 ± 2.1 | 8.49 ± 0.38 |
| Trained through a learned step size, learning switched off | 3 | 0.1214 ± 0.0023 | -7.85 ± 0.14 | 3.0 ± 0.0 | 0.2128 ± 0.0009 | 0.2128 ± 0.0009 | 1.40 ± 0.09 | 21.3 ± 0.8 | 0.0 ± 0.0 | 9.28 ± 0.34 |

Paired differences over seeds, own setting minus the other (negative: lower with learning):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| Full setting | learning switched off | +0.0009 ± 0.0015 (n=5) | -0.26 ± 0.10 |
| Full setting | learning steps kept, updates discarded | +0.0015 ± 0.0010 (n=5) | -0.09 ± 0.06 |
| Fixed learning schedule | learning switched off | -0.0001 ± 0.0008 (n=5) | -0.18 ± 0.05 |
| Fixed learning schedule | learning steps kept, updates discarded | +0.0016 ± 0.0005 (n=5) | -0.10 ± 0.03 |
| World model, fixed schedules | learning steps kept, updates discarded | +0.0020 ± 0.0007 (n=5) | -0.12 ± 0.04 |
| Trained through its learning step | learning switched off | -0.0009 ± 0.0012 (n=5) | -0.14 ± 0.08 |
| Trained through its learning step | learning steps kept, updates discarded | -0.0008 ± 0.0005 (n=5) | +0.05 ± 0.03 |
| Trained through a learned step size | learning switched off | +0.0038 ± 0.0017 (n=3) | -0.46 ± 0.13 |
| Trained through a learned step size | learning steps kept, updates discarded | +0.0033 ± 0.0010 (n=3) | -0.20 ± 0.06 |
