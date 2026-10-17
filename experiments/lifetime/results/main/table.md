| Model | seeds | error | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|
| Full setting | 5 | 0.1078 ± 0.0009 | -7.23 ± 0.06 | 3.0 ± 0.0 | 0.2153 ± 0.0006 | 0.2112 ± 0.0002 | 1.14 ± 0.09 | 19.9 ± 0.6 | 13.5 ± 1.5 | 8.94 ± 0.26 |
| Full setting, learning steps kept, updates discarded | 5 | 0.1062 ± 0.0006 | -7.13 ± 0.04 | 2.8 ± 0.2 | 0.2111 ± 0.0002 | 0.2111 ± 0.0002 | 1.14 ± 0.09 | 19.9 ± 0.6 | 13.5 ± 1.5 | 8.93 ± 0.26 |
| Full setting, learning switched off | 5 | 0.1078 ± 0.0009 | -7.03 ± 0.05 | 3.0 ± 0.0 | 0.2110 ± 0.0002 | 0.2110 ± 0.0002 | 1.50 ± 0.07 | 20.0 ± 0.8 | 0.0 ± 0.0 | 9.65 ± 0.27 |
| Fixed learning schedule | 5 | 0.1068 ± 0.0027 | -7.12 ± 0.16 | 2.8 ± 0.2 | 0.2144 ± 0.0004 | 0.2115 ± 0.0003 | 1.08 ± 0.03 | 18.6 ± 1.2 | 12.5 ± 0.0 | 8.66 ± 0.11 |
| Fixed learning schedule, learning steps kept, updates discarded | 5 | 0.1061 ± 0.0031 | -7.08 ± 0.18 | 2.8 ± 0.2 | 0.2114 ± 0.0004 | 0.2114 ± 0.0004 | 1.08 ± 0.03 | 18.6 ± 1.2 | 12.5 ± 0.0 | 8.67 ± 0.11 |
| Fixed learning schedule, learning switched off | 5 | 0.1072 ± 0.0033 | -6.97 ± 0.20 | 2.8 ± 0.2 | 0.2113 ± 0.0003 | 0.2113 ± 0.0003 | 1.42 ± 0.03 | 18.5 ± 1.2 | 0.0 ± 0.0 | 9.37 ± 0.11 |
| Workspace computation, no learning | 5 | 0.1061 ± 0.0011 | -6.90 ± 0.06 | 3.0 ± 0.0 | 0.2110 ± 0.0001 | 0.2110 ± 0.0001 | 1.47 ± 0.05 | 17.4 ± 0.7 | 0.0 ± 0.0 | 9.54 ± 0.20 |
| World model, fixed schedules | 5 | 0.1122 ± 0.0025 | -7.32 ± 0.15 | 3.0 ± 0.0 | 0.2155 ± 0.0003 | 0.2124 ± 0.0003 | 0.75 ± 0.00 | 12.5 ± 0.0 | 12.5 ± 0.0 | 7.45 ± 0.00 |
| World model, fixed schedules, learning steps kept, updates discarded | 5 | 0.1106 ± 0.0022 | -7.23 ± 0.13 | 3.0 ± 0.0 | 0.2123 ± 0.0002 | 0.2123 ± 0.0002 | 0.75 ± 0.00 | 12.5 ± 0.0 | 12.5 ± 0.0 | 7.45 ± 0.00 |
| Recurrent policy | 5 | 0.1461 ± 0.0084 | -9.04 ± 0.50 | 3.8 ± 0.2 | — | — | 0.00 ± 0.00 | 18.9 ± 0.8 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| Trained through its learning step | 5 | 0.0971 ± 0.0038 | -6.59 ± 0.23 | 2.4 ± 0.2 | 0.2116 ± 0.0006 | 0.2111 ± 0.0004 | 1.11 ± 0.06 | 20.6 ± 0.6 | 13.7 ± 1.4 | 8.82 ± 0.18 |
| Trained through its learning step, learning steps kept, updates discarded | 5 | 0.0974 ± 0.0045 | -6.60 ± 0.28 | 2.4 ± 0.2 | 0.2105 ± 0.0003 | 0.2105 ± 0.0003 | 1.10 ± 0.06 | 20.6 ± 0.6 | 13.9 ± 1.5 | 8.80 ± 0.18 |
| Trained through its learning step, learning switched off | 5 | 0.0980 ± 0.0051 | -6.45 ± 0.31 | 2.4 ± 0.2 | 0.2102 ± 0.0003 | 0.2102 ± 0.0003 | 1.52 ± 0.05 | 20.8 ± 0.5 | 0.0 ± 0.0 | 9.71 ± 0.17 |
| Predictions only | 5 | 0.1058 ± 0.0018 | -6.60 ± 0.10 | 3.0 ± 0.0 | 0.2099 ± 0.0001 | 0.2099 ± 0.0001 | 0.00 ± 0.00 | 18.7 ± 0.7 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| Trained to predict, reading no prediction | 5 | 0.1088 ± 0.0024 | -6.79 ± 0.14 | 3.0 ± 0.0 | 0.2097 ± 0.0001 | 0.2097 ± 0.0001 | 0.00 ± 0.00 | 19.4 ± 0.5 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| Trained through a learned step size | 3 | 0.1022 ± 0.0008 | -6.89 ± 0.03 | 3.0 ± 0.0 | 0.2153 ± 0.0012 | 0.2096 ± 0.0007 | 1.01 ± 0.11 | 21.1 ± 0.8 | 14.4 ± 1.5 | 8.50 ± 0.33 |
| Trained through a learned step size, learning steps kept, updates discarded | 3 | 0.0985 ± 0.0016 | -6.68 ± 0.07 | 2.7 ± 0.3 | 0.2091 ± 0.0006 | 0.2091 ± 0.0006 | 1.00 ± 0.11 | 21.2 ± 0.9 | 14.6 ± 1.7 | 8.48 ± 0.33 |
| Trained through a learned step size, learning switched off | 3 | 0.0987 ± 0.0013 | -6.47 ± 0.08 | 2.7 ± 0.3 | 0.2091 ± 0.0006 | 0.2091 ± 0.0006 | 1.39 ± 0.08 | 21.0 ± 1.0 | 0.0 ± 0.0 | 9.23 ± 0.30 |

Paired differences over seeds, own setting minus the other (negative: lower with learning):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| Full setting | learning switched off | -0.0000 ± 0.0011 (n=5) | -0.20 ± 0.08 |
| Full setting | learning steps kept, updates discarded | +0.0015 ± 0.0003 (n=5) | -0.09 ± 0.02 |
| Fixed learning schedule | learning switched off | -0.0004 ± 0.0010 (n=5) | -0.15 ± 0.06 |
| Fixed learning schedule | learning steps kept, updates discarded | +0.0006 ± 0.0006 (n=5) | -0.04 ± 0.03 |
| World model, fixed schedules | learning steps kept, updates discarded | +0.0016 ± 0.0004 (n=5) | -0.09 ± 0.02 |
| Trained through its learning step | learning switched off | -0.0009 ± 0.0015 (n=5) | -0.13 ± 0.09 |
| Trained through its learning step | learning steps kept, updates discarded | -0.0002 ± 0.0009 (n=5) | +0.02 ± 0.05 |
| Trained through a learned step size | learning switched off | +0.0034 ± 0.0006 (n=3) | -0.42 ± 0.06 |
| Trained through a learned step size | learning steps kept, updates discarded | +0.0036 ± 0.0008 (n=3) | -0.22 ± 0.04 |
