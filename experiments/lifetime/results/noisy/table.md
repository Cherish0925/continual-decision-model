| Model | seeds | error | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|
| CDM | 5 | 0.1249 ± 0.0006 | -8.28 ± 0.05 | 3.0 ± 0.0 | 0.3380 ± 0.0005 | 0.3312 ± 0.0004 | 1.17 ± 0.10 | 19.8 ± 0.8 | 15.4 ± 2.0 | 9.11 ± 0.30 |
| CDM, learning steps kept, updates discarded | 5 | 0.1188 ± 0.0008 | -7.90 ± 0.05 | 3.0 ± 0.0 | 0.3305 ± 0.0004 | 0.3305 ± 0.0004 | 1.16 ± 0.09 | 19.8 ± 0.8 | 14.9 ± 1.9 | 9.07 ± 0.28 |
| CDM, learning switched off | 5 | 0.1199 ± 0.0014 | -7.76 ± 0.10 | 3.0 ± 0.0 | 0.3304 ± 0.0004 | 0.3304 ± 0.0004 | 1.59 ± 0.08 | 19.9 ± 0.9 | 0.0 ± 0.0 | 9.99 ± 0.30 |
| CDM, fixed learning schedule | 5 | 0.1287 ± 0.0055 | -8.43 ± 0.33 | 3.0 ± 0.0 | 0.3371 ± 0.0009 | 0.3321 ± 0.0005 | 1.06 ± 0.07 | 19.2 ± 1.3 | 12.5 ± 0.0 | 8.60 ± 0.26 |
| CDM, fixed learning schedule, learning steps kept, updates discarded | 5 | 0.1243 ± 0.0064 | -8.16 ± 0.38 | 3.0 ± 0.0 | 0.3313 ± 0.0006 | 0.3313 ± 0.0006 | 1.06 ± 0.06 | 19.2 ± 1.4 | 12.5 ± 0.0 | 8.58 ± 0.21 |
| CDM, fixed learning schedule, learning switched off | 5 | 0.1257 ± 0.0069 | -8.06 ± 0.42 | 3.0 ± 0.0 | 0.3311 ± 0.0004 | 0.3311 ± 0.0004 | 1.39 ± 0.07 | 19.2 ± 1.4 | 0.0 ± 0.0 | 9.25 ± 0.26 |
| CDM without learning | 5 | 0.1214 ± 0.0027 | -7.80 ± 0.16 | 3.0 ± 0.0 | 0.3312 ± 0.0003 | 0.3312 ± 0.0003 | 1.46 ± 0.07 | 17.6 ± 0.6 | 0.0 ± 0.0 | 9.49 ± 0.26 |
| World model, fixed schedules | 5 | 0.1350 ± 0.0026 | -8.67 ± 0.16 | 3.6 ± 0.2 | 0.3380 ± 0.0004 | 0.3341 ± 0.0004 | 0.75 ± 0.00 | 12.5 ± 0.0 | 12.5 ± 0.0 | 7.46 ± 0.00 |
| World model, fixed schedules, learning steps kept, updates discarded | 5 | 0.1309 ± 0.0016 | -8.43 ± 0.10 | 3.2 ± 0.2 | 0.3333 ± 0.0004 | 0.3333 ± 0.0004 | 0.75 ± 0.00 | 12.5 ± 0.0 | 12.5 ± 0.0 | 7.46 ± 0.00 |
| Recurrent policy | 5 | 0.1616 ± 0.0082 | -9.95 ± 0.49 | 4.0 ± 0.0 | — | — | 0.00 ± 0.00 | 19.1 ± 0.8 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| CDM trained through its learning step | 5 | 0.1118 ± 0.0053 | -7.50 ± 0.32 | 3.0 ± 0.0 | 0.3314 ± 0.0005 | 0.3301 ± 0.0003 | 1.07 ± 0.09 | 21.6 ± 0.7 | 15.7 ± 2.5 | 8.78 ± 0.27 |
| CDM trained through its learning step, learning steps kept, updates discarded | 5 | 0.1102 ± 0.0063 | -7.38 ± 0.39 | 3.0 ± 0.0 | 0.3292 ± 0.0003 | 0.3292 ± 0.0003 | 1.08 ± 0.08 | 21.4 ± 0.6 | 14.7 ± 1.9 | 8.78 ± 0.24 |
| CDM trained through its learning step, learning switched off | 5 | 0.1103 ± 0.0066 | -7.18 ± 0.40 | 3.0 ± 0.0 | 0.3290 ± 0.0003 | 0.3290 ± 0.0003 | 1.53 ± 0.08 | 21.6 ± 0.5 | 0.0 ± 0.0 | 9.77 ± 0.28 |
| Predictions without workspace computation or learning | 5 | 0.1202 ± 0.0025 | -7.45 ± 0.14 | 3.0 ± 0.0 | 0.3299 ± 0.0002 | 0.3299 ± 0.0002 | 0.00 ± 0.00 | 19.0 ± 0.8 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| CDM trained through a learned step size | 3 | 0.1127 ± 0.0013 | -7.56 ± 0.10 | 3.0 ± 0.0 | 0.3344 ± 0.0019 | 0.3293 ± 0.0009 | 1.00 ± 0.16 | 21.9 ± 0.5 | 17.1 ± 2.7 | 8.59 ± 0.45 |
| CDM trained through a learned step size, learning steps kept, updates discarded | 3 | 0.1078 ± 0.0006 | -7.25 ± 0.05 | 3.0 ± 0.0 | 0.3284 ± 0.0009 | 0.3284 ± 0.0009 | 1.00 ± 0.15 | 21.8 ± 0.6 | 16.2 ± 2.5 | 8.55 ± 0.43 |
| CDM trained through a learned step size, learning switched off | 3 | 0.1075 ± 0.0004 | -6.99 ± 0.03 | 3.0 ± 0.0 | 0.3283 ± 0.0009 | 0.3283 ± 0.0009 | 1.43 ± 0.10 | 21.7 ± 0.7 | 0.0 ± 0.0 | 9.40 ± 0.37 |

Paired differences over seeds, own setting minus the other (negative: lower with learning):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| CDM | learning switched off | +0.0050 ± 0.0014 (n=5) | -0.53 ± 0.10 |
| CDM | learning steps kept, updates discarded | +0.0061 ± 0.0011 (n=5) | -0.38 ± 0.07 |
| CDM, fixed learning schedule | learning switched off | +0.0030 ± 0.0018 (n=5) | -0.37 ± 0.11 |
| CDM, fixed learning schedule | learning steps kept, updates discarded | +0.0045 ± 0.0012 (n=5) | -0.27 ± 0.07 |
| World model, fixed schedules | learning steps kept, updates discarded | +0.0041 ± 0.0011 (n=5) | -0.24 ± 0.06 |
| CDM trained through its learning step | learning switched off | +0.0015 ± 0.0014 (n=5) | -0.31 ± 0.09 |
| CDM trained through its learning step | learning steps kept, updates discarded | +0.0016 ± 0.0011 (n=5) | -0.12 ± 0.07 |
| CDM trained through a learned step size | learning switched off | +0.0052 ± 0.0009 (n=3) | -0.57 ± 0.08 |
| CDM trained through a learned step size | learning steps kept, updates discarded | +0.0049 ± 0.0007 (n=3) | -0.31 ± 0.04 |
