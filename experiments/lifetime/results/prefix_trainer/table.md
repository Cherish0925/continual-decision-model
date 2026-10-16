| Model | seeds | error | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|
| CDM | 2 | 0.1097 ± 0.0070 | -7.26 ± 0.40 | 3.0 ± 0.0 | 0.2146 ± 0.0003 | 0.2116 ± 0.0009 | 0.98 ± 0.04 | 18.3 ± 1.3 | 11.5 ± 0.7 | 8.25 ± 0.10 |
| CDM, learning switched off | 2 | 0.1097 ± 0.0064 | -7.09 ± 0.37 | 3.0 ± 0.0 | 0.2115 ± 0.0010 | 0.2115 ± 0.0010 | 1.29 ± 0.01 | 18.3 ± 1.4 | 0.0 ± 0.0 | 8.86 ± 0.03 |
| CDM, fixed learning schedule | 2 | 0.1029 ± 0.0001 | -6.89 ± 0.02 | 3.0 ± 0.0 | 0.2139 ± 0.0001 | 0.2113 ± 0.0003 | 0.99 ± 0.06 | 20.8 ± 0.1 | 12.5 ± 0.0 | 8.30 ± 0.20 |
| CDM, fixed learning schedule, learning switched off | 2 | 0.1024 ± 0.0001 | -6.67 ± 0.01 | 3.0 ± 0.0 | 0.2113 ± 0.0004 | 0.2113 ± 0.0004 | 1.31 ± 0.07 | 20.9 ± 0.2 | 0.0 ± 0.0 | 8.92 ± 0.24 |
| CDM without learning | 2 | 0.1021 ± 0.0005 | -6.68 ± 0.02 | 2.5 ± 0.5 | 0.2112 ± 0.0003 | 0.2112 ± 0.0003 | 1.55 ± 0.03 | 18.8 ± 0.4 | 0.0 ± 0.0 | 9.79 ± 0.10 |
| World model, fixed schedules | 2 | 0.1084 ± 0.0039 | -7.09 ± 0.24 | 3.0 ± 0.0 | 0.2145 ± 0.0000 | 0.2117 ± 0.0000 | 0.75 ± 0.00 | 12.5 ± 0.0 | 12.5 ± 0.0 | 7.43 ± 0.00 |
| Recurrent policy | 2 | 0.1667 ± 0.0061 | -10.31 ± 0.36 | 4.0 ± 0.0 | — | — | 0.00 ± 0.00 | 21.4 ± 0.9 | 0.0 ± 0.0 | 4.13 ± 0.00 |
| CDM trained through its learning step | 2 | 0.0982 ± 0.0056 | -6.64 ± 0.35 | 3.0 ± 0.0 | 0.2117 ± 0.0003 | 0.2113 ± 0.0006 | 1.08 ± 0.26 | 20.5 ± 0.2 | 13.2 ± 3.5 | 8.67 ± 0.80 |
| CDM trained through its learning step, learning switched off | 2 | 0.0978 ± 0.0051 | -6.42 ± 0.27 | 2.5 ± 0.5 | 0.2106 ± 0.0004 | 0.2106 ± 0.0004 | 1.46 ± 0.15 | 20.1 ± 0.7 | 0.0 ± 0.0 | 9.47 ± 0.56 |

Paired differences over seeds, own setting minus the other (negative: lower with learning):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| CDM | learning switched off | -0.0001 ± 0.0006 (n=2) | -0.17 ± 0.03 |
| CDM, fixed learning schedule | learning switched off | +0.0005 ± 0.0002 (n=2) | -0.21 ± 0.01 |
| CDM trained through its learning step | learning switched off | +0.0004 ± 0.0005 (n=2) | -0.22 ± 0.09 |
