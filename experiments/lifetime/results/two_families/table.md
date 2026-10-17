| Model | seeds | error | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|
| CDM trained through a learned step size | 3 | 0.2193 ± 0.0226 | -13.99 ± 1.38 | 36.7 ± 12.0 | 0.2123 ± 0.0014 | 0.2153 ± 0.0017 | 0.99 ± 0.11 | 21.1 ± 0.3 | 14.1 ± 1.9 | 8.41 ± 0.32 |
| CDM trained through a learned step size, learning steps kept, updates discarded | 3 | 0.2211 ± 0.0202 | -14.10 ± 1.24 | 37.7 ± 11.6 | 0.2137 ± 0.0019 | 0.2137 ± 0.0019 | 0.96 ± 0.11 | 21.1 ± 0.2 | 14.8 ± 2.1 | 8.34 ± 0.32 |
| CDM trained through a learned step size, learning switched off | 3 | 0.2182 ± 0.0190 | -13.71 ± 1.14 | 36.3 ± 10.0 | 0.2133 ± 0.0019 | 0.2133 ± 0.0019 | 1.34 ± 0.06 | 21.0 ± 0.4 | 0.0 ± 0.0 | 9.06 ± 0.24 |
| CDM on both families | 3 | 0.0800 ± 0.0036 | -5.46 ± 0.20 | 7.0 ± 0.0 | 0.2075 ± 0.0014 | 0.2046 ± 0.0005 | 0.97 ± 0.06 | 17.9 ± 2.1 | 11.8 ± 4.8 | 8.22 ± 0.26 |
| CDM on both families, learning steps kept, updates discarded | 3 | 0.0805 ± 0.0044 | -5.49 ± 0.22 | 7.0 ± 0.0 | 0.2046 ± 0.0006 | 0.2046 ± 0.0006 | 0.97 ± 0.06 | 17.9 ± 2.1 | 11.8 ± 4.8 | 8.23 ± 0.26 |
| CDM on both families, learning switched off | 3 | 0.0876 ± 0.0073 | -5.74 ± 0.44 | 7.3 ± 0.3 | 0.2050 ± 0.0005 | 0.2050 ± 0.0005 | 1.29 ± 0.14 | 17.6 ± 2.2 | 0.0 ± 0.0 | 8.87 ± 0.50 |
| CDM on both families, without learning | 3 | 0.0889 ± 0.0031 | -5.89 ± 0.18 | 7.0 ± 0.0 | 0.2044 ± 0.0003 | 0.2044 ± 0.0003 | 1.54 ± 0.04 | 19.4 ± 1.0 | 0.0 ± 0.0 | 9.80 ± 0.16 |
| Recurrent policy on both families | 3 | 0.0870 ± 0.0054 | -5.46 ± 0.35 | 7.3 ± 0.3 | — | — | 0.00 ± 0.00 | 18.5 ± 1.7 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| CDM on thermal processes only | 3 | 0.0670 ± 0.0006 | -4.60 ± 0.01 | 6.0 ± 0.0 | 0.2069 ± 0.0005 | 0.2051 ± 0.0002 | 0.99 ± 0.09 | 18.8 ± 1.1 | 7.6 ± 1.2 | 8.11 ± 0.37 |
| CDM on thermal processes only, learning steps kept, updates discarded | 3 | 0.0663 ± 0.0010 | -4.55 ± 0.04 | 6.0 ± 0.0 | 0.2050 ± 0.0002 | 0.2050 ± 0.0002 | 0.99 ± 0.09 | 18.8 ± 1.1 | 7.6 ± 1.2 | 8.10 ± 0.38 |
| CDM on thermal processes only, learning switched off | 3 | 0.0660 ± 0.0009 | -4.43 ± 0.05 | 6.0 ± 0.0 | 0.2048 ± 0.0002 | 0.2048 ± 0.0002 | 1.16 ± 0.13 | 19.0 ± 1.1 | 0.0 ± 0.0 | 8.41 ± 0.47 |

Paired differences over seeds, own setting minus the other (negative: lower with learning):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| CDM trained through a learned step size | learning switched off | +0.0011 ± 0.0050 (n=3) | -0.28 ± 0.31 |
| CDM trained through a learned step size | learning steps kept, updates discarded | -0.0017 ± 0.0026 (n=3) | +0.11 ± 0.16 |
| CDM on both families | learning switched off | -0.0075 ± 0.0051 (n=3) | +0.28 ± 0.25 |
| CDM on both families | learning steps kept, updates discarded | -0.0005 ± 0.0010 (n=3) | +0.03 ± 0.06 |
| CDM on thermal processes only | learning switched off | +0.0009 ± 0.0004 (n=3) | -0.17 ± 0.04 |
| CDM on thermal processes only | learning steps kept, updates discarded | +0.0007 ± 0.0004 (n=3) | -0.04 ± 0.03 |
