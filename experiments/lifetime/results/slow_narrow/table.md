| Model | seeds | error | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|
| CDM | 5 | 0.2782 ± 0.0029 | -17.51 ± 0.19 | 21.2 ± 1.7 | 0.2130 ± 0.0005 | 0.2106 ± 0.0004 | 1.16 ± 0.12 | 18.9 ± 0.9 | 14.6 ± 2.1 | 9.04 ± 0.36 |
| CDM, learning steps kept, updates discarded | 5 | 0.2772 ± 0.0028 | -17.45 ± 0.19 | 21.0 ± 1.8 | 0.2108 ± 0.0004 | 0.2108 ± 0.0004 | 1.16 ± 0.12 | 18.8 ± 0.9 | 14.6 ± 2.2 | 9.05 ± 0.35 |
| CDM, learning switched off | 5 | 0.2783 ± 0.0042 | -17.30 ± 0.27 | 22.6 ± 3.9 | 0.2105 ± 0.0004 | 0.2105 ± 0.0004 | 1.55 ± 0.10 | 18.9 ± 1.0 | 0.0 ± 0.0 | 9.83 ± 0.36 |
| CDM, fixed learning schedule | 5 | 0.2746 ± 0.0053 | -17.24 ± 0.31 | 20.6 ± 2.8 | 0.2120 ± 0.0003 | 0.2114 ± 0.0002 | 1.21 ± 0.05 | 17.0 ± 1.5 | 12.5 ± 0.0 | 9.13 ± 0.18 |
| CDM, fixed learning schedule, learning steps kept, updates discarded | 5 | 0.2722 ± 0.0055 | -17.10 ± 0.32 | 19.4 ± 2.6 | 0.2114 ± 0.0003 | 0.2114 ± 0.0003 | 1.21 ± 0.05 | 17.1 ± 1.6 | 12.5 ± 0.0 | 9.14 ± 0.19 |
| CDM, fixed learning schedule, learning switched off | 5 | 0.2740 ± 0.0060 | -17.03 ± 0.36 | 19.4 ± 3.0 | 0.2114 ± 0.0003 | 0.2114 ± 0.0003 | 1.58 ± 0.06 | 17.0 ± 1.5 | 0.0 ± 0.0 | 9.92 ± 0.22 |
| CDM without learning | 5 | 0.2756 ± 0.0071 | -17.12 ± 0.43 | 21.0 ± 2.8 | 0.2106 ± 0.0004 | 0.2106 ± 0.0004 | 1.62 ± 0.10 | 15.7 ± 1.2 | 0.0 ± 0.0 | 10.08 ± 0.35 |
| World model, fixed schedules | 5 | 0.2781 ± 0.0057 | -17.32 ± 0.34 | 20.0 ± 2.1 | 0.2128 ± 0.0003 | 0.2124 ± 0.0005 | 0.75 ± 0.00 | 12.5 ± 0.0 | 12.5 ± 0.0 | 7.46 ± 0.00 |
| World model, fixed schedules, learning steps kept, updates discarded | 5 | 0.2770 ± 0.0047 | -17.25 ± 0.29 | 19.6 ± 1.7 | 0.2122 ± 0.0004 | 0.2122 ± 0.0004 | 0.75 ± 0.00 | 12.5 ± 0.0 | 12.5 ± 0.0 | 7.46 ± 0.00 |
| Recurrent policy | 5 | 0.3001 ± 0.0102 | -18.29 ± 0.61 | 32.1 ± 7.6 | — | — | 0.00 ± 0.00 | 17.6 ± 0.8 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| CDM trained through its learning step | 5 | 0.2621 ± 0.0151 | -16.54 ± 0.91 | 19.2 ± 7.2 | 0.2090 ± 0.0005 | 0.2107 ± 0.0004 | 1.17 ± 0.05 | 20.7 ± 0.7 | 14.0 ± 0.8 | 9.05 ± 0.21 |
| CDM trained through its learning step, learning steps kept, updates discarded | 5 | 0.2651 ± 0.0159 | -16.73 ± 0.96 | 20.6 ± 8.4 | 0.2099 ± 0.0003 | 0.2099 ± 0.0003 | 1.15 ± 0.05 | 20.7 ± 0.7 | 14.3 ± 0.8 | 9.00 ± 0.21 |
| CDM trained through its learning step, learning switched off | 5 | 0.2666 ± 0.0166 | -16.63 ± 1.00 | 21.4 ± 8.9 | 0.2095 ± 0.0005 | 0.2095 ± 0.0005 | 1.62 ± 0.08 | 20.9 ± 0.6 | 0.0 ± 0.0 | 10.09 ± 0.30 |
| Predictions without workspace computation or learning | 5 | 0.2676 ± 0.0024 | -16.33 ± 0.14 | 17.4 ± 0.8 | 0.2091 ± 0.0003 | 0.2091 ± 0.0003 | 0.00 ± 0.00 | 17.5 ± 0.6 | 0.0 ± 0.0 | 4.15 ± 0.00 |
| CDM trained through a learned step size | 3 | 0.2598 ± 0.0161 | -16.39 ± 0.95 | 18.0 ± 7.5 | 0.2115 ± 0.0011 | 0.2087 ± 0.0010 | 1.02 ± 0.15 | 20.4 ± 0.7 | 14.7 ± 2.2 | 8.56 ± 0.45 |
| CDM trained through a learned step size, learning steps kept, updates discarded | 3 | 0.2546 ± 0.0155 | -16.08 ± 0.91 | 16.3 ± 5.8 | 0.2082 ± 0.0008 | 0.2082 ± 0.0008 | 1.01 ± 0.14 | 20.5 ± 0.7 | 15.2 ± 2.1 | 8.52 ± 0.43 |
| CDM trained through a learned step size, learning switched off | 3 | 0.2574 ± 0.0165 | -16.03 ± 0.97 | 17.7 ± 6.7 | 0.2080 ± 0.0008 | 0.2080 ± 0.0008 | 1.40 ± 0.12 | 20.3 ± 0.9 | 0.0 ± 0.0 | 9.27 ± 0.42 |

Paired differences over seeds, own setting minus the other (negative: lower with learning):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| CDM | learning switched off | -0.0000 ± 0.0050 (n=5) | -0.21 ± 0.32 |
| CDM | learning steps kept, updates discarded | +0.0010 ± 0.0018 (n=5) | -0.06 ± 0.11 |
| CDM, fixed learning schedule | learning switched off | +0.0005 ± 0.0028 (n=5) | -0.21 ± 0.17 |
| CDM, fixed learning schedule | learning steps kept, updates discarded | +0.0024 ± 0.0007 (n=5) | -0.14 ± 0.04 |
| World model, fixed schedules | learning steps kept, updates discarded | +0.0012 ± 0.0011 (n=5) | -0.07 ± 0.07 |
| CDM trained through its learning step | learning switched off | -0.0045 ± 0.0021 (n=5) | +0.09 ± 0.13 |
| CDM trained through its learning step | learning steps kept, updates discarded | -0.0031 ± 0.0016 (n=5) | +0.19 ± 0.10 |
| CDM trained through a learned step size | learning switched off | +0.0024 ± 0.0004 (n=3) | -0.36 ± 0.05 |
| CDM trained through a learned step size | learning steps kept, updates discarded | +0.0053 ± 0.0015 (n=3) | -0.31 ± 0.10 |
