| Model | seeds | error | setpoint | largest output | reward/100 | response | prediction | shared alone | refine/tick | measure/100 | learn/100 | GPU µs/tick |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Recurrent policy | 2 | 0.3980 ± 0.0066 | 0.2218 ± 0.0039 | 0.5738 ± 0.0093 | -24.16 ± 0.37 | 60.0 ± 0.0 | — | — | 0.00 ± 0.00 | 11.8 ± 2.5 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Predictions only | 3 | 0.3069 ± 0.0371 | 0.1619 ± 0.0152 | 0.4516 ± 0.0592 | -18.68 ± 2.23 | 38.0 ± 14.0 | 0.2101 ± 0.0002 | 0.2101 ± 0.0002 | 0.00 ± 0.00 | 13.7 ± 2.0 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Trained to predict, reading no prediction | 2 | 0.2968 ± 0.0369 | 0.1560 ± 0.0136 | 0.4374 ± 0.0601 | -18.00 ± 2.16 | 25.5 ± 10.5 | 0.2109 ± 0.0015 | 0.2109 ± 0.0015 | 0.00 ± 0.00 | 7.2 ± 6.8 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Trained through deployment | 3 | 0.1766 ± 0.0038 | 0.1321 ± 0.0015 | 0.2210 ± 0.0088 | -11.12 ± 0.24 | 6.7 ± 0.3 | 0.2123 ± 0.0004 | 0.2238 ± 0.0022 | 0.76 ± 0.03 | 20.3 ± 0.9 | 27.4 ± 1.1 | 9.23 ± 0.03 |
| Trained through deployment, learning steps kept, updates discarded | 3 | 0.2812 ± 0.0121 | 0.1648 ± 0.0048 | 0.3974 ± 0.0202 | -17.44 ± 0.74 | 13.0 ± 1.5 | 0.2223 ± 0.0009 | 0.2223 ± 0.0009 | 0.74 ± 0.03 | 20.3 ± 0.8 | 28.0 ± 1.1 | 9.20 ± 0.05 |
| Trained through deployment, refinements kept, workspace not updated | 3 | 0.2594 ± 0.0664 | 0.1728 ± 0.0381 | 0.3458 ± 0.0953 | -16.18 ± 4.04 | 22.7 ± 15.7 | 0.2126 ± 0.0004 | 0.2276 ± 0.0020 | 1.21 ± 0.23 | 18.7 ± 2.2 | 21.9 ± 3.0 | 10.51 ± 0.67 |
| Trained through deployment, no computation between requests | 3 | 0.1870 ± 0.0034 | 0.1335 ± 0.0013 | 0.2403 ± 0.0079 | -11.65 ± 0.21 | 7.0 ± 0.0 | 0.2137 ± 0.0003 | 0.2240 ± 0.0022 | 0.45 ± 0.02 | 20.3 ± 0.9 | 16.8 ± 0.6 | 7.28 ± 0.03 |
| Trained through deployment, no refinement | 3 | 0.2547 ± 0.0664 | 0.1697 ± 0.0373 | 0.3394 ± 0.0960 | -15.70 ± 3.99 | 20.7 ± 13.7 | 0.2121 ± 0.0005 | 0.2291 ± 0.0024 | 0.00 ± 0.00 | 18.9 ± 2.1 | 36.1 ± 3.9 | 7.02 ± 0.29 |
| Same, one-tick workspace gradients | 3 | 0.1534 ± 0.0030 | 0.1177 ± 0.0006 | 0.1890 ± 0.0054 | -9.68 ± 0.17 | 5.0 ± 0.0 | 0.2109 ± 0.0004 | 0.2285 ± 0.0020 | 0.62 ± 0.09 | 18.7 ± 1.0 | 28.4 ± 3.0 | 8.79 ± 0.15 |
| Same, one-tick workspace gradients, learning steps kept, updates discarded | 3 | 0.3072 ± 0.0101 | 0.1749 ± 0.0036 | 0.4392 ± 0.0171 | -18.98 ± 0.60 | 16.7 ± 1.9 | 0.2219 ± 0.0007 | 0.2219 ± 0.0007 | 0.61 ± 0.08 | 18.8 ± 1.3 | 29.5 ± 2.2 | 8.83 ± 0.16 |
| Same, one-tick workspace gradients, refinements kept, workspace not updated | 3 | 0.1994 ± 0.0081 | 0.1541 ± 0.0073 | 0.2445 ± 0.0091 | -12.55 ± 0.60 | 6.7 ± 0.3 | 0.2132 ± 0.0007 | 0.2652 ± 0.0059 | 1.27 ± 0.58 | 16.3 ± 4.0 | 24.1 ± 6.2 | 10.91 ± 1.73 |
| Same, one-tick workspace gradients, no computation between requests | 3 | 0.1655 ± 0.0045 | 0.1210 ± 0.0007 | 0.2100 ± 0.0082 | -10.33 ± 0.27 | 6.0 ± 0.0 | 0.2122 ± 0.0004 | 0.2282 ± 0.0016 | 0.37 ± 0.04 | 18.7 ± 1.0 | 17.6 ± 1.2 | 7.03 ± 0.07 |
| Same, one-tick workspace gradients, no refinement | 3 | 0.1914 ± 0.0054 | 0.1516 ± 0.0089 | 0.2310 ± 0.0021 | -11.88 ± 0.35 | 6.3 ± 0.7 | 0.2123 ± 0.0010 | 0.2614 ± 0.0064 | 0.00 ± 0.00 | 16.6 ± 3.7 | 41.3 ± 2.2 | 7.40 ± 0.17 |
| Same, no learning | 3 | 0.3260 ± 0.0047 | 0.1770 ± 0.0032 | 0.4747 ± 0.0084 | -20.20 ± 0.31 | 28.3 ± 2.4 | 0.2125 ± 0.0005 | 0.2125 ± 0.0005 | 1.59 ± 0.13 | 18.1 ± 1.0 | 0.0 ± 0.0 | 10.28 ± 0.47 |
| Same, no learning, refinements kept, workspace not updated | 3 | 0.3558 ± 0.0039 | 0.2029 ± 0.0084 | 0.5084 ± 0.0028 | -21.98 ± 0.21 | 46.0 ± 5.6 | 0.2241 ± 0.0014 | 0.2241 ± 0.0014 | 1.46 ± 0.26 | 19.5 ± 2.0 | 0.0 ± 0.0 | 9.80 ± 0.99 |
| Same, no learning, no computation between requests | 3 | 0.3277 ± 0.0043 | 0.1784 ± 0.0034 | 0.4766 ± 0.0076 | -20.17 ± 0.27 | 29.0 ± 2.1 | 0.2127 ± 0.0005 | 0.2127 ± 0.0005 | 0.97 ± 0.04 | 18.1 ± 1.1 | 0.0 ± 0.0 | 7.98 ± 0.17 |
| Same, no learning, no refinement | 3 | 0.3637 ± 0.0063 | 0.2072 ± 0.0091 | 0.5198 ± 0.0062 | -22.17 ± 0.39 | 50.3 ± 4.8 | 0.2243 ± 0.0014 | 0.2243 ± 0.0014 | 0.00 ± 0.00 | 19.3 ± 2.1 | 0.0 ± 0.0 | 4.31 ± 0.00 |
| Same, one meta step per window | 3 | 0.1864 ± 0.0066 | 0.1317 ± 0.0029 | 0.2409 ± 0.0103 | -11.69 ± 0.41 | 7.0 ± 0.0 | 0.2126 ± 0.0004 | 0.2122 ± 0.0001 | 0.77 ± 0.05 | 18.5 ± 1.6 | 25.1 ± 1.1 | 9.09 ± 0.12 |
| Same, one meta step per window, learning steps kept, updates discarded | 3 | 0.2336 ± 0.0179 | 0.1442 ± 0.0052 | 0.3228 ± 0.0309 | -14.54 ± 1.09 | 9.3 ± 1.3 | 0.2147 ± 0.0001 | 0.2147 ± 0.0001 | 0.76 ± 0.06 | 18.6 ± 1.6 | 25.8 ± 1.3 | 9.09 ± 0.12 |
| Same, one meta step per window, refinements kept, workspace not updated | 3 | 0.2024 ± 0.0154 | 0.1381 ± 0.0065 | 0.2666 ± 0.0249 | -12.64 ± 0.93 | 6.7 ± 0.3 | 0.2133 ± 0.0003 | 0.2319 ± 0.0028 | 0.65 ± 0.11 | 18.3 ± 1.0 | 25.3 ± 1.3 | 8.67 ± 0.34 |
| Same, one meta step per window, no computation between requests | 3 | 0.1912 ± 0.0064 | 0.1323 ± 0.0030 | 0.2499 ± 0.0097 | -11.88 ± 0.40 | 7.3 ± 0.3 | 0.2132 ± 0.0003 | 0.2126 ± 0.0002 | 0.46 ± 0.02 | 18.5 ± 1.6 | 15.8 ± 0.4 | 7.21 ± 0.03 |
| Same, one meta step per window, no refinement | 3 | 0.1994 ± 0.0144 | 0.1366 ± 0.0053 | 0.2621 ± 0.0240 | -12.36 ± 0.88 | 6.7 ± 0.3 | 0.2133 ± 0.0004 | 0.2328 ± 0.0030 | 0.00 ± 0.00 | 18.3 ± 1.1 | 36.6 ± 0.7 | 7.05 ± 0.05 |
| Same, cheap targets only | 2 | 0.2223 ± 0.0227 | 0.1604 ± 0.0094 | 0.2841 ± 0.0359 | -13.84 ± 1.38 | 8.5 ± 0.5 | 0.2158 ± 0.0004 | 0.2355 ± 0.0002 | 0.77 ± 0.07 | 17.0 ± 0.2 | 23.6 ± 1.6 | 8.98 ± 0.14 |
| Same, cheap targets only, learning steps kept, updates discarded | 2 | 0.3105 ± 0.0297 | 0.1865 ± 0.0129 | 0.4342 ± 0.0464 | -19.17 ± 1.79 | 18.5 ± 6.5 | 0.2257 ± 0.0005 | 0.2257 ± 0.0005 | 0.74 ± 0.03 | 16.9 ± 0.7 | 24.9 ± 0.2 | 8.97 ± 0.08 |
| Same, cheap targets only, refinements kept, workspace not updated | 2 | 0.2504 ± 0.0285 | 0.1696 ± 0.0120 | 0.3311 ± 0.0449 | -15.66 ± 1.85 | 9.5 ± 1.5 | 0.2174 ± 0.0002 | 0.2780 ± 0.0423 | 1.45 ± 0.54 | 18.1 ± 3.2 | 16.5 ± 2.4 | 11.00 ± 1.87 |
| Same, cheap targets only, no computation between requests | 2 | 0.2358 ± 0.0236 | 0.1653 ± 0.0099 | 0.3062 ± 0.0372 | -14.56 ± 1.43 | 9.5 ± 1.5 | 0.2172 ± 0.0004 | 0.2368 ± 0.0007 | 0.45 ± 0.08 | 16.9 ± 0.1 | 14.7 ± 0.8 | 7.12 ± 0.23 |
| Same, cheap targets only, no refinement | 2 | 0.2451 ± 0.0255 | 0.1674 ± 0.0102 | 0.3227 ± 0.0408 | -15.10 ± 1.55 | 9.0 ± 1.0 | 0.2178 ± 0.0003 | 0.2719 ± 0.0354 | 0.00 ± 0.00 | 17.5 ± 2.5 | 33.0 ± 3.8 | 6.79 ± 0.29 |
| Same, four adapter blocks | 2 | 0.1739 ± 0.0049 | 0.1428 ± 0.0026 | 0.2049 ± 0.0072 | -10.94 ± 0.33 | 7.5 ± 0.5 | 0.2117 ± 0.0006 | 0.2228 ± 0.0014 | 0.76 ± 0.07 | 18.9 ± 3.3 | 26.4 ± 3.0 | 9.13 ± 0.03 |
| Same, four adapter blocks, learning steps kept, updates discarded | 2 | 0.2654 ± 0.0182 | 0.1650 ± 0.0063 | 0.3655 ± 0.0301 | -16.46 ± 1.13 | 12.5 ± 2.5 | 0.2199 ± 0.0011 | 0.2199 ± 0.0011 | 0.70 ± 0.06 | 19.4 ± 2.9 | 27.4 ± 2.9 | 8.99 ± 0.00 |
| Same, four adapter blocks, refinements kept, workspace not updated | 2 | 0.2526 ± 0.0549 | 0.1693 ± 0.0215 | 0.3358 ± 0.0881 | -15.75 ± 3.37 | 13.5 ± 5.5 | 0.2182 ± 0.0044 | 0.2371 ± 0.0037 | 1.72 ± 0.82 | 9.2 ± 6.7 | 14.4 ± 11.9 | 11.87 ± 2.19 |
| Same, four adapter blocks, no computation between requests | 2 | 0.1825 ± 0.0062 | 0.1434 ± 0.0030 | 0.2215 ± 0.0094 | -11.36 ± 0.40 | 7.5 ± 0.5 | 0.2130 ± 0.0003 | 0.2231 ± 0.0018 | 0.42 ± 0.03 | 18.8 ± 3.2 | 16.6 ± 1.3 | 7.15 ± 0.02 |
| Same, four adapter blocks, no refinement | 2 | 0.2229 ± 0.0337 | 0.1542 ± 0.0075 | 0.2916 ± 0.0599 | -13.67 ± 1.92 | 10.5 ± 2.5 | 0.2159 ± 0.0028 | 0.2380 ± 0.0029 | 0.00 ± 0.00 | 9.9 ± 6.4 | 27.2 ± 13.8 | 6.35 ± 1.04 |
| Same, following values from copies | 1 | 0.2244 ± 0.0000 | 0.1659 ± 0.0000 | 0.2828 ± 0.0000 | -14.03 ± 0.00 | 7.0 ± 0.0 | 0.2142 ± 0.0000 | 0.2245 ± 0.0000 | 0.64 ± 0.00 | 22.8 ± 0.0 | 31.9 ± 0.0 | 9.10 ± 0.00 |
| Same, following values from copies, learning steps kept, updates discarded | 1 | 0.4114 ± 0.0000 | 0.2472 ± 0.0000 | 0.5753 ± 0.0000 | -25.33 ± 0.00 | 39.0 ± 0.0 | 0.2270 ± 0.0000 | 0.2270 ± 0.0000 | 0.66 ± 0.00 | 22.6 ± 0.0 | 31.8 ± 0.0 | 9.18 ± 0.00 |
| Same, following values from copies, refinements kept, workspace not updated | 1 | 0.3775 ± 0.0000 | 0.2334 ± 0.0000 | 0.5213 ± 0.0000 | -23.27 ± 0.00 | 21.0 ± 0.0 | 0.2134 ± 0.0000 | 0.2578 ± 0.0000 | 0.80 ± 0.00 | 22.2 ± 0.0 | 23.5 ± 0.0 | 9.08 ± 0.00 |
| Same, following values from copies, no computation between requests | 1 | 0.2507 ± 0.0000 | 0.1749 ± 0.0000 | 0.3263 ± 0.0000 | -15.52 ± 0.00 | 9.0 ± 0.0 | 0.2166 ± 0.0000 | 0.2252 ± 0.0000 | 0.40 ± 0.00 | 22.7 ± 0.0 | 18.4 ± 0.0 | 7.19 ± 0.00 |
| Same, following values from copies, no refinement | 1 | 0.3670 ± 0.0000 | 0.2264 ± 0.0000 | 0.5073 ± 0.0000 | -22.52 ± 0.00 | 20.0 ± 0.0 | 0.2130 ± 0.0000 | 0.2559 ± 0.0000 | 0.00 ± 0.00 | 22.1 ± 0.0 | 34.4 ± 0.0 | 6.89 ± 0.00 |
| Same, copies kept, values not followed | 1 | 0.1917 ± 0.0000 | 0.1441 ± 0.0000 | 0.2392 ± 0.0000 | -12.02 ± 0.00 | 6.0 ± 0.0 | 0.2154 ± 0.0000 | 0.2254 ± 0.0000 | 0.75 ± 0.00 | 18.4 ± 0.0 | 28.2 ± 0.0 | 9.25 ± 0.00 |
| Same, copies kept, values not followed, learning steps kept, updates discarded | 1 | 0.3561 ± 0.0000 | 0.2061 ± 0.0000 | 0.5058 ± 0.0000 | -21.96 ± 0.00 | 23.0 ± 0.0 | 0.2290 ± 0.0000 | 0.2290 ± 0.0000 | 0.79 ± 0.00 | 18.7 ± 0.0 | 28.0 ± 0.0 | 9.40 ± 0.00 |
| Same, copies kept, values not followed, refinements kept, workspace not updated | 1 | 0.2527 ± 0.0000 | 0.1734 ± 0.0000 | 0.3319 ± 0.0000 | -15.87 ± 0.00 | 10.0 ± 0.0 | 0.2165 ± 0.0000 | 0.2318 ± 0.0000 | 1.76 ± 0.00 | 20.0 ± 0.0 | 13.9 ± 0.0 | 11.98 ± 0.00 |
| Same, copies kept, values not followed, no computation between requests | 1 | 0.2065 ± 0.0000 | 0.1483 ± 0.0000 | 0.2646 ± 0.0000 | -12.81 ± 0.00 | 7.0 ± 0.0 | 0.2177 ± 0.0000 | 0.2258 ± 0.0000 | 0.44 ± 0.00 | 18.3 ± 0.0 | 17.2 ± 0.0 | 7.26 ± 0.00 |
| Same, copies kept, values not followed, no refinement | 1 | 0.2522 ± 0.0000 | 0.1736 ± 0.0000 | 0.3305 ± 0.0000 | -15.55 ± 0.00 | 10.0 ± 0.0 | 0.2149 ± 0.0000 | 0.2314 ± 0.0000 | 0.00 ± 0.00 | 19.9 ± 0.0 | 32.5 ± 0.0 | 6.75 ± 0.00 |
| New activity head over the fixed model | 2 | 0.1475 ± 0.0011 | 0.1158 ± 0.0001 | 0.1791 ± 0.0023 | -9.33 ± 0.07 | 5.0 ± 0.0 | 0.2107 ± 0.0002 | 0.2286 ± 0.0003 | 0.56 ± 0.09 | 20.6 ± 1.2 | 29.6 ± 1.9 | 8.64 ± 0.19 |
| New activity head over the fixed model, learning steps kept, updates discarded | 2 | 0.3181 ± 0.0001 | 0.1822 ± 0.0003 | 0.4537 ± 0.0004 | -19.65 ± 0.02 | 19.5 ± 0.5 | 0.2206 ± 0.0002 | 0.2206 ± 0.0002 | 0.53 ± 0.14 | 21.7 ± 1.1 | 31.3 ± 1.5 | 8.63 ± 0.40 |
| New activity head over the fixed model, refinements kept, workspace not updated | 2 | 0.2161 ± 0.0094 | 0.1767 ± 0.0060 | 0.2554 ± 0.0127 | -13.48 ± 0.55 | 7.5 ± 0.5 | 0.2125 ± 0.0011 | 0.2588 ± 0.0012 | 0.92 ± 0.06 | 16.6 ± 3.7 | 21.7 ± 1.7 | 9.40 ± 0.35 |
| New activity head over the fixed model, no computation between requests | 2 | 0.1568 ± 0.0028 | 0.1188 ± 0.0004 | 0.1946 ± 0.0052 | -9.81 ± 0.16 | 6.0 ± 0.0 | 0.2118 ± 0.0002 | 0.2279 ± 0.0002 | 0.32 ± 0.07 | 20.8 ± 1.1 | 18.7 ± 1.9 | 6.92 ± 0.13 |
| New activity head over the fixed model, no refinement | 2 | 0.2134 ± 0.0078 | 0.1778 ± 0.0053 | 0.2488 ± 0.0102 | -13.20 ± 0.44 | 7.5 ± 0.5 | 0.2119 ± 0.0009 | 0.2557 ± 0.0006 | 0.00 ± 0.00 | 16.9 ± 3.7 | 38.2 ± 1.5 | 7.17 ± 0.11 |

Paired differences over seeds, own setting minus the other (negative: lower in the own setting):

| Model | other setting | error | reward/100 |
|---|---|---|---|
| Trained through deployment | learning steps kept, updates discarded | -0.1046 ± 0.0087 (n=3) | +6.32 ± 0.52 |
| Trained through deployment | no refinement | -0.0781 ± 0.0643 (n=3) | +4.58 ± 3.86 |
| Trained through deployment | no computation between requests | -0.0104 ± 0.0007 (n=3) | +0.53 ± 0.05 |
| Trained through deployment | refinements kept, workspace not updated | -0.0828 ± 0.0644 (n=3) | +5.06 ± 3.91 |
| Same, one-tick workspace gradients | learning steps kept, updates discarded | -0.1538 ± 0.0126 (n=3) | +9.30 ± 0.76 |
| Same, one-tick workspace gradients | no refinement | -0.0380 ± 0.0080 (n=3) | +2.20 ± 0.50 |
| Same, one-tick workspace gradients | no computation between requests | -0.0121 ± 0.0016 (n=3) | +0.64 ± 0.10 |
| Same, one-tick workspace gradients | refinements kept, workspace not updated | -0.0460 ± 0.0100 (n=3) | +2.87 ± 0.71 |
| Same, no learning | no refinement | -0.0376 ± 0.0104 (n=3) | +1.97 ± 0.67 |
| Same, no learning | no computation between requests | -0.0016 ± 0.0009 (n=3) | -0.02 ± 0.06 |
| Same, no learning | refinements kept, workspace not updated | -0.0298 ± 0.0084 (n=3) | +1.78 ± 0.51 |
| Same, one meta step per window | learning steps kept, updates discarded | -0.0472 ± 0.0131 (n=3) | +2.85 ± 0.79 |
| Same, one meta step per window | no refinement | -0.0131 ± 0.0087 (n=3) | +0.68 ± 0.53 |
| Same, one meta step per window | no computation between requests | -0.0048 ± 0.0012 (n=3) | +0.19 ± 0.08 |
| Same, one meta step per window | refinements kept, workspace not updated | -0.0161 ± 0.0095 (n=3) | +0.95 ± 0.56 |
| Same, cheap targets only | learning steps kept, updates discarded | -0.0882 ± 0.0070 (n=2) | +5.33 ± 0.42 |
| Same, cheap targets only | no refinement | -0.0228 ± 0.0482 (n=2) | +1.26 ± 2.93 |
| Same, cheap targets only | no computation between requests | -0.0135 ± 0.0009 (n=2) | +0.72 ± 0.06 |
| Same, cheap targets only | refinements kept, workspace not updated | -0.0281 ± 0.0512 (n=2) | +1.82 ± 3.23 |
| Same, four adapter blocks | learning steps kept, updates discarded | -0.0915 ± 0.0133 (n=2) | +5.52 ± 0.80 |
| Same, four adapter blocks | no refinement | -0.0491 ± 0.0387 (n=2) | +2.73 ± 2.25 |
| Same, four adapter blocks | no computation between requests | -0.0086 ± 0.0012 (n=2) | +0.42 ± 0.07 |
| Same, four adapter blocks | refinements kept, workspace not updated | -0.0787 ± 0.0598 (n=2) | +4.81 ± 3.70 |
| Same, following values from copies | learning steps kept, updates discarded | -0.1870 ± 0.0000 (n=1) | +11.30 ± 0.00 |
| Same, following values from copies | no refinement | -0.1426 ± 0.0000 (n=1) | +8.49 ± 0.00 |
| Same, following values from copies | no computation between requests | -0.0263 ± 0.0000 (n=1) | +1.49 ± 0.00 |
| Same, following values from copies | refinements kept, workspace not updated | -0.1531 ± 0.0000 (n=1) | +9.25 ± 0.00 |
| Same, copies kept, values not followed | learning steps kept, updates discarded | -0.1644 ± 0.0000 (n=1) | +9.95 ± 0.00 |
| Same, copies kept, values not followed | no refinement | -0.0605 ± 0.0000 (n=1) | +3.53 ± 0.00 |
| Same, copies kept, values not followed | no computation between requests | -0.0148 ± 0.0000 (n=1) | +0.79 ± 0.00 |
| Same, copies kept, values not followed | refinements kept, workspace not updated | -0.0610 ± 0.0000 (n=1) | +3.85 ± 0.00 |
| New activity head over the fixed model | learning steps kept, updates discarded | -0.1706 ± 0.0010 (n=2) | +10.32 ± 0.05 |
| New activity head over the fixed model | no refinement | -0.0659 ± 0.0067 (n=2) | +3.86 ± 0.37 |
| New activity head over the fixed model | no computation between requests | -0.0093 ± 0.0016 (n=2) | +0.48 ± 0.10 |
| New activity head over the fixed model | refinements kept, workspace not updated | -0.0686 ± 0.0083 (n=2) | +4.15 ± 0.48 |
