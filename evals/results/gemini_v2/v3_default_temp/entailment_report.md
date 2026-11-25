
### Answerable queries (single + multi)
| System | pairs | entailed | partial | unsupported | invalid ID | unsupported among valid [95% CI] | answers with >=1 bad step [95% CI] |
|---|---|---|---|---|---|---|---|
| LLM only | 398 | 79 | 193 | 67 | 59 | 0.198 [0.148, 0.253] | 0.667 [0.567, 0.756] |
| Single-pass RAG | 370 | 304 | 61 | 2 | 3 | 0.005 [0.000, 0.014] | 0.045 [0.011, 0.090] |
| Single-pass RAG + abstain | 279 | 227 | 48 | 4 | 0 | 0.014 [0.003, 0.028] | 0.053 [0.013, 0.105] |
| Full pipeline (final steps) | 22 | 19 | 3 | 0 | 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Single-domain
| System | pairs | entailed | partial | unsupported | invalid ID | unsupported among valid [95% CI] | answers with >=1 bad step [95% CI] |
|---|---|---|---|---|---|---|---|
| LLM only | 197 | 50 | 75 | 38 | 34 | 0.233 [0.149, 0.331] | 0.580 [0.440, 0.720] |
| Single-pass RAG | 202 | 176 | 25 | 0 | 1 | 0.000 [0.000, 0.000] | 0.020 [0.000, 0.060] |
| Single-pass RAG + abstain | 156 | 133 | 20 | 3 | 0 | 0.019 [0.000, 0.042] | 0.068 [0.000, 0.150] |
| Full pipeline (final steps) | 22 | 19 | 3 | 0 | 0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Multi-domain
| System | pairs | entailed | partial | unsupported | invalid ID | unsupported among valid [95% CI] | answers with >=1 bad step [95% CI] |
|---|---|---|---|---|---|---|---|
| LLM only | 201 | 29 | 118 | 29 | 25 | 0.165 [0.119, 0.221] | 0.775 [0.650, 0.900] |
| Single-pass RAG | 168 | 128 | 36 | 2 | 2 | 0.012 [0.000, 0.032] | 0.077 [0.000, 0.175] |
| Single-pass RAG + abstain | 123 | 94 | 28 | 1 | 0 | 0.008 [0.000, 0.028] | 0.032 [0.000, 0.121] |

### Edge cases that should have been refused but were answered
| System | answered of 10 | steps | invalid ID | judged unsupported |
|---|---|---|---|---|
| LLM only | 10 | 27 | 6 | 8 |
| Single-pass RAG | 10 | 16 | 8 | 0 |
| Single-pass RAG + abstain | 0 | 0 | 0 | 0 |

Judge errors (excluded from rates): 0