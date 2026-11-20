
### Overall
| System | n | Gold-prov. recall | Cit. precision | Cit. validity | Domain cover | Abstain acc. | Wrongly declined | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| LLM, no retrieval | 100 | 0.561 | 0.486 | 0.809 | 0.879 | 0.000 | 0.000 | 11.914 |
| Single-pass RAG (hybrid) | 100 | 0.393 | 0.347 | 0.857 | 0.781 | 0.000 | 0.000 | 8.340 |
| Full pipeline | 100 | 0.458 | 0.234 | 1.000 | 0.869 | 0.900 | 0.044 | 68.380 |

### Single-domain
| System | n | Gold-prov. recall | Cit. precision | Cit. validity | Domain cover | Abstain acc. | Wrongly declined | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| LLM, no retrieval | 50 | 0.562 | 0.532 | 0.746 | 0.920 | -- | 0.000 | 14.595 |
| Single-pass RAG (hybrid) | 50 | 0.607 | 0.491 | 0.909 | 0.940 | -- | 0.000 | 9.867 |
| Full pipeline | 50 | 0.573 | 0.362 | 1.000 | 0.980 | -- | 0.040 | 56.757 |

### Multi-domain
| System | n | Gold-prov. recall | Cit. precision | Cit. validity | Domain cover | Abstain acc. | Wrongly declined | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| LLM, no retrieval | 40 | 0.560 | 0.458 | 0.860 | 0.827 | -- | 0.000 | 10.736 |
| Single-pass RAG (hybrid) | 40 | 0.240 | 0.227 | 0.880 | 0.583 | -- | 0.000 | 7.524 |
| Full pipeline | 40 | 0.376 | 0.170 | 1.000 | 0.730 | -- | 0.050 | 90.640 |

### Edge cases (should abstain)
| System | n | Gold-prov. recall | Cit. precision | Cit. validity | Domain cover | Abstain acc. | Wrongly declined | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| LLM, no retrieval | 10 | -- | -- | 0.769 | -- | 0.000 | -- | 3.218 |
| Single-pass RAG (hybrid) | 10 | -- | -- | 0.517 | -- | 0.000 | -- | 3.969 |
| Full pipeline | 10 | -- | -- | 1.000 | -- | 0.900 | -- | 37.455 |

### Retrieval only (no LLM): gold provisions in the ranked list
| Retriever | Queries | Subset | Recall@3 | Recall@8 | Recall@20 | Any gold in top 8 | MRR |
|---|---|---|---|---|---|---|---|
| bm25 | 90 | all | 0.220 | 0.299 | 0.421 | 0.578 | 0.395 |
| bm25 | 50 | single | 0.416 | 0.539 | 0.708 | 0.740 | 0.521 |
| bm25 | 40 | multi | 0.080 | 0.128 | 0.216 | 0.375 | 0.237 |
| dense | 90 | all | 0.360 | 0.463 | 0.575 | 0.856 | 0.615 |
| dense | 50 | single | 0.551 | 0.697 | 0.843 | 0.940 | 0.729 |
| dense | 40 | multi | 0.224 | 0.296 | 0.384 | 0.750 | 0.472 |
| hybrid | 90 | all | 0.332 | 0.458 | 0.551 | 0.822 | 0.579 |
| hybrid | 50 | single | 0.551 | 0.753 | 0.843 | 0.940 | 0.720 |
| hybrid | 40 | multi | 0.176 | 0.248 | 0.344 | 0.675 | 0.402 |

### LaTeX rows (overall)
LLM, no retrieval & 0.561 & 0.809 & 0.000 & 0.000 \\
Single-pass RAG (hybrid) & 0.393 & 0.857 & 0.000 & 0.000 \\
Full pipeline & 0.458 & 1.000 & 0.900 & 0.044 \\