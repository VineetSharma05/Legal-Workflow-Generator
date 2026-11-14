# Evaluation report (/home/nishant/Academics/engineering/capstone/Legal-Workflow-Generator/evals/results/gemini_v2)

Gold-provision recall is pooled over the 90 answerable queries; [95% bootstrap CI over queries].

## Systems (Table VII)

| System | n | Recall (all) [95% CI] | Single | Multi | Stat.-level (all) | Precision | Edge refused | Wrongly declined | Latency mean / median (s) |
|---|---|---|---|---|---|---|---|---|---|
| Pipeline (suggestions off) | 100 | 0.565 [0.493, 0.638] | 0.663 | 0.496 | 0.818 | 0.268 | 9/10 | 3/90 (3 single, 0 multi) | 62.7 / 60.6 |
| Pipeline (suggestions on) | 100 | 0.617 [0.547, 0.687] | 0.708 | 0.552 | 0.869 | 0.256 | 8/10 | 2/90 (1 single, 1 multi) | 79.9 / 64.2 |
| Pipeline (suggestions on, repeat) | 100 | 0.617 [0.547, 0.687] | 0.708 | 0.552 | 0.869 | 0.256 | 8/10 | 2/90 (1 single, 1 multi) | 70.1 / 59.1 |
| LLM only (ids given) | 100 | 0.561 [0.498, 0.624] | 0.562 | 0.560 | 0.780 | 0.486 | 0/10 | 0/90 (0 single, 0 multi) | 11.9 / 2.9 |
| LLM only (names only) | 100 | 0.603 [0.546, 0.660] | 0.596 | 0.608 | 0.804 | 0.390 | 0/10 | 0/90 (0 single, 0 multi) | 8.6 / 4.8 |
| RAG, BM25 | 100 | 0.266 [0.208, 0.336] | 0.472 | 0.120 | 0.514 | 0.254 | 0/10 | 0/90 (0 single, 0 multi) | 8.4 / 2.9 |
| RAG, hybrid (no abstain) | 100 | 0.355 [0.288, 0.431] | 0.573 | 0.200 | 0.556 | 0.298 | 0/10 | 0/90 (0 single, 0 multi) | 5.6 / 1.9 |
| RAG, hybrid (abstain) | 100 | 0.350 [0.284, 0.424] | 0.584 | 0.184 | 0.509 | 0.424 | 10/10 | 11/90 (2 single, 9 multi) | 7.0 / 3.2 |

## Recall by number of gold domains (Table VIII)

| System | 1 dom. (n=50) | 2 dom. (n=20) | 3 dom. (n=12) | 4 dom. (n=5) | 5 dom. (n=3) |
|---|---|---|---|---|---|
| Pipeline (suggestions off) | 0.663 | 0.490 | 0.500 | 0.550 | 0.438 |
| Pipeline (suggestions on) | 0.708 | 0.510 | 0.553 | 0.700 | 0.500 |
| Pipeline (suggestions on, repeat) | 0.708 | 0.510 | 0.553 | 0.700 | 0.500 |
| LLM only (ids given) | 0.562 | 0.588 | 0.500 | 0.550 | 0.625 |
| LLM only (names only) | 0.596 | 0.588 | 0.658 | 0.600 | 0.562 |
| RAG, BM25 | 0.472 | 0.196 | 0.053 | 0.100 | 0.062 |
| RAG, hybrid (no abstain) | 0.573 | 0.275 | 0.237 | 0.050 | 0.062 |
| RAG, hybrid (abstain) | 0.584 | 0.255 | 0.184 | 0.100 | 0.062 |

## Statute-level recall

| System | All | Single | Multi |
|---|---|---|---|
| Pipeline (suggestions off) | 0.818 | 0.876 | 0.776 |
| Pipeline (suggestions on) | 0.869 | 0.944 | 0.816 |
| Pipeline (suggestions on, repeat) | 0.869 | 0.944 | 0.816 |
| LLM only (ids given) | 0.780 | 0.787 | 0.776 |
| LLM only (names only) | 0.804 | 0.775 | 0.824 |
| RAG, BM25 | 0.514 | 0.742 | 0.352 |
| RAG, hybrid (no abstain) | 0.556 | 0.809 | 0.376 |
| RAG, hybrid (abstain) | 0.509 | 0.775 | 0.320 |

## Paired differences in gold-provision recall (pipeline minus other system)

| Pipeline run | Against | All [95% CI] | Single [95% CI] | Multi [95% CI] |
|---|---|---|---|---|
| Pipeline (suggestions off) | Pipeline (suggestions on) | -0.051 [-0.100, +0.000] | -0.045 [-0.102, +0.012] | -0.056 [-0.128, +0.017] |
| Pipeline (suggestions off) | Pipeline (suggestions on, repeat) | -0.051 [-0.100, +0.000] | -0.045 [-0.102, +0.012] | -0.056 [-0.128, +0.017] |
| Pipeline (suggestions off) | LLM only (ids given) | +0.005 [-0.084, +0.094] | +0.101 [-0.032, +0.227] | -0.064 [-0.176, +0.051] |
| Pipeline (suggestions off) | LLM only (names only) | -0.037 [-0.110, +0.038] | +0.067 [-0.044, +0.178] | -0.112 [-0.200, -0.023] |
| Pipeline (suggestions off) | RAG, BM25 | +0.299 [0.216, 0.376] | +0.191 [0.067, 0.323] | +0.376 [0.279, 0.477] |
| Pipeline (suggestions off) | RAG, hybrid (no abstain) | +0.210 [0.129, 0.290] | +0.090 [-0.034, +0.213] | +0.296 [0.197, 0.398] |
| Pipeline (suggestions off) | RAG, hybrid (abstain) | +0.215 [0.133, 0.293] | +0.079 [-0.034, +0.191] | +0.312 [0.214, 0.411] |
| Pipeline (suggestions on) | Pipeline (suggestions off) | +0.051 [0.000, 0.100] | +0.045 [-0.011, +0.103] | +0.056 [-0.017, +0.128] |
| Pipeline (suggestions on) | Pipeline (suggestions on, repeat) | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] |
| Pipeline (suggestions on) | LLM only (ids given) | +0.056 [-0.033, +0.145] | +0.146 [0.011, 0.281] | -0.008 [-0.120, +0.101] |
| Pipeline (suggestions on) | LLM only (names only) | +0.014 [-0.067, +0.095] | +0.112 [0.011, 0.216] | -0.056 [-0.158, +0.049] |
| Pipeline (suggestions on) | RAG, BM25 | +0.350 [0.269, 0.422] | +0.236 [0.119, 0.359] | +0.432 [0.339, 0.525] |
| Pipeline (suggestions on) | RAG, hybrid (no abstain) | +0.262 [0.179, 0.342] | +0.135 [0.012, 0.250] | +0.352 [0.252, 0.455] |
| Pipeline (suggestions on) | RAG, hybrid (abstain) | +0.266 [0.185, 0.341] | +0.124 [0.012, 0.232] | +0.368 [0.275, 0.462] |
| Pipeline (suggestions on, repeat) | Pipeline (suggestions off) | +0.051 [0.000, 0.100] | +0.045 [-0.011, +0.103] | +0.056 [-0.017, +0.128] |
| Pipeline (suggestions on, repeat) | Pipeline (suggestions on) | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] | +0.000 [0.000, 0.000] |
| Pipeline (suggestions on, repeat) | LLM only (ids given) | +0.056 [-0.033, +0.145] | +0.146 [0.011, 0.281] | -0.008 [-0.120, +0.101] |
| Pipeline (suggestions on, repeat) | LLM only (names only) | +0.014 [-0.067, +0.095] | +0.112 [0.011, 0.216] | -0.056 [-0.158, +0.049] |
| Pipeline (suggestions on, repeat) | RAG, BM25 | +0.350 [0.269, 0.422] | +0.236 [0.119, 0.359] | +0.432 [0.339, 0.525] |
| Pipeline (suggestions on, repeat) | RAG, hybrid (no abstain) | +0.262 [0.179, 0.342] | +0.135 [0.012, 0.250] | +0.352 [0.252, 0.455] |
| Pipeline (suggestions on, repeat) | RAG, hybrid (abstain) | +0.266 [0.185, 0.341] | +0.124 [0.012, 0.232] | +0.368 [0.275, 0.462] |

## Pipeline (suggestions off): routing and coverage (Tables IV-VI)

| Metric | Overall | Single | Multi | Edge |
|---|---|---|---|---|
| Answered % | 88.0 | 94.0 | 100.0 | 10.0 |
| Domain recall | 0.955 | 1.0 | 0.898 | None |
| Domain precision | 0.925 | 0.882 | 0.978 | None |
| Domain exact match | 0.733 | 0.78 | 0.675 | None |
| Primary domain accuracy | 0.956 | 0.92 | 1.0 | None |
| Citations cover every asked domain | 0.889 | 0.94 | 0.825 | None |
| Mean latency (s) | 62.734 | 56.366 | 78.567 | 31.239 |

| Domains | n | Domain recall | Gold-prov. recall | Domain cover | Latency (s) |
|---|---|---|---|---|---|
| 2 | 20 | 0.875 | 0.49 | 0.85 | 66.236 |
| 3 | 12 | 0.889 | 0.5 | 0.75 | 75.997 |
| 4 | 5 | 0.95 | 0.55 | 0.8 | 114.894 |
| 5 | 3 | 1.0 | 0.438 | 1.0 | 110.51 |

| Domain | n | Detect. recall | Cited recall | FP |
|---|---|---|---|---|
| data_protection | 34 | 1.0 | 1.0 | 0 |
| corporate_governance | 33 | 0.818 | 0.727 | 3 |
| ip_licensing | 28 | 0.964 | 0.929 | 3 |
| taxation | 33 | 0.97 | 0.818 | 6 |
| employment | 33 | 0.939 | 0.848 | 5 |

LLM calls: 1159 (11.6 per query); tokens: 1635903 (16359 per query).

## Pipeline (suggestions on): routing and coverage (Tables IV-VI)

| Metric | Overall | Single | Multi | Edge |
|---|---|---|---|---|
| Answered % | 90.0 | 98.0 | 97.5 | 20.0 |
| Domain recall | 0.961 | 1.0 | 0.913 | None |
| Domain precision | 0.919 | 0.872 | 0.978 | None |
| Domain exact match | 0.744 | 0.76 | 0.725 | None |
| Primary domain accuracy | 0.956 | 0.92 | 1.0 | None |
| Citations cover every asked domain | 0.919 | 0.98 | 0.844 | None |
| Mean latency (s) | 79.864 | 74.703 | 98.074 | 32.83 |

| Domains | n | Domain recall | Gold-prov. recall | Domain cover | Latency (s) |
|---|---|---|---|---|---|
| 2 | 20 | 0.875 | 0.51 | 0.825 | 70.884 |
| 3 | 12 | 0.917 | 0.553 | 0.834 | 90.159 |
| 4 | 5 | 1.0 | 0.7 | 0.85 | 136.12 |
| 5 | 3 | 1.0 | 0.5 | 1.0 | 247.59 |

| Domain | n | Detect. recall | Cited recall | FP |
|---|---|---|---|---|
| data_protection | 34 | 1.0 | 1.0 | 0 |
| corporate_governance | 33 | 0.818 | 0.758 | 3 |
| ip_licensing | 28 | 1.0 | 1.0 | 3 |
| taxation | 33 | 0.97 | 0.848 | 7 |
| employment | 33 | 0.97 | 0.879 | 5 |

LLM calls: 1151 (11.5 per query); tokens: 1980303 (19803 per query).

## Pipeline (suggestions on, repeat): routing and coverage (Tables IV-VI)

| Metric | Overall | Single | Multi | Edge |
|---|---|---|---|---|
| Answered % | 90.0 | 98.0 | 97.5 | 20.0 |
| Domain recall | 0.961 | 1.0 | 0.913 | None |
| Domain precision | 0.919 | 0.872 | 0.978 | None |
| Domain exact match | 0.744 | 0.76 | 0.725 | None |
| Primary domain accuracy | 0.956 | 0.92 | 1.0 | None |
| Citations cover every asked domain | 0.919 | 0.98 | 0.844 | None |
| Mean latency (s) | 70.071 | 59.361 | 91.088 | 39.554 |

| Domains | n | Domain recall | Gold-prov. recall | Domain cover | Latency (s) |
|---|---|---|---|---|---|
| 2 | 20 | 0.875 | 0.51 | 0.825 | 64.862 |
| 3 | 12 | 0.917 | 0.553 | 0.834 | 115.562 |
| 4 | 5 | 1.0 | 0.7 | 0.85 | 116.462 |
| 5 | 3 | 1.0 | 0.5 | 1.0 | 125.74 |

| Domain | n | Detect. recall | Cited recall | FP |
|---|---|---|---|---|
| data_protection | 34 | 1.0 | 1.0 | 0 |
| corporate_governance | 33 | 0.818 | 0.758 | 3 |
| ip_licensing | 28 | 1.0 | 1.0 | 3 |
| taxation | 33 | 0.97 | 0.848 | 7 |
| employment | 33 | 0.97 | 0.879 | 5 |

LLM calls: 1151 (11.5 per query); tokens: 1979910 (19799 per query).

## Stability (two runs, same settings)

| Metric | Run 1 | Run 2 | Mean | |diff| |
|---|---|---|---|---|
| recall | 0.617 | 0.617 | 0.617 | 0.000 |
| recall_single | 0.708 | 0.708 | 0.708 | 0.000 |
| recall_multi | 0.552 | 0.552 | 0.552 | 0.000 |
| statute_recall | 0.869 | 0.869 | 0.869 | 0.000 |
| precision | 0.256 | 0.256 | 0.256 | 0.000 |
| edge_refused | 8 | 8 | 8.0 | 0 |
| wrongly_declined | 2 | 2 | 2.0 | 0 |
