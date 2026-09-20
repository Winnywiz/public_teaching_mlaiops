# Lab 1 MLflow run evidence

Experiment: `itcs355-lab1`  
Committed code: `380c87554153f7df2e1f0a8805a99707b39f4329`  
Raw data fingerprint: `422cccb9136e8140`  
DVC data version: `1c886b512c8a5c9bf723da1cd119fc80.dir`  

Each run used seed `20260101`, group-aware machine splits, separate validation/test metrics, and a logged model output. The five configurations vary model capacity or regularisation rather than only changing the seed.

| Run | n_estimators | max_depth | min_samples_leaf | val ROC-AUC | test ROC-AUC | test PR-AUC | MLflow run ID | model output |
|---|---:|---:|---:|---:|---:|---:|---|---|
| study-baseline | 200 | 8 | 5 | 0.83637 | 0.84824 | 0.47762 | `696a5731cbf54c778c6cabf58b039926` | `m-32df855cd6ee49cfb90e22e7d62eb95b` |
| study-shallow | 100 | 4 | 5 | 0.84260 | 0.85329 | 0.49653 | `d6f37d68fe77417bb86b5f4627139459` | `m-4c1721a7d3fb49529e19602ae88b1dc9` |
| study-deep | 300 | 12 | 5 | 0.83537 | 0.84310 | 0.47068 | `aa35367c3cd34868bb30cd9b402aeafc` | `m-cff31070fff24b5aa8e4454081768f0b` |
| study-regularized | 200 | 8 | 10 | 0.84173 | 0.84910 | 0.47646 | `4150e87113c94f02904d9f588e2aa42e` | `m-e38f804081be4a6e89937faff6ee1a59` |
| study-fine-grained | 400 | 8 | 2 | 0.83745 | 0.84727 | 0.48008 | `836ec0ed4be64ca5a7678d68f8b08bf7` | `m-f34df92fb5f5432784bca7f967f09d18` |

The baseline is the configuration used by `make reproduce`; its test ROC-AUC is `0.8482378548603715`, which is within the README claim `0.848 ± 0.010`.
