# Reproducing "You Don't Bring Me Flowers" (RecSys 2025, Best Full Paper)

Reproducibility study of [*You Don't Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control*](https://arxiv.org/abs/2507.16829) (De Toni et al., RecSys 2025, Best Full Paper Award), carried out for the Information Retrieval course of the MSc in Computer Engineering (AI & Data Science) at Politecnico di Bari.

The paper bounds the fraction of unwanted items in a recommendation list with conformal risk control, using only binary "not interested" feedback. Its `hybrid` strategy replaces risky items with videos the user already watched, so the list keeps its size.

Official code: [geektoni/mitigating-harm-recsys](https://github.com/geektoni/mitigating-harm-recsys)

## What I did

- Re-ran the whole conformal evaluation of the paper (the 16 commands in `experiments/*.sh`, 10 runs each) on the authors' pre-computed scores, on Google Colab without a GPU.
- Compared every regenerated CSV with the published one, metric by metric, and regenerated the paper's figures from both.
- Traced the differences back to the code and reported them to the first author, who invited me to contribute fixes.

## Results

| Method | CSVs compared | Outcome |
|---|---|---|
| `remove` | 6 | Match the published results (max difference < 1e-4) |
| `hybrid` | 10 | Differ: nDCG up to 1.3e-2, H(S,X) up to 1e-4 |
| β = 0.1 and 1.5 | 2 | Not produced by the provided scripts |

**Why `hybrid` differs.** `ranker_kuairand.py` seeds NumPy once and then splits the users across `min(--cores, os.cpu_count())` worker processes. Every worker starts from the same random state and draws the replacement items for its own chunk of users (`src/score_functions.py`), so the draw each user gets depends on the number of workers. The calibration of γ uses the same draws. I checked it directly on run 0 of `hybrid` with SIGformer:

- 8 workers, as in the official scripts: identical to the authors' CSV (26 rows, all 5 metrics);
- 2 workers: nDCG differs by up to 1.6e-3 and Recall by up to 6.5e-4, while H(S,X), |S| and the number of replaced items are identical.

Anyone running the scripts on a machine with fewer than 8 CPUs gets different numbers, with no warning.

## Other reproducibility notes

1. The README installs `requirement.txt`, but the file is `requirements.txt`, and it does not list `torchsparsegradutils`, which `methods/SIGformer/model.py` imports. Since `train_rec.py` imports every model, no training starts without it.
2. The train/test splits (`methods/kuairand/training/*.txt`) are not in the repository (`*.txt` is in `.gitignore`), but the evaluation needs them: they must be generated first with `train_rec.py --build-only-dataset`.
3. The plotting commands in the README do not set `PYTHONPATH`, but `plot_figure_5.py` and `plot_figure_6.py` import from `analytics`.
4. The scripts use `include_groups=True` in `groupby.apply`, which does not work with pandas 3.
5. The batch sizes in the README differ from `methods/run_all.sh` (for example `lightgcl 2014`), and `ncfharm` is only in `run_all.sh`.
6. `experiments/run_ablation_beta.sh` produces 4 values of β, while `results/02_ablation_beta/` contains 6 (also 0.1 and 1.5).
7. `methods/LightGCL/model.py` calls `.cuda(...)` directly, so training LightGCL needs an NVIDIA GPU.
8. The grid of α in the CSVs depends on the split, so comparing runs requires normalising by each run's maximum α, as the plotting script does.

## Contributions to the original repository

- [Pull request #2](https://github.com/geektoni/mitigating-harm-recsys/pull/2): fixes the install instructions and documents how to generate the splits (notes 1, 2 and 5).
- In progress: a per-user random generator, so that `hybrid` no longer depends on `--cores`.

## Notebooks

The notebooks are in Italian and run on Google Colab.

| Notebook | What it does |
|---|---|
| [`01_reevaluation_cpu_parts_A-C.ipynb`](notebooks/01_reevaluation_cpu_parts_A-C.ipynb) | Setup, the paper's figures from the published CSVs, full re-evaluation on the authors' scores and comparison with their CSVs. CPU only; includes the outputs of my run. |
| [`02_full_pipeline_with_retraining.ipynb`](notebooks/02_full_pipeline_with_retraining.ipynb) | The same steps plus retraining of the 6 rankers from scratch (100 epochs × 10 runs), resumable after Colab disconnections. Needs a GPU; this copy has no outputs. |

## Credits and license

The paper, the method and the original code are by Giovanni De Toni, Erasmo Purificato, Emilia Gómez, Andrea Passerini, Bruno Lepri and Cristian Consonni. This repository contains only my reproduction notebooks, released under GPL-3.0 like the original code.

Author: Fabrizio Corsini · [LinkedIn](https://www.linkedin.com/in/corsini-fabrizio)
