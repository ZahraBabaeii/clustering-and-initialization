# Assignment 5 — Clustering and Initialization

Comparing k-means with random initialization against k-means++ initialization,
both run with the same from-scratch Lloyd's-algorithm loop.

## Research question

Does careful (k-means++) seeding produce lower and/or more consistent
clustering cost (inertia) than uniform random seeding, on a synthetic 2D
dataset and on a real, higher-dimensional, multiclass dataset?

## Requirements vs. our choices

**Explicit assignment requirements** (Statistical Methods for ML, A.Y. 2025/26,
Assignment 5): implement k-means and k-means++ from scratch in NumPy (no
`sklearn` clustering functions), use the same Lloyd loop for both, run
multiple trials, measure inertia and variability, and include both a
synthetic dataset and a real-world, high-dimensional, multiclass dataset,
comparing results *within* each dataset.

**Our experimental choices**, not mandated by the assignment: the synthetic
generator's exact size/cluster count/spread/seed; `load_digits` as the
MNIST-style real dataset; 30 trials per method per dataset; `max_iter=300`,
`tol=1e-4`; the empty-cluster heuristic; and the specific plots produced.
These are documented here so they can be justified in the report.

## Methods

- **`init_random`**: draws `k` distinct row indices uniformly without replacement.
- **`init_kmeans_pp`**: first center uniform; each subsequent center sampled
  with probability proportional to its squared distance to the nearest
  already-chosen center (falls back to uniform sampling if all remaining
  points already coincide with a chosen center).
- **Lloyd's algorithm**: alternates nearest-center assignment and mean-center
  update, shared by both initializations.
- **Stopping rule**: after each iteration we compute the relative decrease in
  inertia, `(inertia_prev − inertia_new) / inertia_prev`, and stop once it is
  `≤ tol` (so `tol=0` correctly recognizes an exactly unchanged solution) or
  once `max_iter` is reached. A genuine inertia increase beyond a scale-aware
  numerical tolerance is detected explicitly and raises an error — it is never
  classified as convergence.
- **Empty-cluster handling**: if a cluster gets no points, its center is
  reassigned to the point currently farthest from its own assigned center
  (the biggest outlier under the current partition). If several clusters are
  empty at once, each gets the next-farthest available point.

## Datasets

- **Synthetic**: 1,000 points, 2D, 4 fixed Gaussian centers
  (`(0,0), (4,0), (0,4), (4,4)`), std=1.3 (some inter-cluster overlap),
  dataset seed=0. True labels are generated but never used for fitting or for
  selecting representative runs.
- **Real**: `sklearn.datasets.load_digits`, all 64 pixel features, `k=10`.
  Pixel intensities divided by 16 (their known max) to rescale to `[0, 1]`;
  **not** z-scored — this is our preprocessing choice, not a verified
  professor requirement. `n=1797`.

## Experiment settings

- 30 single-initialization trials per method per dataset (120 total runs),
  same seed list (`10000`–`10029`) reused across both methods and both
  datasets.
- `max_iter=300`, `tol=1e-4` for every run.
- `QUICK_MODE` toggle in the notebook: `True` runs 5 trials/method/dataset for
  a fast sanity check; `False` runs the full 30. **The results in this
  repository were produced with `QUICK_MODE=False`.**

## Installation

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the notebook from a fresh kernel

1. Open the project folder in VS Code, so the working directory matches the
   notebook's location — the code writes to relative paths
   (`outputs/figures/`, `outputs/tables/`).
2. Open `clustering_project.ipynb`. In the top-right kernel picker, select
   **Select Kernel → Python Environments** and choose the interpreter inside
   `.venv` created above (it's typically listed as `.venv (Python 3.x.x)`).
   If it doesn't appear, run **Python: Select Interpreter** from the Command
   Palette first, point it at `.venv`, then reopen the kernel picker.
3. Restart the kernel, then **Run All**.
4. Section 6 prints `All correctness checks passed.` if the included checks
   (reproducibility, shapes, independent inertia recomputation, non-increasing
   inertia, edge cases) all passed — this is evidence the implementation
   behaves as expected on these checks, not a formal proof of correctness.
5. Leave `QUICK_MODE=False` to reproduce the reported results (takes longer
   than the quick check).

## Output files

A full run produces **seven files**: two CSVs and five figures.

- `outputs/tables/per_run_results.csv` — one row per (dataset, method, seed)
  trial: `dataset, method, seed, inertia, iterations, converged,
  runtime_seconds`. 120 rows for a full run, `4 × N_TRIALS_QUICK` for a quick
  run.
- `outputs/tables/summary_stats.csv` — per dataset×method: mean, std, median,
  min, max, IQR of inertia, convergence rate, mean runtime.
- `outputs/figures/inertia_distribution_{dataset}.png` — boxplot of inertia by
  method, one file per dataset (`synthetic`, `digits` → 2 figures).
- `outputs/figures/convergence_{dataset}.png` — representative convergence
  curves (inertia vs. iteration) per method, one file per dataset (2 figures).
- `outputs/figures/synthetic_clusters.png` — synthetic scatter plot with
  cluster assignments and centers, for the representative run of each method
  (1 figure).

**Quick mode overwrites full-run outputs.** All seven output files above are
written to fixed paths regardless of `QUICK_MODE`. Re-running the notebook
with `QUICK_MODE=True` (e.g. for a fast sanity check) will silently overwrite
the full-run CSVs and figures with the 5-trial versions. Back up (or copy
elsewhere) the `outputs/` folder before running a quick check after
generating results you want to keep.

## Verification of the supplied results

Before writing this README we checked `per_run_results.csv` directly:
120 unique `(dataset, method, seed)` combinations, 30 per group across the 4
groups (`{synthetic, digits} × {random, kmeans++}`), no missing or duplicate
runs, no NaNs, and 100% convergence (all 120 runs converged within
`max_iter=300`). Recomputing mean/std/median/min/max/IQR/convergence
rate/mean runtime directly from the per-run CSV reproduced the supplied
`summary_stats.csv` exactly (all values matched). No inconsistencies found.

## Results (from `summary_stats.csv`, `QUICK_MODE=False`, 30 trials/group)

| Dataset   | Method    | Mean inertia | Std   | Median  | IQR    | Converged | Mean runtime (s) |
|-----------|-----------|-------------:|------:|--------:|-------:|:---------:|------------------:|
| digits    | random    | 4653.66      | 93.28 | 4635.26 | 170.47 | 100%      | 0.137             |
| digits    | kmeans++  | 4617.98      | 91.63 | 4576.28 | 53.18  | 100%      | 0.143             |
| synthetic | random    | 2777.39      | 0.17  | 2777.32 | 0.23   | 100%      | 0.003             |
| synthetic | kmeans++  | 2777.41      | 0.16  | 2777.32 | 0.23   | 100%      | 0.003             |

*Inertia is compared within each dataset only — the two datasets are on
different scales and are not directly comparable.*

## Interpretation (descriptive, no significance testing performed)

- On **digits**, k-means++ has a slightly lower mean inertia and a
  substantially tighter IQR (53 vs. 170) than random init — meaning the
  middle 50% of its inertia values (25th to 75th percentile across the 30
  seeds) are more tightly grouped around the median. However, the standard
  deviations are similar (91.63 vs. 93.28), and k-means++ still produced some
  poor outlier runs (its max, 4927.28, exceeds random's max, 4847.96), so
  this is not an unqualified claim of greater consistency. With only 30
  trials and no hypothesis test, we also cannot claim the mean difference is
  statistically significant.
- On **synthetic**, the two methods reached similar inertia values in these
  trials (median and IQR match to the precision shown). We do not draw
  conclusions about the number or shape of underlying local optima from
  this; the small remaining differences between methods could also partly
  reflect the stopping tolerance (`tol=1e-4`) rather than a genuine
  difference in solution quality.
- We do **not** claim k-means++ always wins; the synthetic results show a
  case where it makes little visible difference.
- Runtimes are comparable between methods; the small runtime differences seen
  here (e.g. kmeans++ slightly slower on digits) may reflect the extra
  distance computations during seeding, but were not measured with repeated
  timing to control for system noise, so should be read qualitatively.

## Limitations

- Only one synthetic configuration (fixed centers/std/seed) was tested — no
  sweep over overlap or cluster count.
- 30 trials/group, no formal statistical test (e.g. a rank-sum test) on the
  inertia distributions.
- Runtimes are single-shot `time.perf_counter()` measurements per trial, not
  repeated/controlled benchmarks.
- Only Euclidean distance / the k-means objective was studied; no comparison
  to other clustering algorithms.
- `load_digits` (64-d) is smaller and lower-dimensional than MNIST; used as
  the assignment's permitted real-world, high-dimensional, multiclass
  substitute.

## References

- Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages of
  careful seeding.* SODA 2007.
- Lloyd, S. (1982). *Least squares quantization in PCM.* IEEE Transactions on
  Information Theory.
- Pedregosa et al. — scikit-learn `load_digits` dataset documentation.
- Course assignment brief: *Experimental Projects*, Statistical Methods for
  Machine Learning, A.Y. 2025/26 (Cesa-Bianchi, Esposito, Foscari).

## Report

The final report is included as [report.pdf](report.pdf). Its LaTeX source is
included as [report.tex](report.tex).
