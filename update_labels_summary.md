# `update_labels` — Updated Specification

## What it does

For each non-white row in `X_train`, look at the `K` nearest **white** neighbours in feature space and, if the neighbours agree strongly enough and are close enough, rewrite the non-white row's label to match the neighbours' consensus. Unlike `SCM` or `add_comparators`, **no rows are appended** — the training set keeps its original shape, only labels of a subset of non-white rows change.

## Pipeline step by step

1. **Split by race.** Pull out the white rows as reference points, the non-white rows as candidates for relabelling.
2. **Strip protected attributes from the distance metric.** Race and sex are removed from the feature vectors used for nearest-neighbour search, so the match is based on the rest of the profile (age, priors, charge degree, juvenile history).
3. **Fit K-NN on whites.** `NearestNeighbors(n_neighbors=K, metric="euclidean")` is trained on the white subset, then queried with every non-white row → returns `(distances, neighbour_indices)` of shape `(n_nonwhite, K)`.
4. **Derive a candidate label.** For each non-white row, the majority label among its K white neighbours becomes the candidate. The fraction of neighbours backing that majority is recorded.
5. **Agreement gate.** Drop relabelling for any row where the majority fraction is below `AUG_RELABEL_AGREEMENT_THRESHOLD`. This kills ambiguous 6/11 vs 5/11 flips.
6. **Distance gate.** Compute the nearest-neighbour distance for every non-white row, then keep only rows whose nearest-distance falls inside the closest `AUG_RELABEL_DISTANCE_PERCENTILE` percentile. This removes "far-away white twin" matches that would just inject noise.
7. **Apply.** Rows surviving both gates get their label replaced with the candidate. All other rows (white rows + gated-out non-white rows) keep their original labels.
8. **Return.** `(X_aug, y_aug)` with the same shape as the input but a relabelled subset of `y`.

## Configuration knobs (`pipeline/config.py`)

| Knob | Current value | Meaning |
|---|---|---|
| `AUG_RELABEL_K_NEIGHBORS` | `11` | Number of white neighbours consulted per non-white row. `1` = original single-NN behaviour. |
| `AUG_RELABEL_AGREEMENT_THRESHOLD` | `0.8` | Minimum fraction of the K neighbours that must agree on a label before relabelling is allowed. `None` disables the gate. |
| `AUG_RELABEL_DISTANCE_PERCENTILE` | `25` | Only relabel non-white rows whose nearest-white distance is within this percentile of all such distances. `None` = relabel everyone regardless of distance. |

## Typical diagnostic output

```
[update_labels] K=11  agreement_thr=0.8  dist_pct=25
  -> relabelled 440/2752 (skipped 1987 low-agreement, 325 far-distance)
[data] Labels changed: 123/5554 (2.2%)
```

## Why this is better than the original single-NN version

The first implementation relabelled **every** non-white row from its single nearest white neighbour. Two failure modes:

* **Outlier neighbour drives the label.** A single atypical white person could flip a non-white person's label, adding noise in both directions.
* **Matches across a large gap.** Non-white rows with no close white analogue still got a label from a far-away neighbour, injecting label noise that a classifier cannot recover from.

The K-neighbour majority vote fixes the first problem; the distance percentile fixes the second. Rows that fail either gate simply keep their original label instead of being dropped — so the training-set size is unchanged and no information is thrown away, the augmentation just declines to intervene where the signal is weak.

## Interaction with the rest of the pipeline

* Works the same way in the **LDP block** (Step 6c): `X_train` is first perturbed by randomised response on the race bit, then `update_labels` is called on the LDP-ified data. Because the race bit may have been flipped, the set of "non-white" rows used as relabelling candidates is a slightly different sample, but the algorithm is otherwise identical.
* It is one of four scenarios swept by the unified pipeline, alongside `SCM`, `add_comparators`, and `add_comparators_bidir`. Each ends up as its own row (or rows) in `unified_all_steps_per_scenario.csv`.
