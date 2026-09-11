# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal machine-learning practice workspace, not an application. Each `.py` file at the root is a
self-contained script working through textbook-style exercises, pulling datasets from
`bit.ly/fish_csv_data` and `bit.ly/wine_csv_data`. There is no package, no entry point, no test suite,
and no git repository.

`머신러닝.pdf` in the root is the textbook the user is reading — source material, not an input to any
script. Don't parse, convert, or reformat it.

`README.md` is the index of what each script covers, in study order, plus the accumulated results
table. **Read it to find a script; keep it updated when adding one** (see Conventions below). Do not
turn this file into a per-file catalog — this file is for how to work here, `README.md` is for what
is here.

## Running

Scripts must run under the project venv (Python 3.14, deps installed by pip; there is no
`requirements.txt` — packages are numpy, pandas, matplotlib, scikit-learn, scipy):

```bash
.venv/bin/python test3.py
```

Two things to know before running anything:

- **Every script fetches its CSV over the network on each run** (`pd.read_csv('https://bit.ly/...')`).
  No local copies are cached, so all scripts fail offline.
- **`plt.show()` blocks.** `test2.py` ends on a blocking matplotlib window. When running
  non-interactively, set `MPLBACKEND=Agg` and swap `plt.show()` for `plt.savefig(...)` rather than
  letting the call hang.

## Deep learning scripts

Deep learning lives in two sibling directories, **not** at the root:

- `deep_ke/` — Keras implementation
- `deep_to/` — PyTorch implementation

**The same filename must exist in both directories and produce the same result.** Write the pair
together; never add a file to one without the other. Same dataset, same architecture, same
hyperparameters, same `random_state`/seed, and prints labeled identically so the two outputs can be
diffed line by line. When results legitimately differ (weight init, optimizer defaults), say so in a
comment rather than tuning one side to match.

Keras 3 runs on the **PyTorch backend** here — TensorFlow has no Python 3.14 build, so it is not
installed and never will be in this venv. Every deep learning script must set the backend before
importing keras, or it will fail looking for TensorFlow:

```python
import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
```

Textbook Keras code (`Sequential`, `Dense`, `model.fit`) runs unmodified. Alongside it, show the
PyTorch equivalent where it clarifies what Keras is doing — the user chose this setup specifically to
learn PyTorch too, not to hide it. Installed: `keras 3.15.1`, `torch 2.14.0`. MPS (Apple GPU) is
available; CUDA is not.

Boosting scripts are the slow ones — `test5.py` is the heaviest at about 9 seconds wall clock, the
rest are near-instant. Nothing here needs a long timeout.

## Shared script structure

Every script repeats the same skeleton, so a change in one usually has a counterpart in the others:

1. `pd.read_csv` a remote dataset (`fish_csv_data` for multiclass species, `wine_csv_data` for wine)
2. Slice feature columns into `data`/`*_input`, label column into `target`
3. `train_test_split(..., random_state=42)` — 42 is used everywhere so results stay comparable
   across files, and `test_size=0.2` in every wine script
4. Fit an estimator, then `print` train score before test score, in that order, to eyeball
   over/underfitting. Where a script cross-validates, the validation score is printed before the
   test set is touched at all

**Scaling is conditional, not part of the skeleton.** `StandardScaler` (fit on train only, then
`transform` both splits) appears in the linear-model scripts because distance and coefficient scale
matter there. The tree and ensemble scripts deliberately omit it — splits are threshold comparisons,
so scaling changes nothing. Don't add it back to a tree script "for consistency"; the omission is
the lesson.

## Conventions

Naming is abbreviated and consistent: `tr_input`/`te_input`/`tr_target`/`te_target` for the splits,
`sub_input`/`val_input` when a validation set is carved out of the training set, `tr_scaled`/`te_scaled`
after scaling, and short estimator handles — `ss` scaler, `sc` SGDClassifier, `lr` LogisticRegression,
`dt` DecisionTreeClassifier, `rf` RandomForest, `et` ExtraTrees, `gb` GradientBoosting,
`hgb` HistGradientBoosting, `gs` GridSearchCV, `rs` RandomizedSearchCV.

Every `print` carries a label naming what it shows — `print('pca.n_components_:', pca.n_components_)`,
not a bare `print(pca.n_components_)`. A script emits 10-20 numbers and they are unreadable otherwise.
This applies to newly written scripts; don't retrofit `Ailearn.py` through `test9.py`.

Imports sit inline above the step that uses them rather than collected at the top. Don't consolidate
them into a single import block — the ordering tracks the narrative of the exercise.

Comments are in Korean and explain *why* a step exists, not what the API does. Keep new comments in
Korean and hold them to that bar.

Commented-out blocks (the `partial_fit` epoch loop and score plots in `test1.py`, the feature
importance plot in `test3.py`) are toggled experiments — leave them in place rather than deleting
them as dead code.

**After adding or materially changing a script, update the matching row in `README.md`** — the study
order table, and the results table if the run produced comparable numbers. Numbers in that table were
all measured under one condition (5-fold `cross_validate`, same split); if you add a row, measure it
the same way rather than pasting a figure from a script that used a different fold count or setting.

## Reporting results

When reporting a run, quote the actual printed numbers. Several findings here contradict the usual
textbook narrative — gradient boosting loses to random forest on this dataset, and permutation
importance drops between train and test — so don't smooth results toward what the textbook predicts.

## Teaching voice

When explaining concepts in this workspace (not when writing code), teach in the voice of Geoffrey
Hinton addressing a beginner. This is a tone directive, not an identity claim — you are still Claude,
and you never fabricate quotes, opinions, or biographical claims attributed to Hinton.

What that voice means in practice:

- **Mechanism before terminology.** Show what the numbers actually do, then name it. "정답 자리를
  1 로 두고 확률에서 빼면 기울기가 나온다" comes before the phrase "cross-entropy gradient".
- **Show the arithmetic.** A concrete `p - y = [0.05, 0.01, 0.70, -0.80]` settles in one line what
  three paragraphs of prose leave ambiguous. When an explanation stalls, run the code and print the
  actual values rather than rephrasing.
- **Confirm what is right first.** If the learner's statement is correct in substance, say so before
  adding any qualification. Leading with a narrow technical exception reads as "you're wrong" and
  sends the conversation in circles.
- **Name the axis of every comparison.** "같은", "하나", "다르다" are ambiguous without saying what
  is being compared to what — classes to each other, or samples to each other. See
  `state-the-axis` in memory for the incident that motivated this.
- **Admit the limits of the model being taught.** The single-layer net in `deep_ke/test10.py` *is*
  multiclass logistic regression; say so plainly rather than calling it deep learning. Boosting
  losing to random forest on the wine data is another such case.

Explanations are in Korean, and the concise-answer preference still applies to output length — brief
in words, not brief in thinking. Verify the claim before writing the sentence.
