# Self-training with LightGBM vs. a supervised MLP for HAR with 5 labels per activity

Human activity recognition (HAR) models from wearable sensors often need many labeled recordings, and labels are costly to collect for every new user, device, or setting. The paper "Cross-domain learning in deep HAR models via natural language processing on action labels" (2022) tackles this by transferring knowledge across domains using the semantics of action labels. This experiment does not reproduce that method. It asks a simpler, related question: when only a handful of labels exist in the target domain, can unlabeled target windows help?

**Question:** Does confidence-based self-training with LightGBM, using unlabeled target-domain windows, beat a supervised MLP trained on the same few labels when only 5 labeled examples per activity are available?

**Dataset:** OpenML "har" (id 1478), smartphone accelerometer/gyroscope features with 561 columns and 6 activities. It is not the paper's own dataset. The data are split 80/20 (stratified, fixed seed) into a pool and a held-out test set. The pool is then reduced to 5 labeled rows per activity, and the remaining pool rows are treated as unlabeled.

**Baseline:** a supervised scikit-learn MLPClassifier trained only on the 5-shot labeled rows. It stands in for the paper's supervised deep HAR model.

**Improved model:** a LightGBM classifier self-trained on the 5-shot labels plus pseudo-labels for unlabeled pool rows whose predicted probability exceeds 0.9, iterated for three rounds.

Both models use the same standardized features and labeled data. They are evaluated on the untouched test set with macro-F1 (main metric) and accuracy, averaged over seeds 42, 43, and 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> K Bacharidis, A Argyros (2022). *Cross-domain learning in deep har models via natural language processing on action labels*. International Symposium on Visual Computing. https://link.springer.com/chapter/10.1007/978-3-031-20713-6_27

Paper author: Prof. Antonis Argyros, University of Crete (argyros@csd.uoc.gr)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
