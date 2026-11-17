# Data

| File | Rows | Columns | Description |
|---|---|---|---|
| `train.csv` | 7,352 | 563 | Training windows from 21 subjects |
| `test.csv` | 2,947 | 563 | Test windows from 9 held-out subjects (2, 4, 9, 10, 12, 13, 18, 20, 24) |

Each row is one 2.56 s window (128 samples at 50 Hz, 50 % overlap) recorded by a waist-mounted smartphone.
Columns: 561 engineered accelerometer/gyroscope features scaled to [-1, 1], `subject` (volunteer ID) and
`Activity` (target: `LAYING`, `SITTING`, `STANDING`, `WALKING`, `WALKING_DOWNSTAIRS`, `WALKING_UPSTAIRS`).
There are no missing values and no duplicate rows. Source: UCI Machine Learning Repository,
*Human Activity Recognition Using Smartphones* (Anguita et al., 2013).

## Optional external datasets (`data/external/`)

Module 2 of the notebook (§6–§11, the Markov/HMM part) needs the original raw archives. They are too large for the repository
(about 328 MB and 241 MB unpacked). From the repository root run `python scripts/download_data.py`, which downloads
and unpacks them here; if the download is blocked, fetch the zips from the UCI pages below and run
`python scripts/download_data.py --from-zip uci <zip>` (or `hapt`). The resulting layout is:

```
data/external/
├── human+activity+recognition+using+smartphones/
│   └── UCI HAR Dataset/            # raw Inertial Signals: §6 temporal audit, needed by §6–§10
└── smartphone+based+recognition+of+human+activities+and+postural+transitions/
    ├── RawData/labels.txt          # HAPT validation: §11
    ├── Train/
    └── Test/
```

| Folder | UCI dataset |
|---|---|
| `human+activity+recognition+using+smartphones/` | Human Activity Recognition Using Smartphones (dataset 240) |
| `smartphone+based+recognition+of+human+activities+and+postural+transitions/` | Smartphone-Based Recognition of Human Activities and Postural Transitions (HAPT, dataset 341) |

`data/external/` is git-ignored. Sections §1–§5, Review 2 (§R2) and the Unit 5 extension run with `train.csv` and
`test.csv` alone. All saved outputs are already in the notebook, so it can be read without downloading anything.

## Licence and citation

Both UCI datasets are licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**
(https://creativecommons.org/licenses/by/4.0/). `train.csv` and `test.csv` are the UCI HAR feature tables in
CSV form, unchanged in content. If you use them, cite:

- D. Anguita, A. Ghio, L. Oneto, X. Parra and J. L. Reyes-Ortiz, "A public domain dataset for human activity
  recognition using smartphones," *21st European Symposium on Artificial Neural Networks, Computational
  Intelligence and Machine Learning (ESANN)*, Bruges, Belgium, 2013.
  Dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- J. L. Reyes-Ortiz, L. Oneto, A. Samà, X. Parra and D. Anguita, "Transition-aware human activity recognition
  using smartphones," *Neurocomputing*, vol. 171, pp. 754–767, 2016. Dataset: https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions
