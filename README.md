# titanic-disaster

Predicts Titanic survival with a logistic regression, run in a Python container and an R container. Each script loads the Kaggle data, prints every change it makes to the data, fits the model, prints the training accuracy, and predicts on the test set.

The dataset is not included in the repo.

## Repo layout

```
titanic-disaster/
├── README.md
├── .gitignore
├── CODEOWNERS
└── src/
    ├── data/               train.csv, test.csv, gender_submission.csv go here (not committed)
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic_model.R
```

## Requirements

- Docker
- A Kaggle account

## 1. Clone the repo

```
git clone https://github.com/rhparks34/titanic-disaster.git
cd titanic-disaster
```

## 2. Download the data

Download the data from https://www.kaggle.com/competitions/titanic/data ("Download All", after accepting the competition rules). Unzip it and put `train.csv`, `test.csv` and `gender_submission.csv` in `src/data/`.

## 3. Run the Python container

From the repo root:

```
docker build -t titanic-python -f src/python/Dockerfile .
docker run --rm titanic-python
```

The build copies the CSV files from `src/data/` into the image, so it has to run from the repo root after step 2.

## 4. Run the R container

From the repo root:

```
docker build -t titanic-r -f src/r/Dockerfile .
docker run --rm titanic-r
```
