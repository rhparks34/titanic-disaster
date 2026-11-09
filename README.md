# titanic-disaster

This repo predicts which passengers survived the Titanic using a logistic regression. The same steps run in two Docker containers, one in Python and one in R. Each script loads the Kaggle data, prints every change it makes to the data, fits the model, prints the training accuracy, and then predicts on the test set.

The dataset is not included in this repo. Step 2 below shows how to download it.

## Repo layout

```
titanic-disaster/
├── README.md
├── .gitignore              ignores src/data, .DS_Store and virtual environments
└── src/
    ├── data/               put train.csv, test.csv and gender_submission.csv here (not committed)
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic_model.R
```

## What you need

- Git
- Docker Desktop, installed and running (check with `docker --version`)
- A Kaggle account, to download the data

You don't need Python or R installed. Everything runs inside the containers.

## Step 1: Clone the repo

```
git clone https://github.com/rhparks34/titanic-disaster.git
cd titanic-disaster
```

Run every command below from this `titanic-disaster` folder (the repo root).

## Step 2: Download the data into src/data

1. Go to https://www.kaggle.com/competitions/titanic/data and sign in.
2. If Kaggle asks, click "Join Competition" and accept the rules. The download is blocked until you do.
3. Click "Download All". This gives you `titanic.zip`.
4. Unzip it and move the three CSV files into the `src/data/` folder of this repo.

Check that the files are in the right place:

```
ls src/data
```

You should see `gender_submission.csv  test.csv  train.csv`.

## Step 3: Run the Python container

```
docker build -t titanic-python -f src/python/Dockerfile .
docker run --rm titanic-python
```

The `.` at the end of the build command matters. It builds from the repo root so Docker can copy the CSV files from `src/data` into the image. If the build says `"/src/data/train.csv": not found`, the files aren't in `src/data/` or you aren't in the repo root.

The first build takes a minute or two to install the packages in `requirements.txt`.

## Step 4: Run the R container

```
docker build -t titanic-r -f src/r/Dockerfile .
docker run --rm titanic-r
```

The first build takes a few minutes because `install_packages.R` installs dplyr.

## What the output shows

Both containers print the same sections, labeled with the homework question numbers. Q13 loads `train.csv`. Q14 shows missing values and survival rates, then prints each change to the data. Q15 fits the logistic regression and prints the coefficients, and Q16 prints the training accuracy. Q17 makes the same changes to `test.csv` and predicts survival.

For Q18, `test.csv` has no `Survived` column (Kaggle keeps those answers private), so the scripts estimate test accuracy by refitting the model on 713 passengers from `train.csv` and testing it on the other 178.
