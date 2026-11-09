# Build from the repo root:  docker build -t titanic-r -f src/r/Dockerfile .
FROM rocker/r-ver:4.5.1

# Same layout as the repo, so the script finds the data at /app/src/data
WORKDIR /app/src/r

COPY src/r/install_packages.R .
RUN Rscript install_packages.R

# The Kaggle CSVs have to be downloaded into src/data before building (see README)
COPY src/data/train.csv src/data/test.csv src/data/gender_submission.csv /app/src/data/
COPY src/r/titanic_model.R .

CMD ["Rscript", "titanic_model.R"]
