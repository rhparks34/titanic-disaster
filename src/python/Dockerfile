# Build from the repo root:  docker build -t titanic-python -f src/python/Dockerfile .
FROM python:3.12.16-slim

# Same layout as the repo, so the script finds the data at /app/src/data
WORKDIR /app/src/python

COPY src/python/requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check -r requirements.txt

# The Kaggle CSVs have to be downloaded into src/data before building (see README)
COPY src/data/train.csv src/data/test.csv src/data/gender_submission.csv /app/src/data/
COPY src/python/titanic_model.py .

CMD ["python", "titanic_model.py"]
