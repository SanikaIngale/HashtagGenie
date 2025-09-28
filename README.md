## Steps to run:

python -m venv venv

.\venv\Scripts\activate.bat

pip install -r backend/requirements.txt

python backend/train.py --data data/hashtags.csv --model models/model.joblib

python backend/app.py
