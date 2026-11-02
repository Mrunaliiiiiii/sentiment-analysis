# 💬 Twitter Sentiment Analysis Using Machine Learning

Classifies tweets as **positive** or **negative** using TF-IDF features and classic machine-learning models (Naive Bayes, Logistic Regression, Linear SVM), with a **Streamlit** web app for live predictions.

> **Author:** `<Your Full Name>` · **Roll No:** `<Your Roll No>` · **Course:** BCA · **Project:** `<Project Title>`

<!-- Add a screenshot of the Streamlit app here, e.g. ![App screenshot](images/app.png) -->

## Objective

Build an end-to-end data science project: explore a real dataset, clean noisy text, train and compare several models, evaluate them properly, and turn the best one into a small web app.

## Dataset

[**Sentiment140**](https://www.kaggle.com/datasets/kazanova/sentiment140): 1.6 million tweets labelled `0` (negative) and `4` (positive). The labels were produced automatically from emoticons, so they are noisy.

The dataset is **not** stored in this repository (238 MB). To reproduce the results:

1. Download it from the Kaggle link above.
2. Put `training.1600000.processed.noemoticon.csv` inside the `data/` folder.

`data/sample.csv` is a **synthetic mock file** (1,000 made-up tweets in the same format) that exists only so the code can be tested without the download. Results on it are meaningless.

## Project structure

```
sentiment-analysis/
├── notebooks/sentiment_analysis.ipynb   # full workflow: EDA → cleaning → models → evaluation
├── sentiment_utils.py                   # text cleaning + prediction helpers (shared by notebook and app)
├── app.py                               # Streamlit demo
├── models/                              # saved vectorizer + model (.joblib) + metadata.json
├── data/
│   ├── sample.csv                       # 1,000-row synthetic mock sample
│   └── make_mock_sample.py              # script that generates the mock sample
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick start

```bash
# 1. create an environment (Anaconda)
conda create -n sentiment python=3.11 -y
conda activate sentiment

# 2. install dependencies
pip install -r requirements.txt

# 3. train: open the notebook and run all cells (this also saves the model to models/)
jupyter notebook notebooks/sentiment_analysis.ipynb

# 4. launch the web app
streamlit run app.py
```

If the full dataset is missing, the notebook falls back to the mock sample and prints a warning. The notebook samples 50,000 tweets per class by default; lower `SAMPLES_PER_CLASS` in the first cell if it runs slowly.

## Method

| Step | What is done |
|---|---|
| Loading | Read the headerless `latin-1` CSV, keep label + text, map `4 → 1` |
| Cleaning the data | Drop missing tweets and exact duplicates, then take a balanced sample (same number of tweets per class) |
| Text preprocessing | Decode HTML entities, lowercase, remove URLs/@mentions, expand `n't` to `not`, keep letters only, shorten stretched letters (`sooo`), remove stop words **but keep negations** (`not`, `no`, `never`) |
| Features | TF-IDF with unigrams + bigrams (up to 50,000 features), fitted on the training set only |
| Split | 80 / 20, stratified |
| Models | Multinomial Naive Bayes, Logistic Regression, Linear SVM |
| Evaluation | Accuracy, precision, recall, F1, confusion matrices, ROC-AUC computed from model scores, 3-fold cross-validation |
| Analysis | Top words per class, error analysis |

## Results

> Fill this table in with the numbers from **your own run** on the full dataset. Do not use numbers from the mock sample.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | | | | | |
| Logistic Regression | | | | | |
| Linear SVM | | | | | |

**Key learnings:** `<write 3 to 4 points in your own words: which model won and why, what the errors looked like, what surprised you>`

## Limitations

- Labels come from emoticons, not from human annotators, so some are wrong.
- Only two classes: there is no *neutral* class (uncertain predictions are flagged in the app instead).
- Sarcasm, slang and very short tweets are hard for bag-of-words models.
- The training tweets are from 2009; language has changed since then.
- The model learned from tweets, so it may be less accurate on other text such as product reviews.

## Ideas for improvement

- Compare with and without stemming or lemmatization.
- Add a neutral class, or test on Amazon/Flipkart product reviews to measure how much accuracy drops on a different kind of text.
- Try a pretrained transformer such as DistilBERT.
- Tune hyperparameters with `GridSearchCV` on a validation set.

## Security note

No API keys or passwords are used or stored in this project. If you ever add any, keep them in a `.env` file (already listed in `.gitignore`) and never commit them.
