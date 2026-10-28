# Customer Feedback Sentiment Analysis

A Python project that labels customer reviews as **positive**, **neutral**, or **negative**, cleans the review text, trains a TF-IDF and Logistic Regression classifier, and serves predictions through a FastAPI endpoint.

## What it does

1. Randomly samples 50,000 records from the Amazon `Reviews.csv` dataset using a fixed seed.
2. Assigns sentiment labels with NLTK VADER compound-score thresholds (`>= 0.05` positive, `<= -0.05` negative, otherwise neutral).
3. Cleans and lemmatizes review text with spaCy while retaining negation terms.
4. Converts text to unigram and bigram TF-IDF features (up to 10,000 features).
5. Trains a class-weighted Logistic Regression model and reports classification metrics and a confusion matrix.
6. Provides command-line and HTTP prediction options.

> **Note on labels:** The training labels are generated automatically by VADER. Evaluation measures how well the classifier reproduces those rule-generated labels; it is not an evaluation against independently human-annotated sentiment.

## Repository layout

```text
customer_analysis_feedback_project/
├── main.py                         # Runs the data preparation pipeline
├── requirements.txt
├── src/
│   ├── api/app.py                  # FastAPI prediction endpoint
│   ├── data/                       # Loading and text processing
│   ├── features/                   # Feature utilities
│   └── models/                     # Training, evaluation, CLI prediction
├── data/                           # Local datasets (not committed)
└── models/                         # Locally generated model artifacts
```

## Requirements

- Python 3.10 or later
- The Amazon review CSV at `data/raw/Reviews.csv`, with a `Text` column
- spaCy English model `en_core_web_sm`

Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The first preprocessing run may download NLTK's VADER lexicon if it is not already installed.

## Run the project

Run commands from the repository root.

### Prepare the sample and clean reviews

Place the source CSV at `data/raw/Reviews.csv`, then run:

```bash
python main.py
```

This writes `data/processed/cleaned_reviews.csv`. The script samples 50,000 reviews and expects the input dataset to contain a `Text` column.

### Train the model

```bash
python -m src.models.train
```

The trained classifier and TF-IDF vectorizer are written to `models/`.

### Evaluate

```bash
python -m src.models.evaluate
```

The script prints a classification report and confusion matrix for the held-out split.

### Predict from the command line

```bash
python -m src.models.predict
```

Enter a review at the prompt. Stop with `Ctrl+C`.

### Start the prediction API

After generating the model files:

```bash
uvicorn src.api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` to use the interactive API documentation. Send a `POST` request to `/predict` with JSON such as:

```json
{"text": "The product arrived quickly and works great."}
```

The response contains the predicted sentiment, for example:

```json
{"sentiment": "positive"}
```

## Data and model files

The review dataset and generated CSVs are excluded from Git because they are large and may have separate redistribution or privacy constraints. Obtain the source dataset from its original provider and place it at `data/raw/Reviews.csv` before running the pipeline. The trained `.pkl` files are generated locally and also excluded; training instructions above recreate them.

## Limitations

- VADER-derived labels are weak labels, not verified human judgments.
- Sentiment models can misread sarcasm, domain-specific wording, and mixed opinions.
- The API loads model files at startup, so training must be completed before starting it.
- This is an educational project and should not be treated as a production moderation or decision system without further validation and safeguards.
