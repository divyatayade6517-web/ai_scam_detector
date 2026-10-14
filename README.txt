# AI Shield — AI Spam Message Detector

This is an upgraded Streamlit interface for the model files you already trained.

## Expected project structure

Keep these files together in your existing project folder:

```text
ai_spam_chech/
├── app.py
├── requirements.txt
├── SMSSpamCollection
└── model/
    ├── spam_model.pkl
    └── tfidf_vectorizer.pkl
```

The dataset is optional for message scanning. It is needed for the Model Insights evaluation page.

## Setup in VS Code (Windows)

1. Back up your existing `app.py` if you have one.
2. Copy this bundle's `app.py` and `requirements.txt` into your existing project folder.
3. Do NOT delete your existing `model` folder or its two `.pkl` files.
4. Open **Terminal → New Terminal** in VS Code.
5. Install dependencies:

   ```bash
   py -m pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   py -m streamlit run app.py
   ```

7. Streamlit may ask for an email the first time. This is optional; press Enter to skip it if you prefer.
8. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Features

- Dark, modern dashboard
- Message scanner
- Spam probability when supported by the saved model
- Basic suspicious-term indicators (not proof of spam)
- Session-only scan history
- CSV export
- Dataset-based model evaluation: accuracy, spam precision, spam recall, F1 score, confusion matrix and classification report

## Important notes

- The app uses your existing `model/spam_model.pkl` and `model/tfidf_vectorizer.pkl`.
- `SMSSpamCollection` must be in the same folder as `app.py` to enable Model Insights. The app also checks a few common subfolders.
- History is stored in Streamlit session state and is not a permanent database.
- Only load `.pkl` files you created or trust. Pickle-based files can execute code when loaded.
- If your saved model was trained with a different split or preprocessing, the Model Insights metrics may not exactly represent its original test performance. For a formal report, evaluate using the exact saved test split or retrain and evaluate consistently.
- The app is a learning demo; predictions and probabilities are not guarantees.
