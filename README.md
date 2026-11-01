# NLP Text Preprocessing

This repository contains a simple Natural Language Processing (NLP) text preprocessing pipeline for sentiment-based text data. The project uses a Twitter sentiment dataset and converts raw review text into cleaned, tokenized sequences suitable for downstream machine learning or deep learning models.

## Project Overview

The main notebook, `NLP_text_preproessing.ipynb`, demonstrates the complete preprocessing workflow:

- Loading the dataset
- Inspecting the data
- Removing duplicates and null values
- Dropping unnecessary columns
- Renaming columns for readability
- Lowercasing text
- Removing extra spaces and numbers
- Cleaning punctuation
- Removing stopwords
- Mapping sentiment labels to numeric values
- Tokenizing the cleaned text

## Repository Contents

- `NLP_text_preproessing.ipynb` — Jupyter Notebook containing the full preprocessing workflow
- `twitter_training.csv` — dataset used in the notebook

## Dataset

The project uses a Twitter-style sentiment dataset stored in `twitter_training.csv`.

The data contains text reviews and sentiment labels such as:

- Positive
- Neutral
- Negative
- Irrelevant

The notebook also maps these labels to numeric values:

- Positive = 1
- Neutral = 0
- Negative = -1
- Irrelevant = 2

## Key Preprocessing Steps

The notebook applies the following text-cleaning rules:

1. Convert text to lowercase
2. Remove extra whitespace
3. Remove numeric values
4. Remove punctuation
5. Remove English stopwords
6. Prepare text for tokenization

A typical preprocessing function used in the project looks like this:

```python
def text_preprocessing(text):
    text = text.lower()
    text = re.sub("\s+", " ", text)
    text = re.sub("\d+", " ", text)
    punctuations = "".join([a for a in text if a not in string.punctuation])
    text = " ".join([a for a in punctuations.split() if a not in stopwords.words("english")])
    return text
```

## Libraries Used

The project uses the following Python libraries:

```python
import pandas as pd
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
```

## Installation

Before running the notebook, install the required dependencies:

```bash
pip install pandas tensorflow nltk
```

Then download the NLTK stopwords corpus:

```python
import nltk
nltk.download('stopwords')
```

## Execution

Open the notebook in Jupyter or Google Colab and run all cells sequentially.

The expected workflow is:

```bash
jupyter notebook NLP_text_preproessing.ipynb
```

## Output

After preprocessing, the `reviews` column is transformed into tokenized sequences using Keras `Tokenizer`, producing numeric representation ready for model training.

Example output:

```python
0    [282, 6306, 311]
1    [4, 84, 59, 311]
```

## Notes

- The notebook is designed as a beginner-friendly NLP preprocessing example.
- It focuses on cleaning and transforming text before training a sentiment analysis model.
- The workflow is a strong foundation for tasks such as text classification, sentiment analysis, and chatbot-related text processing.

## Future Improvements

Possible upgrades for this project include:

- Adding lemmatization instead of simple stemming-like cleaning
- Handling emojis and slang text more effectively
- Comparing different preprocessing pipelines
- Building a sentiment classifier on top of the cleaned data
- Saving the processed dataset for model training

## Summary

This repository demonstrates the basic pipeline required for NLP text preprocessing, especially for sentiment analysis. It shows how raw review text can be standardized and converted into machine-readable token sequences for model training.

