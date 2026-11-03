# Machine learning competition solutions

My solutions to 15 public data science competitions and practice problems, written 2016–2019 in R and Python. Most are from [Analytics Vidhya DataHack](https://datahack.analyticsvidhya.com/); one is from the SentiRuEval-2016 shared task on Russian sentiment analysis.

They cover tabular regression and classification, NLP, image and audio classification, and time-series forecasting. Most solutions build the pipeline by hand, using custom functions for imputation, encoding, resampling for class imbalance, evaluation metrics and voting ensembles, and compare many classical models.

Contact: Aleksey Shipitsyn, [LinkedIn](https://www.linkedin.com/in/aleksey-shipitsyn-53417431)

## Overview

| # | Competition | Year | Task | Language | Leaderboard |
|---|---|---|---|---|---|
| 1 | [Loan Prediction III](#1-loan-prediction-iii) | 2016 | binary classification | R | — |
| 2 | [Loan Prediction III, ensemble submission](#2-loan-prediction-iii-ensemble-submission) | 2016 | binary classification | R | — |
| 3 | [BigMart Sales](#3-bigmart-sales) | 2016 | regression | R | — |
| 4 | [Black Friday Sales](#4-black-friday-sales) | 2016 | regression | R | — |
| 5 | [Footfall in parks (Student Hunt)](#5-footfall-in-parks) | 2016 | regression | R | — |
| 6 | [Footfall in parks, lagged and interaction features](#6-footfall-in-parks-lagged-and-interaction-features) | 2016 | regression | R | — |
| 7 | [HR Analytics (WNS Hackathon 2018)](#7-hr-analytics) | 2019 | imbalanced binary classification | Python | 110 / 6 834 (top 1.6 %) |
| 8 | [Recommender Engine](#8-recommender-engine) | 2019 | multiclass classification | R | — |
| 9 | [Predict Number of Upvotes (Enigma Codefest)](#9-predict-number-of-upvotes) | 2019 | regression | Python | 396 / 2 320 (top 17.1 %) |
| 10 | [SentiRuEval-2016](#10-sentirueval-2016) | 2016 | multiclass text classification (Russian) | R | — |
| 11 | [Identify the Sentiments (Linguipedia Codefest)](#11-identify-the-sentiments) | 2019 | binary text classification | Python | 225 / 2 424 (top 9.3 %) |
| 12 | [Twitter Sentiment Analysis](#12-twitter-sentiment-analysis) | 2019 | imbalanced binary text classification | Python | 408 / 10 321 (top 4.0 %) |
| 13 | [Age Detection](#13-age-detection) | 2017 | image classification | R | — |
| 14 | [Digit Recognition](#14-digit-recognition) | 2017 | image classification | R | — |
| 15 | [Identify the Apparels](#15-identify-the-apparels) | 2019 | image classification | Python | 203 / 2 310 (top 8.8 %) |
| 16 | [Urban Sound Classification](#16-urban-sound-classification) | 2017 | audio classification | R | — |
| 17 | [Time Series Forecasting](#17-time-series-forecasting) | 2017 | count forecasting | R | — |

Leaderboard positions are given where I recorded them; "—" means not recorded. Years are taken from the script headers.

The leaderboard positions and the results noted in the code come from the original submissions. The code has since been reviewed and corrected (bugs in imputation, voting, stacking, lag features and evaluation), so rerunning it gives somewhat different numbers.

## Running the code

Each solution is in its own folder. The competition data is not included; download it from the competition page after registering and put the files, with their original names, in a `data/` folder inside the solution's folder. Run R scripts and notebooks from the solution's folder; R scripts switch into `data/` and write their submissions there.

Install the packages with:

```
pip install -r requirements.txt
Rscript install.R
```

The notebooks are saved with the outputs of their original runs; cells changed by the corrections have no saved output.

## Tabular data

### 1. Loan Prediction III
[Competition page](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/) · [`loan-prediction/loan_prediction.R`](loan-prediction/loan_prediction.R)

Predict whether a loan application is approved from applicant demographics, income and credit history.
- Stochastic regression imputation for continuous variables (loan amount, term); sampling or median imputation for categorical ones.
- Domain features: total and net income, annuity, annuity coverage, income-to-loan ratios, values relative to sample means and modes; log transforms and rescaling.
- PCA, keeping the first 13 components; 2-D view of the classes by metric multidimensional scaling.
- Custom `undersample`, `oversample` and `equalsample` functions; every model is rerun on each resampled training set.
- Misclassification rate overall and per class on an 80/20 split.
- Models: logistic regression, multinomial logistic regression (`nnet::multinom`), classification tree, random forest, SVM, kNN, naive Bayes, BART, Gaussian process, and a majority vote of all nine.

### 2. Loan Prediction III, ensemble submission
[`loan-prediction/loan_prediction_ensemble.R`](loan-prediction/loan_prediction_ensemble.R)

The submission version of solution 1, restructured into functions.
- The same preprocessing, features and 13-component PCA.
- The whole pipeline (preprocessing, PCA, models) repeated 10 times, each time on a random 80 % of the training data.
- Final prediction: a majority vote over all models from all 10 runs.
- Models: logistic regression, classification tree, random forest, SVM, naive Bayes, BART, Gaussian process.

### 3. BigMart Sales
[`bigmart-sales/bigmart_sales.R`](bigmart-sales/bigmart_sales.R)

Predict item sales per store from product and store attributes.
- Cleaning inconsistent category labels (e.g. variants of "low fat").
- Missing item weight imputed from the same item in other stores.
- Ordinal encoding of store size; one-hot encoding with a custom function; centring and scaling.
- Models: linear regression, ridge, lasso and elastic net (`glmnet`, cross-validated), averaged into one prediction.

### 4. Black Friday Sales
[`black-friday-sales/black_friday_sales.R`](black-friday-sales/black_friday_sales.R)

Predict a customer's purchase amount for a product from customer demographics and product categories.
- Missing secondary product categories set to zero; binary and ordinal text fields converted to integers.
- Factor levels aligned between training and test sets so unseen categories don't break encoding.
- One-hot encoding with a custom `factor2dummy` function.
- Model: linear regression.

### 5. Footfall in parks
[`park-footfall/park_footfall.R`](park-footfall/park_footfall.R)

Predict daily visitor counts in parks from date, weather and pollution measurements.
- Date features: year, month, day of week, day of year.
- Two-step imputation: first, random draws from observations of the same location type within ±10 days of the year (wrapping around the new year); then sequential regression, where each weather variable is predicted from the variables already completed.
- Centring and scaling.
- Models: linear regression, ridge, lasso, random forest.

### 6. Footfall in parks, lagged and interaction features
[`park-footfall/park_footfall_lags.R`](park-footfall/park_footfall_lags.R)

A second version of solution 5 with a richer feature set.
- Imputation as in solution 5, with a 5-day window.
- Weather values of the same park for the previous 5 days (`features.past`).
- Pairwise products of numeric features (`features.interact2`).
- Models: linear regression, ridge and lasso (cross-validated), random forest.

### 7. HR Analytics
[Competition page](https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018-1/) · [`hr-analytics/hr_analytics.ipynb`](hr-analytics/hr_analytics.ipynb) · 110th of 6 834 (top 1.6 %)

Predict which employees will be promoted from demographics, training scores and performance history.
- Missing education as its own category; missing previous-year rating set to the median.
- One-hot encoding fitted on training and test sets together; standardisation.
- Class imbalance (about 9 % positives) handled with XGBoost `scale_pos_weight`.
- Hyperparameters tuned by `GridSearchCV` on F1 score.
- Models: logistic regression, XGBoost.

### 8. Recommender Engine
[`recommender-engine/recommender_engine.R`](recommender-engine/recommender_engine.R)

Predict how many attempts (in ranges) a user needs to solve a programming problem, from user profiles, problem metadata and past submissions.
- Three tables joined; median and "unknown" imputation.
- Time since registration and since last login from timestamps.
- Problem tags split into a binary tag matrix; one-hot encoding with `fastDummies`.
- Custom weighted F1 score for the skewed target.
- Models: Poisson regression, random forest.

### 9. Predict Number of Upvotes
[Competition page](https://datahack.analyticsvidhya.com/contest/enigma-codefest-machine-learning-1/) · [`upvotes-prediction/upvotes_prediction.ipynb`](upvotes-prediction/upvotes_prediction.ipynb) · 396th of 2 320 (top 17.1 %)

Predict how many upvotes a question gets from the author's reputation, answer and view counts and the question tag.
- Log transform of the skewed target and predictors.
- Number of posts per user as an activity feature.
- Polynomial features up to degree 5; one-hot tags; standardisation.
- Models: linear, lasso and ridge regression (ridge submitted), XGBoost random forest regressor.

## Text

### 10. SentiRuEval-2016
[`sentirueval-2016/sentirueval_2016.R`](sentirueval-2016/sentirueval_2016.R)

Classify Russian tweets about banks and telecom companies as positive, negative or neutral.
- Parsing the XML source files.
- Text cleaning with `tm`: URL and Latin-script word removal, Russian stop words, stemming, bag of words.
- PCA on the term-document matrix, keeping components that explain 85–90 % of the variance.
- Custom oversampling and undersampling for the imbalanced classes.
- Custom macro and micro F1 and misclassification rate.
- Models: SVM, multinomial logistic regression (`nnet::multinom`), LDA, QDA, logistic regression, decision tree, kNN, naive Bayes, linear regression, and a majority vote.

### 11. Identify the Sentiments
[Competition page](https://datahack.analyticsvidhya.com/contest/linguipedia-codefest-natural-language-processing-1/) · [`identify-the-sentiments/identify_the_sentiments.ipynb`](identify-the-sentiments/identify_the_sentiments.ipynb) · 225th of 2 424 (top 9.3 %)

Classify tweets as positive or negative.
- Meta-features: word and character counts, average word length, hashtag, number and uppercase counts, TextBlob polarity of the raw tweet.
- Cleaning: lowercase English tokens, NLTK stop words except negations, Porter stemming, removal of the 10 most frequent words.
- TF-IDF on 1- to 3-grams.
- Stacking: a TF-IDF logistic regression produces out-of-fold predictions that are combined with the meta-features in a second logistic regression; `class_weight='balanced'`.

### 12. Twitter Sentiment Analysis
[Competition page](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/) · [`twitter-sentiment-analysis/twitter_sentiment_analysis.ipynb`](twitter-sentiment-analysis/twitter_sentiment_analysis.ipynb) · 408th of 10 321 (top 4.0 %)

Detect racist or sexist tweets; about 7 % of tweets are positive.
- The same meta-features, cleaning and stacking design as solution 11, with unigram TF-IDF.
- Models: logistic regression as both base and meta model, `class_weight='balanced'`.

## Images and audio

### 13. Age Detection
[`age-detection/age_detection.R`](age-detection/age_detection.R)

Classify the age group (young, middle, old) of people from face images.
- Images resized to 28×28 and converted to grayscale with `imager`; histogram equalisation.
- Pixels flattened into 784 features.
- Random undersampling of the larger classes.
- Models: naive Bayes, random forest.

### 14. Digit Recognition
[`digit-recognition/digit_recognition.R`](digit-recognition/digit_recognition.R)

Recognise handwritten digits from 28×28 images.
- Images read with `imager` and flattened into pixel features.
- Pixels that are blank across the training set removed.
- Optional binarisation with a custom threshold function.
- Models: naive Bayes, random forest.

### 15. Identify the Apparels
[Competition page](https://datahack.analyticsvidhya.com/contest/practice-problem-identify-the-apparels/) · [`identify-the-apparels/identify_the_apparels.ipynb`](identify-the-apparels/identify_the_apparels.ipynb) · 203rd of 2 310 (top 8.8 %)

Classify 28×28 grayscale clothing images into 10 classes (Fashion-MNIST).
- Hand-crafted features instead of a neural network: histogram of oriented gradients (81 features per image); Sobel edges were explored but not used in the model.
- `GridSearchCV` tuning.
- Models: Gaussian naive Bayes, random forest, XGBoost random forest.

### 16. Urban Sound Classification
[`urban-sound-classification/urban_sound_classification.R`](urban-sound-classification/urban_sound_classification.R)

Classify urban sound clips (sirens, dog barks, drilling, engines and others).
- Audio read with `tuneR` and `seewave`; corrupted files skipped; stereo to mono; resampling to 44.1 kHz.
- Every clip cut or zero-padded to 2 seconds.
- Mel-frequency cepstral coefficients (MFCC) flattened into features.
- Random undersampling of the larger classes.
- Models: naive Bayes, random forest.

## Time series

### 17. Time Series Forecasting
[Competition page](https://datahack.analyticsvidhya.com/contest/practice-problem-time-series-2/) · [`time-series-forecasting/time_series_forecasting.R`](time-series-forecasting/time_series_forecasting.R)

Forecast an hourly count series (`Count`) from timestamps only.
- Calendar features from timestamps with a custom `expand.dates` function: year, month, day of month, day of week, day of year, week of year, hour (the function also supports minute, time of day and part of day).
- One-hot encoding of the calendar features; year enters as a number so the trend carries into the test period.
- Model: Poisson regression with log link, capturing trend and seasonality through the calendar features.
