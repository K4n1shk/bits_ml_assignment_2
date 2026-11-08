# ECG Heartbeat Classification – ML Assignment 2

## a. Problem Statement

The goal is to classify a single ECG heartbeat as **normal** or **abnormal** from 187 signal values. Five classifiers are trained on the same split and compared using accuracy, AUC, precision, recall, F1 and MCC. The Streamlit app loads the saved models and evaluates them on an uploaded labelled CSV.

## b. Dataset Description

- **Source:** Kaggle – [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) (PTB Diagnostic ECG Database part). The original recordings are from [PhysioNet PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/), and the beats were extracted and preprocessed by Kachuee et al. (2018), [arXiv:1805.00794](https://arxiv.org/abs/1805.00794).
- **Files used:** `ptbdb_normal.csv` and `ptbdb_abnormal.csv`
- **Instances:** 14,552 heartbeats (4,046 normal, 10,506 abnormal, about 28% / 72%). 7 duplicate rows were removed, leaving 14,545.
- **Features:** 187 numeric features (`f0` to `f186`). Each one is a sample of the ECG signal for one beat, sampled at 125 Hz and normalised to the range 0–1. Shorter beats are padded with zeros at the end.
- **Target:** `target` – 0 = normal, 1 = abnormal (binary classification)
- **Missing values:** none
- **Split:** 80% train (11,636 rows) and 20% test (2,909 rows), stratified on the target. The test split is saved as `test_data.csv` and is the file meant to be uploaded in the app.

The split is by beat, not by patient. Results on this test set should not be read as performance on new patients. Because the classes are uneven, MCC and F1 add useful context to accuracy.

## c. i: GitHub Repository Link: https://github.com/K4n1shk/bits_ml_assignment_2
## c. ii: App Link: https://bitsmlassignment2-dnpjusf2grxcv8wijdrrmy.streamlit.app/

Repository structure:
```
├── app.py                 # streamlit app
├── model_training.ipynb   # data loading, EDA, training, evaluation, saving models
├── model/                 # saved models (joblib .pkl)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── svm.pkl
├── test_data.csv          # test split (187 features + target)
├── requirements.txt
└── README.md
```

## d. Models Used

All 5 models were trained on the same train split and evaluated on the same test split (`test_data.csv`).

| Model | Notes |
|---|---|
| Logistic Regression | StandardScaler + LogisticRegression (max_iter=2000). Balanced class weights gave nearly the same training CV MCC and lower CV accuracy, so the default was kept. |
| Decision Tree | DecisionTreeClassifier(max_depth=15). The depth was chosen from five-fold training CV; the unrestricted tree fit every training row. |
| KNN | StandardScaler + KNeighborsClassifier(k=3). One neighbour had the highest CV accuracy, but three was chosen for less brittle predictions and probabilities with more than two values. |
| Naive Bayes | GaussianNB. It had higher training CV MCC than MultinomialNB, which had higher CV accuracy. |
| SVM | StandardScaler + SVC (RBF kernel, C=50 from a three-fold CV grid), wrapped in CalibratedClassifierCV to provide probabilities for AUC. |

Development comparisons use cross-validation on the training split. The test split is used for the final comparison of the five chosen models. C=50 was the largest SVM value tried, so the grid does not establish a global optimum.

Precision, Recall and F1 are for the abnormal class (label 1).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8223 | 0.8755 | 0.8538 | 0.9095 | 0.8808 | 0.5372 |
| Decision Tree | 0.9292 | 0.9171 | 0.9584 | 0.9429 | 0.9506 | 0.8262 |
| KNN | 0.9343 | 0.9724 | 0.9609 | 0.9476 | 0.9542 | 0.8385 |
| Naive Bayes | 0.6091 | 0.7655 | 0.8880 | 0.5248 | 0.6597 | 0.3197 |
| Support Vector Machine | **0.9656** | **0.9907** | **0.9766** | **0.9757** | **0.9762** | **0.9145** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Abnormal recall is high (0.91), but it gets only about 60% of normal beats right (327 of 809 were predicted as abnormal). MCC is 0.54. Balanced class weights did not improve training CV MCC. |
| Decision Tree | Much better than LR with an MCC of 0.83. Precision and recall are balanced. Without a depth limit it reached 100% train accuracy, and even with max_depth=15 it has the biggest train/test gap (~6%), so it still overfits a bit. Its AUC (0.92) is lower than KNN/SVM because the leaf probabilities are mostly close to 0 or 1. |
| KNN | Second best on this split (MCC 0.84, AUC 0.97). One neighbour had higher CV accuracy, but three gives probabilities beyond only 0 and 1. The saved pipeline stores the training rows and is about 17.5 MB. |
| Naive Bayes | Weakest of the five (accuracy 0.61, MCC 0.32). It misses 998 of 2,100 abnormal beats. Adjacent signal samples are related, which may work against the model's feature-independence assumption. MultinomialNB had lower training CV MCC. |
| Support Vector Machine | Best on all six metrics for this test split: accuracy 0.97, AUC 0.99, MCC 0.91. It misclassifies 49 normal and 51 abnormal beats. Its train/test accuracy gap is about 3 percentage points. |
| **Overall winner** | **SVM** leads this beat-level comparison. KNN and Decision Tree follow on MCC. These results do not establish clinical performance or generalization to new patients. |


## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
