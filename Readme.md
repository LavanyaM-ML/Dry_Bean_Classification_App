# Dry Bean Classification ML Assignment 2

## a. Problem statement

Dry beans of different varieties look similar, and sorting them by hand is slow and inconsistent. The goal of this project is to classify a dry bean into one of **7 varieties** (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira) using shape measurements taken from images of the grains.

Five classification models are trained on the same data and compared using six evaluation metrics. The results are shown in an interactive Streamlit app where a test CSV can be uploaded and any model can be evaluated without retraining.

## b. Dataset description

- **Source:** UCI Machine Learning Repository Dry Bean Dataset
- **Problem type:** Multi-class classification (7 classes)
- **Instances:** 13,611 in the original file (duplicate rows were removed during cleaning)
- **Features:** 16 numeric features (minimum required: 12)
  - Size: Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter
  - Shape: AspectRation, Eccentricity, Extent, Solidity, roundness, Compactness
  - ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4
- **Target:** `Class` (SEKER, BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SIRA)
- **Preprocessing:**
  - Checked for missing values and removed duplicate rows
  - Converted class names to numbers with `LabelEncoder`
  - 80/20 train test split, stratified by class (`random_state=42`)
  - `StandardScaler` fitted on the training data only, then applied to the test data

## c. GitHub repository link

<https://github.com/LavanyaM-ML/Dry_Bean_Classification_App>

**Live Streamlit app:** <PASTE YOUR STREAMLIT APP LINK HERE>

## d. Models used

All five models were trained on the same training split and evaluated on the same test split.
Precision, Recall and F1 are weighted averages over the 7 classes. AUC is one-vs-rest (weighted).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.8966 | 0.9363 | 0.8965 | 0.8966 | 0.8964 | 0.8750 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Naive Bayes | 0.8970 | 0.9899 | 0.8997 | 0.8970 | 0.8972 | 0.8762 |
| Support Vector Machine | 0.9210 | 0.9933 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a simple linear model: accuracy 0.9192 and the highest AUC of all models (0.9934). It is only about 0.2 points behind the SVM on accuracy, which suggests the bean classes are largely separable by a linear boundary once the features are scaled. It is also fast to train and easy to interpret. |
| Decision Tree | The weakest model on most metrics (accuracy 0.8966, MCC 0.8750) and clearly the lowest on AUC (0.9363). A single tree gives hard, step-like probability estimates, so its ranking of classes is poorer than the other models, and it tends to overfit the training data. Its main advantage is that it is easy to explain. |
| KNN | Good performance (accuracy 0.9155, MCC 0.8978), close to Logistic Regression and SVM. It works well here because similar beans have similar measurements and the features were scaled. The drawback is that prediction is slower on large data, since it compares each new bean with all stored training rows. |
| Naive Bayes | Accuracy of 0.8970 is among the lowest, but its AUC (0.9899) is high, so its probability ranking of classes is good even when the final label is wrong. The independence assumption is not realistic here because many features (area, perimeter, axis lengths, convex area) are strongly correlated, which limits accuracy. It is the simplest and fastest model. |
| Support Vector Machine | Best on accuracy (0.9210), precision (0.9214), recall (0.9210), F1 (0.9211) and MCC (0.9045), with AUC (0.9933) essentially tied with Logistic Regression. The RBF kernel captures non-linear boundaries between similar-looking classes. Its cost is a longer training time than the other models. |
| **Overall winner for this dataset** | **Support Vector Machine.** It has the best Accuracy, Precision, Recall, F1 and MCC, and its AUC is within 0.0001 of the best. Logistic Regression is a very close second and is a good choice if speed and simplicity matter. The margin between the top three models (SVM, Logistic Regression, KNN) is small, while Decision Tree and Naive Bayes trail by about 2 percentage points in accuracy. |

## Streamlit app features

- Upload a test CSV (it must contain the 16 feature columns and the `Class` column), or use the bundled `test_data.csv`
- Dropdown to select any of the 5 saved models (loaded from disk, no retraining)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Interactive confusion matrix (counts or percentages) and classification report
- One-vs-rest ROC curves for each bean class
- Side-by-side comparison table and chart of all 5 models on the uploaded data
- Predictions table with a filter for wrong predictions, plus a feature scatter plot

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

