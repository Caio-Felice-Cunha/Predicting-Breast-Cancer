# Predicting Breast Cancer

Predicting breast cancer occurrence (malignant vs. benign) from biopsy measurements with machine learning in R.

[Script](https://github.com/Caio-Felice-Cunha/Predicting-Breast-Cancer/blob/main/Predicting%20Cancer%20Occurrence.R) <br>
[The report (PDF)](https://github.com/Caio-Felice-Cunha/Predicting-Breast-Cancer/blob/main/Predicting-Cancer-Occurrence.pdf)

![image](https://user-images.githubusercontent.com/111542025/236818599-0a5a1868-7290-4854-959e-cbf662da4f01.png)

## This is the 1st version

## Business Problem

Breast tumors can be benign or malignant, and a malignant tumor missed at screening is the costly error. The goal is to classify a tumor as benign or malignant from 30 numeric cell-nucleus measurements, and to compare a few standard classifiers on this task.

Data is the Breast Cancer Wisconsin (Diagnostic) dataset, collected from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). The copy used here is committed as `dataset.csv`.

## Dataset

569 biopsies, 32 columns (an `id`, the `diagnosis` label, and 30 numeric measurements). Class balance:

| Class | Count | Share |
| --- | --- | --- |
| Benign (B) | 357 | 62.7% |
| Malign (M) | 212 | 37.3% |

The `id` column is dropped before modeling so the model cannot "predict" a row from its identifier.

## Solution Strategy

Built in R / RStudio.

* Step 01: Collecting the data.
* Step 02: Pre-processing, adjusting the target variable, and normalizing (min-max) the features.
* Step 03: Training a KNN model (k = 21) on the normalized data.
* Step 04: Evaluating and interpreting the KNN model with a confusion matrix.
* Step 05: Optimizing performance by re-scaling with z-score standardization and re-running KNN.
* Step 06: Building a Support Vector Machine (SVM, radial kernel) model.
* Step 07: Building a Decision Tree (rpart / CART) model.

## Results

Numbers below are taken from the stored report (`Predicting-Cancer-Occurrence.pdf`). Test-set sizes vary because the train/test split was random and unseeded in the run that produced the PDF. The script now calls `set.seed(42)` before each split, so a fresh run is reproducible but its exact counts may differ from the PDF.

| Model | Scaling | Test accuracy | False negatives (missed malignant) |
| --- | --- | --- | --- |
| KNN, k = 21 | min-max normalized | 148/159 = 93.1% | 10 |
| KNN, k = 21 | z-score standardized | 166/174 = 95.4% | 8 |
| SVM, radial kernel | raw features | 95.2% (train 98.3%) | 4 |
| Decision Tree (rpart) | raw features | 91.6% | 6 |

Confusion matrices from the report:

* KNN (min-max), test n = 159: benign observed predicted 98 benign / 1 malign; malign observed predicted 10 benign / 50 malign.
* KNN (z-score), test n = 174: benign observed predicted 107 benign / 0 malign; malign observed predicted 8 benign / 59 malign.
* SVM, test set: predicted benign 94 B / 4 M; predicted malign 4 B / 64 M.
* Decision Tree (rpart), test set: predicted benign 92 B / 8 M; predicted malign 6 B / 60 M.

The headline metric for a screening problem is not raw accuracy but the false-negative count (malignant tumors classified as benign). On this run the z-score KNN and the SVM were the strongest, each leaving fewer missed malignancies than the min-max KNN.

Note on an earlier claim: a previous version of the script and report described the first KNN model as a "98% hit rate." That number did not match the confusion table stored in the same report (148/159, with 10 false negatives). The script now computes accuracy programmatically with `mean(model_knn_v1 == testing_data_labels)` instead of hard-coding it.

## How to Run

1. Install R (4.x). RStudio is optional.
2. Install the required packages:

   ```r
   install.packages(c("class", "gmodels", "e1071", "rpart"))
   ```

3. Run the script from the repository root (so `dataset.csv` is found):

   ```bash
   Rscript "Predicting Cancer Occurrence.R"
   ```

   Or open `Predicting Cancer Occurrence.R` in RStudio and run it top to bottom.

The script reads `dataset.csv`, trains the four models, and prints accuracy and confusion tables to the console.

## Next Steps

* Tune `k` for KNN and `cp` for the decision tree.
* Fit scaling parameters on the training split only (the current script scales over the full dataset before splitting, which leaks test-set statistics).
* Add an actual random-forest model (e.g. `randomForest` or `ranger`) for a fair ensemble comparison.
* Understand the correlation between variables.

## Disclaimer

A good part of this project was done as part of the Data Science Academy "Big Data Analytics with R and Microsoft Azure Machine Learning" course (part of the Data Scientist training).
