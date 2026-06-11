# Predicting Cancer Occurrence

# Definition of the Business Problem: Predicting the Occurrence of Breast Cancer
# https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Step 1 - Collecting the Data

# Breast cancer data include 569 observations of cancer biopsies,
# each with 32 characteristics (variables). A feature is a number of
# identification (ID), another is cancer diagnosis, and 30 are laboratory measurements
# numeric. The diagnosis is coded as "M" to indicate malignant or "B" to indicate
# indicate benign.
data <- read.csv("dataset.csv", stringsAsFactors = FALSE)
str(data)

## Step 2 - Pre-Processing

# Deleting the ID column
# Regardless of the machine learning method, it should always be excluded
# ID variables. Otherwise this can lead to wrong results because the ID
# can be used to uniquely "predict" each example. Therefore, a model
# that includes an identifier can suffer from overfitting,
# and it will be very difficult to use it to generalize other data.
data$id = NULL


# Adjusting the label of the target variable
data$diagnosis = sapply(data$diagnosis, function(x){ifelse(x=='M', 'Malign', 'Benign')})

# Many classifiers require variables to be of type Factor
table(data$diagnosis)
data$diagnosis <- factor(data$diagnosis, levels = c("Benign", "Malign"), labels = c("Benign", "Malign"))
str(data$diagnosis)

# Checking the aspect ratio
round(prop.table(table(data$diagnosis)) * 100, digits = 1) 


# Measures of Central Tendency
# We detected a scaling problem between the data, which then needs to be normalized
# The distance calculation done by kNN is dependent on the scale measurements in the input data.
summary(data[c("radius_mean", "area_mean", "smoothness_mean")])


# Creating a normalization function
## Note: this min-max rescaling to the [0, 1] range is normalization (not standardization).

normalizes <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


# Normalizing the data
data_norm <- as.data.frame(lapply(data[2:31], normalizes))

## Step 3: Training the model with KNN

# Loading the library package
# install.packages("class")
library(class)

# Creating training data and test data
# set.seed makes the random train/test split reproducible across runs.
set.seed(42)
sample_df <- sample(c(TRUE, FALSE), nrow(data_norm),replace = TRUE, prob = c(0.7, 0.3))

training_data <- data_norm[sample_df,]
test_data <- data_norm[!sample_df,]

# Creating the labels for the training and test data
training_data_labels <- data[sample_df, 1]
testing_data_labels <- data[!sample_df, 1]
length(training_data_labels)
length(testing_data_labels)

# Creating the model
model_knn_v1 <- knn(train = training_data, 
                     test = test_data,
                     cl = training_data_labels, 
                     k = 21)


# The knn() function returns a factor object with the predictions for each example in the test dataset
summary(model_knn_v1)


## Step 4: Evaluating and Interpreting the model

# Loading gmodels
library(gmodels)

# Creating a cross table of predicted data x current data

CrossTable(x = testing_data_labels, y = model_knn_v1, prop.chisq = FALSE)

# Interpreting the Results
# The cross table shows 4 cells, the false/true positive and negative counts.
# Columns are the observed (true) labels; rows are the predicted labels.
# Here "positive" means the disease (Malign) per the convention defined below.

# Programmatic accuracy (do not hard-code it: it must match the table above):
accuracy_knn_v1 <- mean(model_knn_v1 == testing_data_labels)
print(accuracy_knn_v1)

# Mapping the four cells to the disease-positive convention (positive = Malign):
# Scenario 1: Benign (Observed) x Benign (Predicted)  - true negative
# Scenario 2: Malign (Observed) x Benign (Predicted)  - false negative (a missed cancer, the worst error here)
# Scenario 3: Benign (Observed) x Malign (Predicted)  - false positive
# Scenario 4: Malign (Observed) x Malign (Predicted)  - true positive

# Reading the Confusion Matrix (Perspective of having or not having the disease):

# True Negative = model predicted NO disease and the person did NOT have it
# False Positive = model predicted disease but the person did NOT have it
# False Negative = model predicted NO disease but the person DID have it
# True Positive = model predicted disease and the person DID have it

# False Positive - Type I Error
# False Negative - Type II Error
# In a cancer screening setting, a False Negative (missed Malign tumor) is the
# most costly error, so it should be reported, not hidden behind a single accuracy number.


## Step 5: Optimizing Model Performance

# Using the scale() function to standardize the data (z-score)
## obs: R's scale() performs z-score standardization (subtract mean, divide by sd), not normalization.

data_z <- as.data.frame(scale(data[-1]))

# Confirming successful transformation
summary(data_z$area_mean)

# Creating new training and testing datasets
set.seed(42)
sample_df <- sample(c(TRUE, FALSE), nrow(data_z),replace = TRUE, prob = c(0.7, 0.3))

training_data <- data_z[sample_df,]
test_data <- data_z[!sample_df,]

# Creating the labels for the training and test data
training_data_labels <- data[sample_df, 1]
testing_data_labels <- data[!sample_df, 1]
length(training_data_labels)
length(testing_data_labels)

# Creating the model
model_knn_v2 <- knn(train = training_data, 
                    test = test_data,
                    cl = training_data_labels, 
                    k = 21)

# Creating a cross table of predicted date x current date
CrossTable(x = testing_data_labels, y = model_knn_v2, prop.chisq = FALSE)



# Try different values for k


# Step 6: Building a model with Support Vector Machine Algorithm (SVM)

# Prepare the dataset
data <- read.csv("dataset.csv", stringsAsFactors = FALSE)
data$diagnosis <- factor(data$diagnosis)

data$id = NULL

## Create column to separate training and test data
## The split index is random; set.seed keeps it reproducible across runs.
set.seed(42)
data[,'index'] <- ifelse(runif(nrow(data)) < 0.7,1,0)

# training and testing date
trainset <- data[data$index==1,]
testset <- data[data$index==0,]

# Get the index
trainColNum <- grep('index', names(trainset))

# Remove index from datasets
trainset <- trainset[,-trainColNum]
testset <- testset[,-trainColNum]

# Get column index of target variable in dataset
## Choose the column that starts with 'diag'
typeColNum <- grep('diag',names(data))

# Create the model
# We set the kernel to radial, as this dataset doesn't have a
# linear plane that can be drawn
library(e1071)
#?svm

model_svm_v1 <- svm(diagnosis ~ ., 
                     data = na.omit(trainset), 
                     type = 'C-classification', 
                     kernel = 'radial') 


# Forecasts

# Predictions on training dates
pred_train <- predict(model_svm_v1, trainset)

# Percentage of correct predictions with training dataset
mean(pred_train == trainset$diagnosis)


# Forecasts on test date
pred_test <- predict(model_svm_v1, testset)

# Percentage of correct predictions with test dataset
mean(pred_test == testset$diagnosis)

# Confusion Matrix
table(pred_test, testset$diagnosis)


# Step 7: Building a model with a Decision Tree (rpart / CART)
# Note: this is a single CART decision tree from the rpart package, not a random forest.
# A random forest would require an ensemble package such as randomForest or ranger.

# Creating the model
library(rpart)
model_tree_v1 = rpart(diagnosis ~ .,
                    data = trainset,
                    control = rpart.control(cp = .0005))

# Forecasts on test date
tree_pred = predict(model_tree_v1, testset, type='class')

# Percentage of correct predictions with test dataset
mean(tree_pred==testset$diagnosis)

# Confusion Matrix
table(tree_pred, testset$diagnosis)