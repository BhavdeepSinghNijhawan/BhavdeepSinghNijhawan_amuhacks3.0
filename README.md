<h1 align="center">MatrixVision</h1>

## Confusion Matrix

A **confusion matrix** is a performance measurement tool used in **machine learning** and statistics to evaluate the accuracy of a classification model. It is a table that allows visualization of the performance of an algorithm by comparing the predicted and actual classes of a dataset.

A confusion matrix typically has four cells, representing the following:

- **True Positives (TP):** The number of data points correctly classified as belonging to the positive class. **Instances that are correctly predicted as positive**.
- **True Negatives (TN):** The number of data points correctly classified as belonging to the negative class. **Instances that are correctly predicted as negative**.
- **False Positives (FP):** The number of data points incorrectly classified as belonging to the positive class (actually negative). **Instances that are incorrectly predicted as positive (Type I error)**.
- **False Negatives (FN):** The number of data points incorrectly classified as belonging to the negative class (actually positive). **Instances that are incorrectly predicted as negative (Type II error)**.

A confusion matrix provides insight into the performance of a classification model, including metrics such as **accuracy**, **precision**, **recall**, and **F1-score**. It is particularly useful for evaluating the performance of binary classification models but can be extended to multi-class classification problems as well.

### Scatter Plot

<p align="center">
  <img src="https://github.com/BhavdeepSinghNijhawan/MatrixVision/assets/143419096/8885e86f-92f3-4bd7-97ef-e2988a189ee8" />
</p>

### Confusion Matrix

<p align="center">
  <img src="https://github.com/BhavdeepSinghNijhawan/MatrixVision/assets/143419096/84751a93-cb06-4156-b60d-b1c162392d6d" />
</p>

### MATLAB

```
clc;
clear all;
close all;
warning off;
```

- **`clc;`:** Clears the command window.
- **`clear all;`:** Clears all variables from the workspace.
- **`close all;`:** Closes all figure windows.
- **`warning off;`:** Turns off all warnings (not generally recommended unless you are sure you want to suppress warnings).

```
M = readtable('M.txt');
J = readtable('J.txt');
V = readtable('V.txt');
plot(M.Var2, M.Var3);
axis equal;
figure;
plot(J.Var2, J.Var3);
axis equal;
figure;
plot(V.Var2, V.Var3);
axis equal;
```

- **`M = readtable('M.txt');`:** Reads data from the file M.txt into a table M.
- **`J = readtable('J.txt');`:** Reads data from the file J.txt into a table J.
- **`V = readtable('V.txt');`:** Reads data from the file V.txt into a table V.
- **`plot(M.Var2, M.Var3);`:** Plots M.Var3 against M.Var2.
- **`axis equal;`:** Sets the aspect ratio of the plot to be equal.
- **`figure;`:** Opens a new figure window.
- **`plot(J.Var2, J.Var3);`:** Plots J.Var3 against J.Var2 in the new figure.
- **`axis equal;`:** Sets the aspect ratio of the plot to be equal for the second plot.
- **`figure;`:** Opens another new figure window.
- **`plot(V.Var2, V.Var3);`:** Plots V.Var3 against V.Var2 in the new figure.
- **`axis equal;`:** Sets the aspect ratio of the plot to be equal for the third plot.

```
durM = M.Var1(end);
durJ = J.Var1(end);
durV = V.Var1(end);
aratioM = range(M.Var3) / range(M.Var2);
aratioJ = range(J.Var3) / range(J.Var2);
aratioV = range(V.Var3) / range(V.Var2);
```

- **`durM = M.Var1(end);`:** Assigns the last value of M.Var1 to durM.
- **`durJ = J.Var1(end);`:** Assigns the last value of J.Var1 to durJ.
- **`durV = V.Var1(end);`:** Assigns the last value of V.Var1 to durV.
- **`aratioM = range(M.Var3) / range(M.Var2);`:** Computes the aspect ratio of M.Var3 to M.Var2.
- **`aratioJ = range(J.Var3) / range(J.Var2);`:** Computes the aspect ratio of J.Var3 to J.Var2.
- **`aratioV = range(V.Var3) / range(V.Var2);`:** Computes the aspect ratio of V.Var3 to V.Var2.

```
figure;
features = readtable('Features.txt'); % Read features from file
gscatter(features.Var1, features.Var2, features.Var3);
knnmodel = fitcknn(features, 'Var3');
testdata = readtable('testdata.txt');
predictions = predict(knnmodel, testdata(:, 1:2));
Observation = [testdata(:, end) predictions];
knnmodel = fitcknn(features, 'Var3', 'NumNeighbors', 5);
predictions = predict(knnmodel, testdata(:, 1:2));
Observation = [testdata(:, end) predictions];
iscorrect = string(predictions) == string(testdata.Var3);
accuracy = sum(iscorrect) / numel(iscorrect);
misclassrate = sum(~iscorrect) / numel(iscorrect);
disp(['Accuracy: ', num2str(accuracy)]);
disp(['Misclassification Rate: ', num2str(misclassrate)]);
figure;
confusionchart(testdata.Var3, predictions);
```

- **`figure;`:** Opens a new figure window.
- **`features = readtable('Features.txt');`:** Reads data from Features.txt into a table features.
- **`gscatter(features.Var1, features.Var2, features.Var3);`:** Creates a scatter plot of features.Var1 vs features.Var2, colored by features.Var3.
- **`knnmodel = fitcknn(features, 'Var3');`:** Trains a kNN classifier using features.Var1 and features.Var2 to predict features.Var3.
- **`testdata = readtable('testdata.txt');`:** Reads test data from testdata.txt into a table testdata.
- **`predictions = predict(knnmodel, testdata(:, 1:2));`:** Uses the trained knnmodel to predict Var3 values for the test data based on the first two columns of testdata.
- **`Observation = [testdata(:, end) predictions];`:** Combines the actual and predicted values into a matrix Observation.
- **`knnmodel = fitcknn(features, 'Var3', 'NumNeighbors', 5);`:** Trains a new kNN classifier with 5 neighbors specified.
- **`predictions = predict(knnmodel, testdata(:, 1:2));`:** Predicts again using the new model.
- **`Observation = [testdata(:, end) predictions];`:** Updates Observation with new predictions.
- **`iscorrect = string(predictions) == string(testdata.Var3);`:** Checks which predictions match the actual values.
- **`accuracy = sum(iscorrect) / numel(iscorrect);`:** Calculates accuracy as the proportion of correct predictions.
- **`misclassrate = sum(~iscorrect) / numel(iscorrect);`:** Calculates the misclassification rate.
- **`disp(['Accuracy: ', num2str(accuracy)]);`:** Displays the accuracy.
- **`disp(['Misclassification Rate: ', num2str(misclassrate)]);`:** Displays the misclassification rate.
- **`figure;`:** Opens another new figure window.
- **`confusionchart(testdata.Var3, predictions);`:** Generates a confusion chart comparing the actual vs predicted classes.

## TOOL

- [MATLAB](https://matlab.mathworks.com/)

## CONTRIBUTOR

- [Bhavdeep Singh Nijhawan](https://www.linkedin.com/in/bhavdeep-singh-nijhawan-739634280)
