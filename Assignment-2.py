#hej
# ------------------------------------------1A-----------------------------------------------
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


hemnet_data = pandas.read_csv('data_assignment_2.csv')

# collecting data about price and living area
x = hemnet_data['Living_area']
y = hemnet_data['Selling_price']
# hemnet_data.plot.scatter(x='Living_area', y='Selling_price', c= 'red')
plt.xlabel('Living area')
plt.ylabel('Selling price')
plt.title('Living area vs Selling price')

model = LinearRegression()
model.fit(x[:, np.newaxis], y)
xfit = np.array([min(x), max(x)])
yfit = model.predict(xfit[:, np.newaxis])

# Plotting the linear regression diagram
plt.xlabel('Living area')
plt.ylabel('Selling price')
plt.title('Living area vs Selling price')
plt.scatter(x, y, c="dodgerblue")
plt.plot(xfit, yfit, label='Regression line', color='deeppink')
plt.legend()
plt.show()

# ------------------------------------------1B-----------------------------------------------
# Values of slope and intercept of the regression line
intercept = model.intercept_
slope = model.coef_
print("Intercept: " + str(round(intercept)))
print("Slope: " + str(slope))

# ------------------------------------------1C-----------------------------------------------
# Calculating predicted prices for areas 100, 150, 200 m2
area_100 = model.predict([[100]])
print("Price for 100 m2: " + str(round(area_100[0])))
area_150 = model.predict([[150]])
print("Price for 150 m2: " + str(round(area_150[0])))
area_200 = model.predict([[200]])
print("Price for 200 m2: " + str(round(area_200[0])))

# ------------------------------------------1D-----------------------------------------------
# Make predictions using your linear regression model
predictions = model.predict(x[:, np.newaxis])

# Calculate the residuals
residuals = y - predictions

# Create a scatter plot of the residuals
plt.scatter(x, residuals, c="dodgerblue")

# Adding a line at y=0
plt.axhline(y=0, color='deeppink')

# Adding labels and a title to the plot
plt.xlabel('Living Area (m2)')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Display the plot
plt.show()

# ------------------------------------------2A-----------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns  # seaborn is a package for nice-looking graphics
from sklearn import metrics

# Loading iris data
iris = load_iris()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)

# Creating a logistic regression model
logisticRegr = LogisticRegression(multi_class='ovr', solver='liblinear')
logisticRegr.fit(x_train, y_train)

# Make predictions on entire test data
predictions = logisticRegr.predict(x_test)

# Use the score method to get the accuracy of model
score = logisticRegr.score(x_test, y_test)

# Creating confusion matrix based on logistic regression model
cm = metrics.confusion_matrix(y_test, predictions)

# Plotting confusion matrix
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'RdPu_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(round(score, 3))
plt.title(all_sample_title, size = 15)
plt.show()

# ------------------------------------------2B-----------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

# Create list of k-values, removing some to prevent ties
k_values = [num for num in range(0, 112) if num % 2 != 0 and num % 3 != 0]

# Create precision lists for uniform and distance weights
dist_prec = []
uni_prec = []

for k in k_values:
    # Create KNN classifyer model for uniform
    knn = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    knn.fit(x_train, y_train)
    prec_pred = knn.predict(x_test)
    precision = metrics.accuracy_score(y_test, prec_pred)
    uni_prec.append(precision)

    # Create KNN classifyer model for distance
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(x_train, y_train)
    prec_pred = knn.predict(x_test)
    precision = metrics.accuracy_score(y_test, prec_pred)
    dist_prec.append(precision)

# Plotting the diagram with uniform and distance as weights
plt.plot(k_values, dist_prec, label="Distance", c='dodgerblue')
plt.plot(k_values, uni_prec, label="Uniform", c='deeppink')
plt.xlabel("k-value")
plt.ylabel("Precision rate")
plt.title("K nearest neighbour model")
plt.show()

# ------------------------------------------2C-----------------------------------------------
# Creating list with three relevant k-values
test_ks = [37, 79, 103]

# Creating KNN models for each k for uniform
for test_k in test_ks:
    knn = KNeighborsClassifier(n_neighbors=test_k, weights="uniform")
    knn.fit(x_train, y_train)
    prec_pred = knn.predict(x_test)
    precision = metrics.accuracy_score(y_test, prec_pred)
    uni_prec.append(precision)

    predictions = knn.predict(x_test)
    score = knn.score(x_test, y_test)

    # Creating confusion matrices based on KNN model
    cm = metrics.confusion_matrix(y_test, predictions)

    # Plotting confusion matrices
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='RdPu_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    title = 'Accuracy Score: {0}'.format(round(score,3)) + "\n k = " + str(test_k)
    plt.title(title, size=15)
    plt.show()
