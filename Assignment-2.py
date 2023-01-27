import sys
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns #seaborn is a package for nice-looking graphics
from sklearn import metrics


#1A
hemnet_data = pandas.read_csv('data_assignment_2.csv')

#  Find a linear regression model that relates the living area to the selling price.
# If you did any data cleaning step(s), describe what you did and explain why.
#  What are the values of the slope and intercept of the regression line?

x = hemnet_data['Living_area']
y = hemnet_data['Selling_price']
hemnet_data.plot.scatter(x='Living_area', y='Selling_price', c= 'red')
plt.xlabel('Living_area')
plt.ylabel('Selling_price')
plt.title('Living_area vs Selling_price')

model = LinearRegression()
model.fit(x[:, np.newaxis], y)

xfit = np.array([min(x), max(x)])
yfit = model.predict(xfit[:, np.newaxis])

#1B
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)

m = model.intercept_
k = model.coef_
print(round(m))
print(k)

#1C
area_100 = k*100 + m
print(round(area_100[0]))
area_150 = k*150 + m
print(round(area_150[0]))
area_200 = k*200 + m
print(round(area_200[0]))

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()

#1D
#Hittade detta på en sida för att beräkna residualer, man bör nog göra om det...
regression = LinearRegression.predict()


import statsmodels.api as sm
X = sm.add_constant(x)
reg = sm.OLS(y, X).fit()
hemnet_data["predicted"] = reg.predict(X)
hemnet_data["residuals"] = reg.resid
sns.scatterplot(data=hemnet_data, x="predicted", y="residuals")
plt.axhline(y=0)

#[(k*x1 + m) for x1 in x]
#  actual - predicted
plt.show()



#2A
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# # Loading iris data
# df = load_iris()
#
# # Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
# print("Size of data set" , df.data.shape)
# # Print to show there are 1797 labels (integers from 0-9)
# print("Label Data Shape", df.target.shape)
# plt.figure(figsize=(20,4))
#
# for index, (image, label) in enumerate(zip(df.data[0:5], df.target[0:5])):
#     plt.subplot(1, 5, index + 1)
#     plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
#     plt.title('Training: %i\n' % label, fontsize = 20)
#
# x_train, x_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.25, random_state=0)
# logisticRegr = LogisticRegression(multi_class='ovr', solver='liblinear')
# logisticRegr.fit(x_train, y_train)
# # Returns a NumPy Array
# # Predict for One Observation (image)
# logisticRegr.predict(x_test[0].reshape(1,-1))
# # Predict for Multiple Observations (images) at Once
# logisticRegr.predict(x_test[0:10])
# # Make predictions on entire test data
# predictions = logisticRegr.predict(x_test)
# # Use the score method to get the accuracy of model
# score = logisticRegr.score(x_test, y_test)
# print(score)
# cm = metrics.confusion_matrix(y_test, predictions)
# plt.figure(figsize=(9,9))
#
# sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# all_sample_title = 'Accuracy Score: {0}'.format(score)
# plt.title(all_sample_title, size = 15)
# plt.savefig('toy_Digits_ConfusionSeabornCodementor.png')
#
# plt.show()

#2B
# To get access to a dataset
# from sklearn import datasets
# Import train_test_split function
# from sklearn.model_selection import train_test_split
# Import knearest neighbors Classifier model
# from sklearn.neighbors import KNeighborsClassifier
# Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
# Load dataset
# iris = datasets.load_iris()
# print(iris.feature_names)
# print(iris.target_names)
