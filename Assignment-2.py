import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
hemnet_data = pandas.read_csv('data_assignment_2.csv')

#  Find a linear regression model that relates the living area to the selling price.
# If you did any data cleaning step(s), describe what you did and explain why.
#  What are the values of the slope and intercept of the regression line?

hemnet_data.plot.scatter(x= 'Living_area', y= 'Selling_price', c= 'red')
plt.xlabel('Living_area')
plt.ylabel('Selling_price')
plt.title('Living_area vs Selling_price')
plt.show()

model = LinearRegression()
model.fit(x[:, np.newaxis], y)
xfit = np.array([0, 10])
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
