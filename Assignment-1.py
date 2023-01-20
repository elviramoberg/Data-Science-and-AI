import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv('gdp-per-capita-worldbank.csv')

gdp = df['GDP per capita']

plt.hist(gdp, bins=50)
plt.show()


#hejejej
#ok