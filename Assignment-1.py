import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv('gdp-per-capita-worldbank.csv')
life_exp = pandas.read_csv('life-expectancy.csv')
years = life_exp['Life expectancy at birth (historical)']


gdp = df["GDP per capita, PPP (constant 2017 international $)"]

plt.hist(gdp, bins=30)
plt.show()


#hejejej
#ok