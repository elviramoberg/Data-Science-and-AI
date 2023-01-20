import pandas
import matplotlib.pyplot as plt
import numpy as np

gdp = pandas.read_csv('gdp-per-capita-worldbank.csv')
life_exp = pandas.read_csv('life-expectancy.csv')

one_year_gdp = gdp[gdp['Year'] == 2020]

one_year_life = life_exp[life_exp['Year'] == 2020]


merge = one_year_gdp.merge(one_year_life, on= ['Entity', 'Code', 'Year'])
print(merge)

merge.plot.scatter(x= 'GDP per capita, PPP (constant 2017 international $)', y= 'Life expectancy at birth (historical)', c= 'red')
plt.xlabel('GDP per capita')
plt.ylabel('Life expectancy')
plt.title(' GDP per capita vs life expectancy for 2020')



plt.show()














#
# filtered_values = np.where((gdp['Year'] == 2020) & (gdp['Entity'] != 'World'))
# #print(filtered_values)
#
#
# dataframe_gdp = pandas.DataFrame(gdp, columns=["country", "code", "year", "GDP per capita"])
# scaled_df = dataframe_gdp[dataframe_gdp["year"] == 2020]
# #print(scaled_df)
#
# #plt.hist(gdp, bins=30)
# #plt.show()
#
#
# #hejejej
# #ok