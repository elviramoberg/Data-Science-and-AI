import pandas
import matplotlib.pyplot as plt
import numpy as np

# A
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

#B
mean_std = np.mean(one_year_life['Life expectancy at birth (historical)']) + np.std(one_year_life['Life expectancy at birth (historical)'])
test = life_exp[(life_exp['Year'] == 2020) & (one_year_life['Life expectancy at birth (historical)'] > mean_std)]
test1 = list(test['Entity'])
print(test1)

#C en standardenhet högt?

mean_std_gdp = np.mean(one_year_gdp['GDP per capita, PPP (constant 2017 international $)'])
print(mean_std_gdp)
std = np.std(one_year_gdp['GDP per capita, PPP (constant 2017 international $)'])
print(std)
testC = life_exp[(life_exp['Year'] == 2020) & (one_year_life['Life expectancy at birth (historical)'] > mean_std) & (one_year_gdp['GDP per capita, PPP (constant 2017 international $)'] < mean_std_gdp)]
testC1 = list(testC['Entity'])
print(testC1)
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