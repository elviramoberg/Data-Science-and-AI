import pandas
import matplotlib.pyplot as plt
import numpy as np

gdp = pandas.read_csv('gdp-per-capita-worldbank.csv')
life_exp = pandas.read_csv('life-expectancy.csv')
years = life_exp['Life expectancy at birth (historical)']

one_year_gdp = gdp[gdp['Year'] == 2020]
gdp_countries = list(one_year_gdp['Entity'])
one_year_gdp_list = list(one_year_gdp["GDP per capita, PPP (constant 2017 international $)"])
#print(len(one_year_gdp_list))

one_year_life = life_exp[life_exp['Year'] == 2020]
print(one_year_life)
one_year_life_list = list(one_year_life['Life expectancy at birth (historical)'])
#print(len(one_year_life_list))

#plt.hist(one_year_gdp, bins=50)
#plt.show()



filtered_values = np.where((gdp['Year'] == 2020) & (gdp['Entity'] != 'World'))
#print(filtered_values)


dataframe_gdp = pandas.DataFrame(gdp, columns=["country", "code", "year", "GDP per capita"])
scaled_df = dataframe_gdp[dataframe_gdp["year"] == 2020]
#print(scaled_df)

#plt.hist(gdp, bins=30)
#plt.show()


#hejejej
#ok