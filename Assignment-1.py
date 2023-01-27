import pandas
import matplotlib.pyplot as plt
import numpy as np

# A
gdp_per_capita = pandas.read_csv('gdp-per-capita-worldbank.csv')
life_exp = pandas.read_csv('life-expectancy.csv')
gdp_per_capita_2020 = gdp_per_capita[(gdp_per_capita['Year'] == 2020) & (gdp_per_capita['Code']) & (gdp_per_capita['Entity'] != "World")]
life_exp_2020 = life_exp[(life_exp['Year'] == 2020) & (life_exp['Code']) & (life_exp['Entity'] != "World")]

merged = gdp_per_capita_2020.merge(life_exp_2020, on= ['Entity', 'Code', 'Year'])

merged.plot.scatter(x= 'GDP per capita, PPP (constant 2017 international $)', y= 'Life expectancy at birth (historical)', c= 'red')
plt.xlabel('GDP per capita')
plt.ylabel('Life expectancy')
plt.title(' GDP per capita vs life expectancy for 2020')
plt.show()

# B
mean = np.mean(life_exp_2020['Life expectancy at birth (historical)'])
print(mean)
std = np.std(life_exp_2020['Life expectancy at birth (historical)'])
mean_and_1std = mean + std
life_exp_countries = list(life_exp_2020[life_exp_2020['Life expectancy at birth (historical)'] > mean_and_1std]['Entity'])

print(f"Countries with life expectancy higher than one standard deviation above mean: {life_exp_countries}")


# C (we used median on both GDP and life expectancy)
total_gdp = pandas.read_csv('gross-domestic-product.csv')
total_gdp_2020 = total_gdp[(total_gdp['Year'] == 2020) & (total_gdp['Code']) & (total_gdp['Entity'] != "World")]
median_gdp = np.median(total_gdp_2020['GDP (constant 2015 US$)'])
mean_gdp = np.mean(total_gdp_2020['GDP (constant 2015 US$)'])
print(mean_gdp)
print(median_gdp)
median_life_exp = np.median(life_exp_2020['Life expectancy at birth (historical)'])
mean_life_exp = np.mean(life_exp_2020['Life expectancy at birth (historical)'])
print(median_life_exp)
print(mean_life_exp)
low_gdp = (total_gdp_2020[(total_gdp_2020['GDP (constant 2015 US$)'] < mean)])
print(len(low_gdp))
high_life_exp = (life_exp_2020[(life_exp_2020['Life expectancy at birth (historical)'] > median_life_exp)])
merged_2 = low_gdp.merge(high_life_exp)
low_gdp_high_life_exp_countries = list(merged_2['Entity'])

print(f"Countries with low GDP but high life expectancy: {low_gdp_high_life_exp_countries}")


# D
high_gdp = (total_gdp_2020[(total_gdp_2020['GDP (constant 2015 US$)'] > median_gdp)])
low_life_exp = (life_exp_2020[(life_exp_2020['Life expectancy at birth (historical)'] < median_life_exp)])
merged_3 = high_gdp.merge(low_life_exp)
high_gdp_low_life_exp_countries = list(merged_3['Entity'])

print(f"Countries with high GDP but low life expectancy: {high_gdp_low_life_exp_countries}")


# E
median_gdp_capita = np.median(gdp_per_capita_2020['GDP per capita, PPP (constant 2017 international $)'])
high_gdp_capita = (gdp_per_capita_2020[(gdp_per_capita_2020['GDP per capita, PPP (constant 2017 international $)'] > median_gdp_capita)])
low_life_exp = (life_exp_2020[(life_exp_2020['Life expectancy at birth (historical)'] < median_life_exp)])
merged_4 = high_gdp_capita.merge(low_life_exp)
high_gdp_capita_low_life_exp_countries = list(merged_4['Entity'])

print(f"Countries with high GDP per capita but low life expectancy: {high_gdp_capita_low_life_exp_countries}")



















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