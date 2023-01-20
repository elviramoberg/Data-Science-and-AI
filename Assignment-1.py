import pandas
import matplotlib.pyplot as plt
import numpy as np

# A
gdp = pandas.read_csv('gdp-per-capita-worldbank.csv')
life_exp = pandas.read_csv('life-expectancy.csv')
total_gdp = pandas.read_csv('gross-domestic-product.csv')

one_year_gdp = gdp[(gdp['Year'] == 2020) & (gdp['Code']) & (gdp['Entity'] != "World")]

one_year_total_gdp = total_gdp[(total_gdp['Year'] == 2020) & (total_gdp['Code']) & (total_gdp['Entity'] != "World")]

one_year_life = life_exp[(life_exp['Year'] == 2020) & (life_exp['Code']) & (life_exp['Entity'] != "World")]
#print(one_year_life)

merge = one_year_gdp.merge(one_year_life, on= ['Entity', 'Code', 'Year'])
#print(merge)

total_merge = one_year_total_gdp.merge(one_year_life, on= ['Entity', 'Code', 'Year'])
#print(total_merge)

merge.plot.scatter(x= 'GDP per capita, PPP (constant 2017 international $)', y= 'Life expectancy at birth (historical)', c= 'red')
total_merge.plot.scatter(x= 'GDP (constant 2015 US$)', y= 'Life expectancy at birth (historical)', c= 'red')
plt.xlabel('GDP per capita')
plt.ylabel('Life expectancy')
plt.title(' GDP per capita vs life expectancy for 2020')


#B
mean_std = np.mean(one_year_life['Life expectancy at birth (historical)']) + np.std(one_year_life['Life expectancy at birth (historical)'])
mean = np.mean(one_year_life['Life expectancy at birth (historical)'])

test = one_year_life[one_year_life['Life expectancy at birth (historical)'] > mean_std]
test1 = list(test['Entity'])
print(test1)


#C median på gdp och medelvärde på liv

# std = np.std(one_year_total_gdp['GDP (constant 2015 US$)'])
# print(std)

mean_gdp = np.mean(one_year_total_gdp['GDP (constant 2015 US$)'])
print(mean_gdp)

median_gdp = np.median(one_year_total_gdp['GDP (constant 2015 US$)'])
print(median_gdp)

test3 = (one_year_total_gdp[(one_year_total_gdp['GDP (constant 2015 US$)'] < median_gdp)])
test4 = test3['Entity']
test5 = (one_year_life[(one_year_life['Life expectancy at birth (historical)'] > mean)])
test6 = test5['Entity']
total = test3.merge(test5)
finish = list(total['Entity'])
print(finish)

# testC = one_year_life[(one_year_life['Life expectancy at birth (historical)'] > mean) & (one_year_total_gdp['GDP (constant 2015 US$)'] < median_gdp)]
#testC = (one_year_life[one_year_life['Life expectancy at birth (historical)'] > mean]) & (one_year_total_gdp[(one_year_total_gdp['GDP (constant 2015 US$)'] < mean_gdp)])
#testC1 = list(testC['Entity'])
#print(testC1)

# D tvärt mot C

test10 = (one_year_total_gdp[(one_year_total_gdp['GDP (constant 2015 US$)'] > median_gdp)])
test11 = test10['Entity']
test12 = (one_year_life[(one_year_life['Life expectancy at birth (historical)'] < mean)])
test13 = test12['Entity']
total = test10.merge(test13)
finish = list(total['Entity'])
print(finish)

# E


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