# Investigation of Water Potability

## Project Motivation

According to the *Global Burden of Disease* study 1.2 million people died prematurely as a result unsafe drinking water. This number is more than three times the amount of homicides globally and equivalent to the total amount of road deaths (1). More specifically, countries which are poorest are most vulnerable to illness from unsafe drinking water. 

![Image](Images/death-tolls.png)

It's clear that unsafe drinking water is a problem for global health. In this project, I will be exploring the inequalities of countries when it comes to water inaccesibility and also using Random Forests and Decicion Trees to predict Water Potability. 


## Predicting Water Potability with Decision Trees and Random Forests

**Where was data from**?

After reading in the data and importing the necessary libraries, this is what our data looks like. 


![Image](Images/water-df.png)


The data consists of 3,276 water samples and contains 9 numeric values. 

    1. pH Value 
    2. Hardness
    3. Solids
    4. Chloramines
    5. Sulfate
    6. Conductivity
    7. Organic_carbon
    8. Trihalomethanes
    9. Turbidity
    10. Potability

Potability value of 1 means potable (drinkable)and 0 not potable (undrinkable). 

Let's check the breakdown of samples that are potable vs unpotable. 

![Image](Images/potability-breakdown.png)


Checking for a correlation between different variables. 

![Image](Images/pairplot.png)

### Data: Source? 

### Works Cited
1. https://ourworldindata.org/water-access
