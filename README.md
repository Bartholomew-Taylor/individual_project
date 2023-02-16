# Snow Crab Individual Project
## Overview: 
#### Analysis and Modeling from Alaskan Snow Crab Data
#### Data pulled from Kaggle (https://www.kaggle.com/datasets/mattop/snowcrab), 1975 - 2018
#### Origin: NOAA fisheries data

## Goals
#### Provide Guiance to fisheries and conservation efforts via data analysis and model production

## Project Plan
 * Acquisition
   * dataset on Snow Crab hauls pulled from Kaggle
 * Preparation 
   * checked for nulls (there were none)
   * outliers removed
   * data split into Train, Validate, and Test sets
   * speparate data split created for time
 * Exploration
   * vizualizaitons 
   * bivariate stats analysis
   * see below
* Feature Engineering
  * dummies created
  * clustering experimented with (geospatial data)
  * scaling for modeling (geospatial data, temperatures, depth)
* Modeling
  * Time Series 
  * Regression
##  Exploration
### Exploration Question 1:
* Does CPUE Size exhibit a seasonality trend?
  -  𝐻0
   : There is no notable seasonal trend in snow crab CPUE.
  -  𝐻𝑎
   : There is a quantifiable trend in seasonality of CPUE.
### Exploration Question 2:
* Is there a correlation between CPUE and Seafloor Temperature?
  -  𝐻0
   : There is no correlation between CPUE and Seafloor Temperature.
  -  𝐻𝑎
   : There is correlation between CPUE and Seafloor Temperature.
### Exploration Question 3:
* Is there a correlation between CPUE and Surface Temperature?
-  𝐻0
 : There is no correlation between CPUE and Surface Temperature.
-  𝐻𝑎
 : There is correlation between CPUE and Surface Temperature.
 ### Exploration Question 4:
*Is there a correlation between depth of catch and CPUE?
-  𝐻0
 : There is no correlation between depth of catch and CPUE.
-  𝐻𝑎
 : There is correlation between depth of catch and CPUE.
#### * see wrangle.py and prepare.py modules for notations on exploration and prep functions

## Modeling
  * Models used for Train and Validation
  * Holts Seasonal Trend (best selected)
  
# Data Dictionary
