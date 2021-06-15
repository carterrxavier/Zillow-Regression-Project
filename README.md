# Zillow-Regression-Project
## Regression Project: Prediction Home Value
 - Xavier Carter
 - 15 June 2021
# Project Summary
## Project Description
 - The Zillow Data Science team wants to be able to predict the values of single unit properties that the tax district assesses using the property data from those with a transaction during the "hot months" (in terms of real estate demand) of May-August, 2017.

Additionally, information is needed outside of the model. Because property taxes are assessed at the county level, the team would like to know:

What states and counties these are located in?
What is the distribution of tax rates for each county?
Goals
Identify the drivers(features) for predicting the property values using the appraised value.
Document the process and analysis through the data science pipeline.
Construct a regression model for predicting home values that will do better than a baseline model.
Project Deliverables
Presentation with slides that summarizes finding about the drivers of property value.
Jupyter Notebook report detailing the process through the pipeline.
Acquire and Prepare files for recreating the process.
README file that documents the project planning with instructions on how to recreate.
Project Planning
What is the definition of a single unit property?

"A housing unit is a single unit within a larger structure that can be used by an individual or household to eat, sleep, and live. The unit can be in any type of residence, such as a house, apartment, or mobile home, and may also be a single unit in a group of rooms." -www.investopedia.com
For the purposes of this project, decided to use properties described as Residential General(260), Single Family Residential(261), Rural Residence(262), Mobile Home(263), and Townhouse(264) from the Zillow database.
Using the above codes and the dates specified produced 28,185 records for initial analysis.
What features are the best for predicting a property's value?

Initial focus on number of bathrooms, number of bedrooms, square feet and age.
What are the counties and tax rates for the transactions in the project?

Use the FIPS code provide in the Zillow database to determine the counties included in the project.
The Federal Information Processing Standard Publication 6-4 is a five-digit Federal Information Processing Standards code which uniquely identified counties and county equivalents in the United States, certain U.S. possessions, and certain freely associated states.
Project Context
The dataset came from the Zillow database.
Data Dictionary
The Zillow database contains many tables. The properties_2017 and predictions_2017 tables were joined together to pull the data into a single pandas DataFrame for this project.

After preparing the data, the remaining features and values are listed below:

Feature	Description	Data Type
parcelid	Unique identifier assigned to each property, set as index	int64
bathrooms	Number of bathrooms	int64
bedrooms	Number of bedrooms	int64
square_feet	Square feet of the property	int64
county_code	FIPS county code for location of property	int64
age	Age of the property	int64
appraised_value	Appraised value of the property	int64
taxes	Tax amount	float64
Initial Hypotheses
Hypothesis 1 - Rejected the Null Hypothesis; the average appraised value of properties with 1600 sq.ft or more appears to be higher than the average appraised value of properties with 1600 sq.ft. or less.
alpha = .05
$H_0$: The average appraised value of properties with 1600 sq.ft. or more is equal to the average appraised value of properties with 1600 sq.ft or less.
$H_a$: The average appraised value of properties with 1600 sq.ft. or more is higher than the average appraised value of properties with 1600 sq.ft or less.
Hypothesis 2 - Rejected the Null Hypothesis; the number of bathrooms appears to have an affect on the appraised value.
alpha = .05
$H_0$: The number of bathrooms has no affect on the appraised value (independent)
$H_a$: The number of bathrooms does have an affect on appraised value (dependent)
Key Findings and Takeaways
Created 3 regression models - OLS (LinearRegression), LassoLars, and TweedieRegressor (GLM) - and tested them with a list of 4 features - 'bathrooms', 'bedrooms', 'square_feet', and 'age' to predict the target value of 'appraised_value'.
Each model was better at predicting the 'appraised_value' than the baseline.
Chose the LassoLars model as the best model with lowest R^2 value. This model outperformed the baseline, so it has value.
Initial exploration and statistical testing revealed that the selected features produced well-fit models, and with more time exploring additional features and/or adjusting hyperparameters could improve the results.
Additional Deliverables
Properties in the project were located in 3 counties in California and were distrubited as follows:
Los Angeles: 15842 (mean tax rate: 1.43%)
Orange: 6807 (mean tax rate: 1.21%)
Ventura: 2239 (mean tax rate: 1.19%)
Reproduce My Project
You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook.

 Read this README.md
 Download the aquire.py, prepare.py and zillow_regression_final_notebook.ipynb files into your working directory
 Add your own env file to your directory. (username, password, host)
 Run the zillow_regression_final_notebook.ipynb notebook
