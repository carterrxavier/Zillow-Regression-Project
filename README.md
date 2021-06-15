# Zillow-Regression-Project
## Regression Project: Home Value Prediction
 - Xavier Carter
 - 15 June 2021
# Project Summary
## Project Description
 - The Zillow Data Science team wants to be able to predict the values of single unit properties that the tax district assesses using the property data from those with a transaction during the "hot months" (in terms of real estate demand) of May-August, 2017.

## Deliverables 
- The README file that gives context to the project.
  * This readme includes key findings, take aways, and hypothesis 
- A CSV file of the State, county and tax rate of each house and graphs showing distributuions of tax rate in each county
- A Final Jupyter notebook containing well organized commented thought process and analysis.

## Project Summary
 - We must aquire, clean, prepare,  and visualize the data in order to help narrow down true drivers for House rediction.
 - Models used were Linear regression, Lasso Lars Regression, Generalized Linear Model, 3rd degree Polynomial Regression 
    * MVP includes
        * Square footage
        * Bathroom count
        * Bedroom Count 
        * House age(year built)
        * County location
       
- I chose 3rd degree polynomial model  as it suceeds with the lowest RMSE score with the highest R_2 score on the training and validating state
- Next steps would be to tweak best model hyperameters to make even more accurate.

## Data Dictionary
After aquireing and prepping the data, these are the variables used for the project

|  Variables             |  Definition                                |  Data Type             |
| :--------------------: | :----------------------------------------: | :--------------------: |
|  parcilid              |  unique identifier                         |  object   |
|  bedroomcnt            |  number of bedrooms for the                |  float    |
|  bathroomcnt            |  number of bathrooms for the property     |  float    |
|  calculatedfinishedsqaurefe|  square footage of property             |  float   |
|  taxdollarvaluecnt        |  The price of the house                 |  float    |
|  transactiondate          |  the date the house was sold            |  datetime     |
|  yearbuilt             |  The year the house was built              |  integer    |
|  regionidzip       |  The zipcode where the house is located         |  integer   |
|  fips          |  a code for the state and county the house is located in |  integer   |


## Process

##### Plan -> **Acquire ->** Prepare -> Explore -> Model -> Deliver
> - Store functions that are needed to acquire data
> - The final function will return a pandas DataFrame.
> - Import the acquire function from the acquire.py module
> - Complete some initial data summarization 
> - Plot distributions of individual variables.
___

##### Plan -> Acquire -> **Prepare ->** Explore -> Model -> Deliver
> - Store functions needed to prepare the iris data; make sure the module contains the necessary imports to run the code. The final function should do the following:
    - Split the data into train/validate/test.
    - Handle any missing values.
    - account for possible outliers
>   - Import the prepare function from the prepare.py module and use it to prepare the data in the Final Report Notebook.
___

##### Plan -> Acquire -> Prepare -> **Explore ->** Model -> Deliver
> - Answer key questions, my hypotheses, and figure out the features that can be used in a classification model to best predict the target variable, churn 
> - Run at least 2 statistical tests in data exploration.
> - Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). 
> - Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.
___

##### Plan -> Acquire -> Prepare -> Explore -> **Model ->** Deliver
> - Establish a baseline accuracy to determine if having a model is better well.
> - Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.
> - Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.
> - Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.
> - Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.
___


<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproduce My Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the wrangle.py, prepare.py, evaluate.py, Explore.py and final_report.ipynb files into your working directory
- [ ] Add your own env file to your directory. (user, password, host)
- [ ] Run the final_report.ipynb notebook
