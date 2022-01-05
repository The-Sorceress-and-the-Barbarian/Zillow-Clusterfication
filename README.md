<div align="center">

<img src="Images/zillow_logo.png" alt="Zillow Logo" title="Zillow Logo" width="300" height="100" align="center"/>
    
# README

### by Jeanette Schulz 2021-12-13

</div align="center">
    
<hr style="border:2px solid blue"> </hr>

# Project Goal
Zillow has a model that is designed to predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction during 2017. The goal of this project is to look for insights that can help possibly improve this model, and make recommendations on how to improve it. 



<hr style="border:2px solid blue"> </hr>

# Project Description
As the most-visited real estate website in the United States, Zillow and its affiliates offer customers an on-demand experience for selling, buying, renting and financing with transparency and nearly seamless end-to-end service. This is all in thanks to Zillow's throughly tested model that helps them predict the value of almost any house. However as we know, the housing market is constantly fluctuating and changing. Thus, Zillow's model should also be constantly changing and improved to ensure the best, most up to date home value approximations. We will analye the data provided from 2017 transactions to see if any new features can be engineered, develope models with these features to compare to zillows current model, and deliver both recommendations for what worked and didn't work in the form of a presentation.


<hr style="border:2px solid blue"> </hr>

# Project Planning
## Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

<b>Planning:</b>  
- Create a README file (check!)
- Ensure my wrangle.py modules are well documents and functional

<b>Acquisition </b>  
- Obtain Zillow data from Codeup mySQL database via wrangle.py

<b>Preparation</b>  
- Clean Zillow data from Codeup mySQL database via wrangle.py

<b>Exploration and Pre-processing</b>  
- Ask and answer statistical questions about the Zillow data
- Visually represent findings with charts

<b>Modeling</b>  
- Split data appropriately 
- Use knowledge acquired from statistical questions to help choose a model
- Create a predictions csv file from the best model

<b>Deliver</b>  
- Deliver a 5 minute presentation via a jupyter notebook walkthrough 
- Answer questions about my code, process, and findings



<hr style="border:2px solid blue"> </hr>

# Data Dictionary

| Feature                       | Datatype                  | Description                                                        |
|:------------------------------|:--------------------------|:-------------------------------------------------------------------|
| bathroomcnt                   | 52442 non-null  float64 | Number of bathrooms in home including fractional bathrooms
| bedroomcnt                    | 52442 non-null  float64 | Number of bedrooms in home 
| calculatedfinishedsquarefeet  | 52360 non-null  float64 | Calculated total finished living area of the home 
| fips                          | 52442 non-null  float64 | Federal Information Processing Standard code 
| yearbuilt                     | 52326 non-null  float64 | The Year the principal residence was built 
| taxvaluedollarcnt             | 52441 non-null  float64 | The total tax assessed value of the parcel
| regionidzip                   | 52416 non-null  float64 | Zip code in which the property is located

<hr style="border:2px solid blue"> </hr>

# Steps to Reproduce

You will need your own env file with database credentials along with all the necessary files listed below to run the `"Final Report"` notebook.

 1. Read this README.md
 2. Download at the aquire.py and Final Report.ipynb file into your working directory
 3. Create a .gitignore for your .env file
 4. Add your own env file to your directory with username, password, and host address. 
 5. Run the final_report.ipynb notebook

<hr style="border:2px solid blue"> </hr>


# Initial Questions for the Project

1. Why do some properties have a much higher value than others when they are located so close to each other?  
2. Are houses with a high bedroom count but a low bathroom count, less valueable?
3. Are houses of a certain year built more valuable than others?
4. Are there certain countys that are harder to predict taxvaluedollarcnt for?