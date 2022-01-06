<div align="center">

<img src="Images/Zillow_Logo.jpeg" alt="Zillow Logo" title="Zillow Logo" width="1000" height="200" align="center"/>
    
# README

### by Joann Balraj and Jeanette Schulz 
### January 07, 2022

</div align="center">
    
<hr style="border:2px solid blue"> </hr>

# Project Description
### What Is Zillow?
> "As the most-visited real estate website in the United States, Zillow and its affiliates offer customers an on-demand experience for selling, buying, renting and financing with transparency and nearly seamless end-to-end service. Zillow Offers buys and sells homes directly in dozens of markets across the country, allowing sellers control over their timeline. Zillow Home Loans, our affiliate lender, provides our customers with an easy option to get pre-approved and secure financing for their next home purchase. Zillow recently launched Zillow Homes, Inc., a licensed brokerage entity, to streamline Zillow Offers transactions."  [zillow.com](https://www.zillow.com/z/corp/about/)
### What is this about?
As the most-visited real estate website in the United States, Zillow and its affiliates offer customers an on-demand experience for selling, buying, renting and financing with transparency and nearly seamless end-to-end service. This is all in thanks to Zillow's throughly tested model that helps them predict the value of almost any house. However as we know, the housing market is constantly fluctuating and changing. Thus, Zillow's model should also be constantly changing and improved to ensure the best, most up to date home value approximations. With a focus in utilizing clustering methodologies, we will analye the data provided from 2017 transactions to see if any new features can be engineered. We will then develope models with these features to predict the Zestimate residual error, and deliver both recommendations for what worked and didn't work in the form of a video presentation.

<hr style="border:2px solid blue"> </hr>

# Project Goal
Zillows existing Zestimate model is pretty good but not perfect, which means modeling errors can be a very powerful way to find areas to improve that existing model. The goal of this project is to create a model to predict these Zestimate errors, and deliver our findings in a video presentation. 

<hr style="border:2px solid blue"> </hr>

# Project Planning
## Plan -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

<b>Planning:</b>  
- Create a README file (check!)
- Ensure my wrangle.py modules are well documents and functional

<b>Acquisition </b>  
- Obtain access to Codeup mySQL database via a `env.py` file
- Gather Zillow data from Codeup mySQL database using a SQL query
- Create a `wrangle.py` file to make future acquisition easier

<b>Preparation</b>  
- Create a `workbook.ipynb` file to put all work in 
- Clean aquired Zillow data:
    - remove missing values, 
    - inspect data integrity issues 
    - ensure columns are proper data type
    - reduce outliers
- Create a scale function for future modeling
- Create split function for future modeling
- Add all new functions to `wrangle.py`

<b>Exploration and Pre-processing</b>  
- Explore the target variable using visualization and statistical testing
- Use clustering methodologies to explore the data and attempt at least 3 combinations of features. 
- Summarize takeaways and conclusions

<b>Modeling</b>  
- Split data appropriately 
- Establish and evaluate baseline model
- Create at least 4 different models and compare their performance. 


<b>Deliver</b>  
- Deliver a 5 minute video presentation via a jupyter notebook walkthrough 
- Answer questions about my code, process, and findings



<hr style="border:2px solid blue"> </hr>

# Data Dictionary

| Feature                       | Datatype                  | Description                                                        |
|:------------------------------|:--------------------------|:-------------------------------------------------------------------|
| bathroomcnt                   | 52442 non-null  float64   | Number of bathrooms in home including fractional bathrooms
| bedroomcnt                    | 52442 non-null  float64   | Number of bedrooms in home 
| calculatedfinishedsquarefeet  | 52360 non-null  float64   | Calculated total finished living area of the home 
| fips                          | 52442 non-null  float64   | Federal Information Processing Standard code 
| yearbuilt                     | 52326 non-null  float64   | The Year the principal residence was built 
| taxvaluedollarcnt             | 52441 non-null  float64   | The total tax assessed value of the parcel
| regionidzip                   | 52416 non-null  float64   | Zip code in which the property is located
| logerror	                    | 52416 non-null  float64   | The log error of actual vs predicted home price

<hr style="border:2px solid blue"> </hr>

# Steps to Reproduce

You will need your own env file with database credentials along with all the necessary files listed below to run the `"Final Report"` notebook.

 1. Read this README.md (check!)
 2. Download at the `wrangle.py` and `Final_Report.ipynb` file into your working directory
 3. Create a `.gitignore` for your `env.py` file
 4. Add your own `env.py` file to your directory with username, password, and host address. 
 5. Run the `Final_Report.ipynb` in a jupyter notebook

<hr style="border:2px solid blue"> </hr>


# Initial Questions for the Project

1. Why do some properties have a much higher value than others when they are located so close to each other?  
2. Are houses with a high bedroom count but a low bathroom count, less valueable?
3. Are houses of a certain year built more valuable than others?
4. Are there certain countys that are harder to predict taxvaluedollarcnt for?
5. Is there a linear correlation between square footage and tax value?
6. Is there a linear correlation between number of bathrooms and tax value?
7. Is there a linear correlation between number of bedrooms and tax value?
