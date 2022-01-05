from env import host, user, password
import os
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split


########################################### mySQL Connection ###########################################

def get_connection(database_name):
    '''
    This function takes in a string representing a database name for the Codeup mySQL server 
    and returns a string that can be used to open a connection to the server.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{database_name}'

########################################### Acquire Zillow Dataframe ###########################################

def get_zillow_data():
    '''
    This function reads the Zillow database from the Codeup mySQL server and  returns a dataframe.
    If a local file does not exist, this function writes data to a csv file as a backup. The local file 
    ensures that data can be accessed, in the event that you cannot talk to the mySQL database. 
    '''
    # The filename will have 2017 at the end to represent that the only data being looked at is 
    # properties from the year 2017
    if os.path.isfile('zillow2017.csv'):
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow2017.csv', index_col=0)
        
    else:
        
        # Read fresh data from database into a DataFrame
        # property land use type id is limited to 'Single Family Residential' properties.
        df =  pd.read_sql(""" SELECT bedroomcnt, 
                                     bathroomcnt, 
                                     calculatedfinishedsquarefeet, 
                                     yearbuilt, 
                                     regionidzip, 
                                     fips,
                                     taxvaluedollarcnt
                              FROM properties_2017
                              JOIN predictions_2017 USING (parcelid)
                              WHERE propertylandusetypeid = 261;""", 
                            get_connection('zillow')
                        )
        # Cache data into a csv backup
        df.to_csv('zillow2017.csv')
    
    # Renaming column names to one's I like better
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'squarefeet',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built'})   
    return df
########################################### Clean Zillow Dataframe ###########################################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def prepare_zillow (zillow):
    # Drop all data with nulls and zeros. This less than 1% of the data, so shouldn't affect modeling
    zillow = zillow.dropna()
    zillow = zillow[(zillow.bathrooms != 0) & (zillow.bedrooms != 0) & (zillow.squarefeet != 0)]
    
    # Change the data types of these columns to integers
    zillow["regionidzip"] = zillow.regionidzip.astype(int)
    zillow["bedrooms"] = zillow.bedrooms.astype(int)
    zillow["year_built"] = zillow.year_built.astype(int)
    zillow["fips"] = zillow.fips.astype(int)

    # Remove extreme outliers (there will still be a few, but our data should be less skewed)
    zillow = remove_outliers(zillow, 1.5, ['bedrooms', 'bathrooms', 'squarefeet', 'tax_value'])
    
    # Feature Engineering
    zillow['years_old'] = 2017 - zillow.year_built

    return zillow

########################################### Wrangle Zillow Dataframe ###########################################

def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    # Acquire and Prep
    zillow = prepare_zillow(get_zillow_data())
    
    # Split
    train_validate, test = train_test_split(zillow, test_size=.2, random_state= 42)
    train, validate = train_test_split(train_validate, test_size=.3, random_state= 42)
    
    return train, validate, test

########################################### Scale Zillow Dataframe ###########################################

def Min_Max_Scaler(train, validate, test):
    """
    Takes in the pre-split data and uses train to fit the scaler. The scaler is then applied to all dataframes and 
    the dataframes are returned in thier scaled form.
    """
    # 1. Create the object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # 2. Fit the object (learn the min and max value)
    scaler.fit(train[['tax_value', 'squarefeet']])

    # 3. Use the object (use the min, max to do the transformation)
    train[['tax_value', 'squarefeet']] = scaler.transform(train[['tax_value', 'squarefeet']])
    test[['tax_value', 'squarefeet']] = scaler.transform(test[['tax_value', 'squarefeet']])
    validate[['tax_value', 'squarefeet']] = scaler.transform(validate[['tax_value', 'squarefeet']])
    
    return train, validate, test
