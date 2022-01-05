############################################# Introduction #############################################

# This WRANGLE.py file is for the Codeup Zillow Project utilizing Clustering methodologies.

# These functions are the combined works from Codeup cohorts Joann Balraj and Jeanette Schulz
# and are here to create a cleaner work enviroment in jupyter notebook for future presenting. 

############################################### Imports ###############################################

from env import host, user, password
import os
import pandas as pd
import sklearn.preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

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
                                     taxvaluedollarcnt,
                                     logerror
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
                              'yearbuilt':'year_built',
                              'logerror': 'logerror'})   
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


########################################### Null Finders ###########################################

def nulls_by_col(df):
    """
    Renders a dataframe that shows the nulls for every column
    """
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    """
    Renders a dataframe that shows the nulls for every row
    """
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'customer_id': 'num_rows'}).reset_index()
    return rows_missing

########################################### Explore Zillow Dataframe ###########################################


def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe)
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3).to_markdown())
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe().to_markdown())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('============================================')



### functions to create clusters and scatter-plot: ###

def create_cluster(df, X, k):
    
    """ 
    Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    # the scaler and kmeans object and unscaled centroids as a dataframe
    """
    
    scaler = MinMaxScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])

    ## sklearn implementation of KMeans

    #define the thing
    kmeans = KMeans(n_clusters = k, random_state = 321)

    # fit the thing
    kmeans.fit(X_scaled)

    # Use (predict using) the thing
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)

    #Create centroids of clusters
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)

    return df, X_scaled, scaler, kmeans, centroids


def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler):
    
    """ 
    Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot
    """
    
    plt.figure(figsize=(14, 9))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')
    plt.title('Visualizing Clusters')


def make_metric_df(y, y_pred, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
            'model': model_name, 
            'RMSE_validate': mean_squared_error(
                y,
                y_pred) ** .5,
            'r^2_validate': explained_variance_score(
                y,
                y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
        {
            'model': model_name, 
            'RMSE_validate': mean_squared_error(
                y,
                y_pred) ** .5,
            'r^2_validate': explained_variance_score(
                y,
                y_pred)
        }, ignore_index=True)

    

def inertia_plot(X):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')          

def plot_actual(train, grp_by_var, x, y):
    plt.figure(figsize=(14, 9))
    sns.scatterplot(x = x, y = y, data = train, hue = grp_by_var)

    #for cluster, subset in train.groupby(grp_by_var):
       # x = x
        #y = y
        #print(x,y)
       # plt.scatter(subset.x,subset.y, label=str(cluster), alpha=.6)
     #centroids.plot.scatter(y=y_var, x=x_var, c='black', marker='x', s=1000, ax=plt.gca(), label='centroid')

    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Visualizing Actual Data - Not Clusters')
    plt.show()      



def calc_cluster_mean(train, cluster):
    """
    Looping through all the regions in the cluster to calculate it's mean and compare it
    to the overall log error mean to see if it may be of significance
    """


    count = train.cluster.max()

    print (count)

    logerror_mean =  train.logerror.mean()

    i = 0
    while i <= count:

            region_mean = train[train.cluster== i].logerror.mean()
        
            difference = abs(logerror_mean - region_mean)
    
            region_mean_df = pd.DataFrame(data=[
                {
                    'region': i,
                    'mean':region_mean,
                    'difference': difference
                }
                ])
        
            print(region_mean_df)
            i += 1