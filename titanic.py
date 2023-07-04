#
#   Written by:  Mark W Kiehl
#   http://mechatronicsolutionsllc.com/
#   http://www.savvysolutions.info/savvycodesolutions/
#
#   Read the Titanic dataset and perform Data Science on it using Pandas and SkyLearn.
#
#   Get the Titanic data set: https://www.kaggle.com/datasets/yasserh/titanic-dataset
#
#   This data science example was inspired by and many portions derived from the following
#   two articles:  
#
#       "Predicting the Survival of Titanic Passengers" by Niklas Donges
#       https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
#       See model under 'model_building_doges()'
#       
#       "Building a Machine Learning Model Step By Step With the Titanic Dataset" by Taha Bilal Uyar
#       https://medium.com/swlh/building-a-machine-learning-model-step-by-step-with-the-titanic-dataset-e3462d849387
#       See model under 'model_building_tbu()'
#
#   The data pre-processing employed is different from what each author did, so the results in
#   each article are not exact, but they are very close. 
#   By following the two articles and reviewing and executing this code, you can learn a lot
#   about data science from these authors. 


# About the Titanic dataset:
"""
SibSp - # of siblings / spouses aboard
Parch - # of parents / children aboard
Embarked - where the traveler mounted from: Southampton, Cherbourg, and Queenstown
Pclass - passenger class (1st, 2nd, 3rd)
Cabin - a cabin identifier where the first character identifies the deck. From top to bottom: U, P, A, B..G, O, T (U = Boat Deck, A = Promenade, O = Orlop. T = Tank Top) |
"""

#Python performance timer
from pickle import NONE
import time
t_start_sec = time.perf_counter()


# pip install memory-profiler
# Use the package memory-profiler to measure memory consumption
# The peak memory is the difference between the starting value of the “Mem usage” column, and the highest value (also known as the “high watermark”).
# IMPORTANT:  See how / where @profile is inserted before a function later in the script.
from memory_profiler import profile


# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import json

# Algorithms
# pip install scikit-learn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


def scale_df(df):
    """
        If the Series df is Gaussian (normally distributed), then it is 
        standardized (values between -1 and 1, with a mean of 0).

        If the Series df is NOT Gaussian (normally distributed) and tests
        positive as log-normal distribution, then it will be Normalized.

        If the Series df is NOT Gaussian/normally distributed and is not
        a log-normal distribution, then it will be Normalized
        (also known as min-max scaling).

        Returns the modified dataframe series.
    """
    # Scaling can improve the convergence speed of various ML algorithms, especially with data sets that have a large variation. 
    # The Normalization method scales values between 0 and 1.
    # The Standardization method scales values between -1 and 1, with a mean of 0.  
    # If the distrubution is normal, then it should be standardized, otherwise normalized.
    # Normalization is a better option when we are not necessarily dealing with a normally distirbuted 
    # data and you are not worried about standardization along the variance axes (i.e. image processing 
    # or neural network expecting values between 0–1).
    # https://medium.com/mlearning-ai/feature-scaling-normalization-or-standardization-74d73ec90366

    from scipy.stats import shapiro
    # Check numerically if a numeric column is normally distributed using the Shapiro-Wilk test
    from scipy.stats import normaltest
    stat, p1 = shapiro(df)

    # Check numerically if a numeric column is normally distributed using the K-squiared test
    from scipy.stats import shapiro
    stat, p2 = shapiro(df)

    # Check numerically if a numeric column is normally distributed using the K-squiared test
    # (use for a log-normal distribution)
    from scipy.stats import kstest
    #perform Kolmogorov-Smirnov test for normality
    stat, pvalue = kstest(df, 'norm')
    
    if p1 > 0.05 and p2 > 0.05 and pvalue <= 0.05:
        print('Series "', df.name,'" is probably Gaussian (normally distributed) and will be scaled using Standardization.')
        df = (df - df.mean() ) / df.std()
    elif pvalue > 0.05:
        print('Series "', df.name, '" is a Log-Normal distribution and will be scaled using Standardization.')
        df = (df - df.mean() ) / df.std()
    else:
        print('Series "', df.name,'" is NOT Gaussian (normally distributed) or Log-Normal distribution, so it will be scaled using Normalization (min-max scaling).')
        #print('stat=%.3f, p1=%0.3f' % (stat,p1)) 
        #print('stat=%.3f, p2=%0.3f' % (stat,p2)) 
        #print('stat=%.3f, pvalue=%0.3f' % (stat,pvalue)) 
        # Min-Max Normalization:
        # Formula: df[‘column’] = df[‘column’] – df[‘column’].min()
        # Formula: df[‘column’] = df[‘column’] / df[‘column’].max()
        df = df - df.min()
        df = df / df.max()
    return df


def show_df_contents(df):
    # A nice online view of the raw data:  https://github.com/jorisvandenbossche/pandas-tutorial/blob/master/data/titanic.csv
    
    # print general information about the DataFrame contents
    print(df.info())
    """
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   PassengerId  891 non-null    int64
     1   Survived     891 non-null    int64
     2   Pclass       891 non-null    int64
     3   Name         891 non-null    object
     4   Sex          891 non-null    object
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64
     7   Parch        891 non-null    int64
     8   Ticket       891 non-null    object
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object
     11  Embarked     889 non-null    object
     """
    
    # print out statistics about the DataFrame
    #print(df.describe())
    """
            PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
    count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
    mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
    std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
    min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
    25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
    50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
    75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
    max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
    """
    # 38% of the passengers in this data set survived.
    
    # print the first few rows of the DataFrame
    print(df.head(),'\n')
    """
    """
    """
           PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
        0            1         0       3  ...   7.2500   NaN         S
        1            2         1       1  ...  71.2833   C85         C
        2            3         1       3  ...   7.9250   NaN         S
        3            4         1       1  ...  53.1000  C123         S
        4            5         0       3  ...   8.0500   NaN         S
    """
    
    # print all of the DataFrame contents for columns 'Survived','Pclass','Sex','Age','Fare'
    """
    columns = ['Survived','Pclass','Sex','Age','Fare']
    print(df[columns].to_string())  # print all of the data for the columns specified by columns
    """

    # print the DataFrame's shape
    """
    print(df.shape,'\n')
    """
    """
    [5 rows x 12 columns]
    """
    
    # print the DataFrame's data types
    """
    print(df.dtypes,'\n')
    """
    """
        891, 12)
    """
    
    # Get a list of the column names
    print(df.columns.values,'\n')
    """
    ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']
    """
    


def data_cleaning(df):
    # Determine what data is missing
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent_1 = df.isnull().sum()/df.isnull().count()*100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
    print('Missing data: \n', missing_data.head(5))
    """
    """
                     Total     %
        Cabin          687  77.1
        Age            177  19.9
        Embarked         2   0.2
        PassengerId      0   0.0
        Survived         0   0.0
    """
    # 77% of the Cabin data is missing, and 20% of Age is missing

    # Convert column 'Sex' values 'male' and 'female' to 0 and 1
    # *** This does the replacement incorrectly:  df['Sex'].replace(['male', 'female'],[1, 0], inplace=True)
    df['Sex_MF01'] = pd.factorize(df['Sex'])[0]      # The [0] returns the codes only
    """
    print(df[['Survived','Sex', 'Sex_MF01']])  # print the first few and last rows in the DataFrame for the columns specified by columns
        Survived     Sex  Sex_MF01
    0           0    male        0
    1           1  female        1
    2           1  female        1
    3           1  female        1
    4           0    male        0
    ..        ...     ...      ...
    886         0    male        0
    887         1  female        1
    888         0  female        1
    889         1    male        0
    890         0    male        0
    """
    #df['Sex'] = df['Sex_num']
    #df.drop(columns=['Sex_num'], inplace=True)
    #print(df[['Survived','Sex']])  # print the first few and last rows in the DataFrame for the columns specified by columns

    # SibSp is the number of siblings / spouses aboard, and Parch is the number of parents / children aboard.
    # Combine SibSp and Parch into a new column 'relatives' as the total number of relatives a person has on the Titanic. 
    df['Relatives'] = df['SibSp'] + df['Parch']
    """
    columns = ['PassengerId','Relatives','SibSp','Parch']
    print(df[columns].to_string())  # print all of the data for the columns specified by columns
    """
    
    # Extract the deck level information from the 'Cabin' column' and
    # convert it to a numeric value.  
    #
    # On the Titanic, there were 10 decks in total, from top to bottom passenger decks were A .. G.  In the data set, only A .. G exist.
    levels = []     # create a new empty list
    for level in df['Cabin']:
        if isinstance(level, float):
            levels.append(level)    # Retains NaN
        else:
            # Convert to a string and then get the first character.  Ex. 'C123' -> 'C'
            s = str(level)
            levels.append(s[0])
    # Create a new column 'Deck' in the DataFrame df
    df['Deck'] = levels
    # Remove the one value incorrectly assigned as 'T' and set to NaN
    df.loc[ df['Deck'] == 'T', 'Deck'] = float('NaN')
    # Print out and compare the 'Cabin' and 'Deck' columns to verify the conversion was valid
    """
    columns = ['Cabin','Deck']
    print(df[columns].to_string()) 
    """
    #print(df['Deck'].value_counts())    #print a summary of the contents of column 'Deck'
    """
        C       59
        B       47
        D       33
        E       32
        A       15
        F       13
        G        4
    """
    # In column 'Deck', replace NaN with 0 and 'A' .. 'G' with 1 .. 7
    df['Deck'].replace([float("NaN"),'A','B','C','D','E','F','G'],[0,1,2,3,4,5,6,7], inplace=True)
    # Convert column 'Deck' to type integer
    df['Deck'] = df['Deck'].astype('int')
    # Print out and compare the 'Cabin' and 'Deck' columns to verify the conversion was valid
    """
    columns = ['Cabin','Deck']
    print(df[columns].to_string()) 
    """
    """
    Results:
        0    688
        3     59
        2     47
        4     33
        5     32
        1     15
        6     13
        7      4
    """
    #print(df['Deck'].value_counts())    #print a summary of the contents of column 'Deck'
    # The DataFrame column 'Cabin' is no longer needed, so delete it
    df.drop(columns = ['Cabin'], inplace=True)
    
    return df



def e_d_a(df):
    # Exploratory Data Analysis (EDA)
    
    # Plot the distribution of 'Age' to make sure the values are normally distributed (since values are missing)
    """
    sns.distplot(df['Age'])
    plt.show()
    """
    # Results:  Looks normally distributed
    
    # Statistically evaluate if the data in 'Age' is normally distributed using the Shapiro-Wilk test
    """
    stat, p = shapiro(df['Age'])    # Requires: from scipy.stats import shapiro
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    if p > 0.05:
        print('Column Age looks Gaussian (fail to reject H0)\n')
    else:
        print('Column Age does not look Gaussian (reject H0)\n')
    
    # Results: Column Age looks Gaussian (fail to reject H0)
    """
    # Use data.corr() to calculate a correlation matrix for the dataset where the columns contain numeric data
    """
    columns = ['Survived','Pclass','Age','Sex']
    corr_mat = df[columns].corr()
    print(corr_mat)
    """
    """
    Results:
              Survived    Pclass       Age       Sex
    Survived  1.000000 -0.338481 -0.077221 -0.543351
    Pclass   -0.338481  1.000000 -0.369226  0.131900
    Age      -0.077221 -0.369226  1.000000  0.093254
    Sex      -0.543351  0.131900  0.093254  1.000000
    """
    # Create a heatmap to visualize the correlation between multiple numeric columns
    sns.heatmap(df[['Survived','Pclass','Age','Sex']].corr(), cmap='RdYlGn', annot=True)
    plt.show()
    
    # Calculate the count and % of missing values
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent_1 = df.isnull().sum()/df.isnull().count()*100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
    print(missing_data.head(5))
    """
    
    # Determine if a passenger having relatives aboard affects their survival.
    #print(df['Relatives'].value_counts())   # Pandas value_counts() function returns object containing counts of unique values. 
    """
    count = df['Relatives'].value_counts()        #Pandas series
    tot = count.sum()     #891
    pct = round(count / tot * 100,1)
    df_count = count.to_frame().reset_index()
    df_count.rename(columns = {'Relatives':'Count'}, inplace = True)
    df_count.drop(columns=['index'], inplace = True)
    df_count.rename(columns = {'Relatives':'Count'}, inplace = True)
    df_pct = pct.to_frame().reset_index()
    df_pct.drop(columns=['index'], inplace = True)
    df_pct.rename(columns = {'Relatives':'Percent'}, inplace = True)
    df_rel = pd.concat([df_count, df_pct], axis=1)
    print(df_rel)
    """
    """
          Count Percent
        0   537    60.3
        1   161    18.1
        2   102    11.4
        3    29     3.3
        4    22     2.5
        5    15     1.7
        6    12     1.3
        7     7     0.8
        8     6     0.7
      60% of the passengers didn't have any relatives aboard. 
      33% of the passengers had 1 to 3 relatives aboard. 
    """
    """
    sns.lineplot(x='Relatives', y='Survived', data=df, err_style="bars")
    plt.show()
    """
    # Results: Having 1 to 3 relatives aboard improved your probability of survival.
    
    # Using the new 'Deck' column, determine if a relationship exists between the probability of survival and the deck level
    """
    sns.lineplot(x='Deck', y='Survived', data=df, err_style="bars")
    plt.show()
    """
    
    
    # Visualize the relationship between passenger class (Pclass) and who survived
    """
    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.show()
    """
    # Results:  Passenger class (Pclass) definitely contributed to a person's chance of survival.
    
    
    # Visualize the relationship of 'Age' and 'Pclass' to 'Survived'
    """
    grid = sns.FacetGrid(df, col='Survived', row='Pclass')
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend();
    plt.show()
    """
    # Results:  Passengers of all ages in Class 1 have a better chance of survival.  Class 3 passengers are unlikely to survive.  
    
    
    
    # Use data.corr() to calculate a correlation matrix for the dataset where the columns contain numeric data
    """
    columns = ['Survived','Deck','Age','Sex','Pclass','Relatives']
    corr_mat = df[columns].corr()
    print(corr_mat)
    """
    """
                   Survived      Deck       Age       Sex    Pclass  Relatives
        Survived   1.000000  0.295812 -0.077221 -0.543351 -0.338481   0.016639
        Deck       0.295812  1.000000  0.174780 -0.149405 -0.568401   0.000112
        Age       -0.077221  0.174780  1.000000  0.093254 -0.369226  -0.301914
        Sex       -0.543351 -0.149405  0.093254  1.000000  0.131900  -0.200988
        Pclass    -0.338481 -0.568401 -0.369226  0.131900  1.000000   0.065997
        Relatives  0.016639  0.000112 -0.301914 -0.200988  0.065997   1.000000
    """
    
    # visualize the distribution of a categorical column    
    #df['Sex'].value_counts().plot(kind='bar')
    #plt.show()
    
    # create a distplot to visualize the distribution of a numeric column
    #sns.distplot(df['Age'],title="Age")
    #plt.show()
    

    """
    # A better version of the prior..
    # Visualize a categorial column as a histogram with the percentage of the total shown
    plt.hist(df['Sex'], weights=np.ones(len(df['Sex'])) / len(df['Sex']))
    plt.show()
    """
    
    # visualize the distribution of a numeric column
    # The 'Age' data is missing values.  Check to see if it is normally distributed. 
    """
    plt.hist(df['Age'])
    plt.ylabel('Age')
    plt.show()
    """
    
    # calculate basic statistics for a numeric column
    #print('Statistics for Fare:')
    #print(df['Fare'].describe())
    """
    Results:
        count    891.000000
        mean      32.204208
        std       49.693429
        min        0.000000
        25%        7.910400
        50%       14.454200
        75%       31.000000
        max      512.329200
    """
    
    # calculate the correlation between two numeric columns
    #print('\nCorrelation between Fare and Survived:')
    #print(df['Fare'].corr(df['Survived']))
    # Result: 0.2573065223849622
    
    # calculate the correlation between two numeric columns
    #print('\nCorrelation between Pclass and Survived:')
    #print(df['Pclass'].corr(df['Survived']))
    # Result: -0.3384810359610148

    
    #rows = len(df.index)
    #print('# rows: {}'.format(rows))

    
    # Embarked - where the traveler mounted from: Southampton, Cherbourg, and Queenstown
    # Create a multi-plot grid for plotting conditional relationships.
    # NOTE: Must pass 'Sex' not 'Sex_MF01' because numerical values will yield incorrect results.
    """
    FacetGrid = sns.FacetGrid(df, row='Embarked', aspect=1.6)
    FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
    FacetGrid.add_legend()
    plt.show()
    """
    # Based on the plots, Pclass seems to be correlated with survival. 
    # Embarked seems to be correlated with survival, depending on the gender.
    # Women on port Q and on port S have a higher chance of survival, but low chance for port C.
    # Men have a high survival probability if they are on port C, but a low probability if they are on port Q or S.

    # The FacetGrid suggested Pclass was correlated with survival.
    """
    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.show()
    """
    # The plot shows nearly a linear relationship between Pclass and survival.

    # Looking more deeply into Pclass..
    """
    grid = sns.FacetGrid(df, col='Survived', row='Pclass', aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()
    plt.show()
    """
    # High probability that a person in 3rd class between the ages of 17 and 40 will not survive.

    # Number of relatives
    """
    sns.pointplot(x='Relatives', y='Survived', data=df)
    plt.show()
    """
    # Plot shows high probabilty of survival with 1 to 3 relatives.


def data_preprocessing(df):

    # Replace the two missing values in Embarked with the most common value (mode)
    mode = df['Embarked'].mode()
    mode = mode.iat[0]      # Convert to a string
    # OR: mode = df['Embarked'].mode().iat[0]
    print('Replacing ', df['Embarked'].isnull().sum(), ' missing values in column Embarked with the mode of ', mode)
    # Replacing  2  missing values in column Embarked with the mode of  S
    df['Embarked'].fillna(mode, inplace=True)
    #print(df['Embarked'].isnull().sum(), ' missing values in column Embarked')  # 0  missing values in column Embarked
    """
    print(df['Embarked'])
    0      S
    1      C
    2      S
    3      S
    4      S
        ..
    886    S
    887    S
    888    S
    889    C
    890    Q
    """
    # Print the count of distinct observations (unique values) for column 'cat' (ignores nan values)
    """
    print('Unique values for col cat: \n', df['Embarked'].value_counts())
    S    646
    C    168
    Q     77
    """
    # Convert the Embarked column codes into numbers where: S = 0, C = 1, Q = 2
    ports = {"S": 0, "C": 1, "Q": 2}
    df['Embarked'] = df['Embarked'].map(ports)

    
    # Age missing values
    # 20% of Age is missing values (177 out of 891).
    # The Age column is data type float64.
    # From the EDA, the Age values are known to be normally distributed.
    # Replace the missing Age values with random values generated based
    # on the mean age and the standard deviation of the age columnn.
    mean = df['Age'].mean()
    #print('Age column mean  = ', mean)        # 29.69911764705882
    std = df['Age'].std()
    #print('Age std = ', std)        # 14.526497332334042
    is_null = df['Age'].isnull().sum()
    #print(is_null, 'age values are null')  # 177 age values are null
    print('Replacing column Age nan values with ', is_null, ' random values between ', mean - std, ' and ', mean + std, ' based on the mean and std of column Age')
    # random.randint(low, high=None, size=None, dtype=int)
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)    # rand_age is a numpy array with 177 values
    #print('rand_age: ', rand_age)       # rand_age:  [27 41 42 .. 41 40 20 40]
    # Replace NaN values in Age column with the random values generated
    age_copy = df["Age"].copy()
    age_copy[np.isnan(age_copy)] = rand_age     # np.isnan() tests element-wise whether it is NaN or not and returns the result as a boolean array
    #print(age_copy)
    #print('# of missing values in age_copy: ',age_copy.isnull().sum(),'\n')
    df['Age'] = age_copy
    #print(df['Age'].isnull().sum(), 'age values are null')  # 0 age values are null  
    # Convert column Age from float64 to int32
    df['Age'] = df['Age'].astype(int)
    
    # Create column AgeGrp from column Age with ranges of ages assigned to a numerical value
    # where 0=0..11 yrs, 1=12..18, 2=19..22, 3=23..27, 4=28..33, 5=34..40, 6=41..66, 7=66..Inf
    df['AgeGrp'] = df['Age'].copy()
    df.loc[df['AgeGrp'] <= 11, 'AgeGrp'] = 0
    df.loc[(df['AgeGrp'] > 11) & (df['AgeGrp'] <= 18), 'AgeGrp'] = 1
    df.loc[(df['AgeGrp'] > 18) & (df['AgeGrp'] <= 22), 'AgeGrp'] = 2
    df.loc[(df['AgeGrp'] > 22) & (df['AgeGrp'] <= 27), 'AgeGrp'] = 3
    df.loc[(df['AgeGrp'] > 27) & (df['AgeGrp'] <= 33), 'AgeGrp'] = 4
    df.loc[(df['AgeGrp'] > 33) & (df['AgeGrp'] <= 40), 'AgeGrp'] = 5
    df.loc[(df['AgeGrp'] > 40) & (df['AgeGrp'] <= 66), 'AgeGrp'] = 6
    df.loc[ df['AgeGrp'] > 66, 'AgeGrp'] = 6
    # Look at the distribution of the AgeGrp ranges and make sure none is considerably larger than the others.
    """
    print(df['AgeGrp'].value_counts())
    4    165
    6    164
    5    136
    3    133
    2    119
    1    106
    0     68
    """

    # Fare
    # Convert Fare from a float to int32 using astype():
    if df['Fare'].isnull().sum() > 0:
        print("Replacing ", df['Fare'].isnull().sum(), " values in col 'Fare' with the value 0 to achieve conversion to integer.")
        df['Fare'] = df['Fare'].fillna(0)
    df['Fare'] = df['Fare'].astype(int)

    # Create new column FareGrp as ranges of fares from column Fare.
    # Use Pandas qcut(Quantile-based discretization function) to create
    # 6 bins for FareGrp from the data in column Fare, 
    # i.e. 0-0.16, 0.16-0.33, 0.33-0.5, 0.5-0.67, 0.67-0.83, 0.83-1.0
    """
    print(pd.qcut(df['Fare'], q=6))
    Categories (6, interval[float64, right]): [(-0.001, 7.0] < (7.0, 8.0] < (8.0, 14.0] < (14.0, 26.0] < (26.0, 52.0] < (52.0, 512.0]]
    """
    print('Creating FareGrp as 6 quantile based bins from column Fare where: 0=(-0.001, 7.0], 1=(7.0, 8.0], 2=(8.0, 14.0], 3=(14.0, 26.0], 4=(26.0, 52.0], 5=(52.0, 512.0]')
    df['FareGrp'] = df['Fare'].copy()
    df.loc[ df['FareGrp'] <= 7.91, 'FareGrp'] = 0
    df.loc[(df['FareGrp'] > 7.91) & (df['FareGrp'] <= 14.454), 'FareGrp'] = 1
    df.loc[(df['FareGrp'] > 14.454) & (df['FareGrp'] <= 31), 'FareGrp']   = 2
    df.loc[(df['FareGrp'] > 31) & (df['FareGrp'] <= 99), 'FareGrp']   = 3
    df.loc[(df['FareGrp'] > 99) & (df['FareGrp'] <= 250), 'FareGrp']   = 4
    df.loc[ df['FareGrp'] > 250, 'FareGrp'] = 5
    df['FareGrp'] = df['FareGrp'].astype(int)
    """
    print(df['FareGrp'].value_counts())
    0    241
    2    223
    1    216
    3    158
    4     44
    5      9
    """    
    # df['FareGrp']: 0=(-0.001, 7.0], 1=(7.0, 8.0], 2=(8.0, 14.0], 3=(14.0, 26.0], 4=(26.0, 52.0], 5=(52.0, 512.0]

    # Name
    # Extract the titles from column Name to create a new feature.
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    # Extract the titles
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    """
    print(print(df['Title'].unique()))
    ['Mr' 'Mrs' 'Miss' 'Master' 'Don' 'Rev' 'Dr' 'Mme' 'Ms' 'Major' 'Lady'
    'Sir' 'Mlle' 'Col' 'Capt' 'Countess' 'Jonkheer']
    """
    # replace titles with a more common title or as Rare
    print('Creating new column titles from column Name where: 1 = Mr, 2 = Miss, 3 = Mrs, 4 = Master, 5 = Rare')
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir'], 'Mr')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    """
    print(df['Title'].value_counts())
    Mr        517
    Miss      185
    Mrs       126
    Master     40
    Rare       23
    """
    # Covert the titles into numbers.  1 = Mr, 2 = Miss, 3 = Mrs, 4 = Master, 5 = Rare
    df['Title'] = df['Title'].map(titles)

    # Create column 'Age_Class' from Age * Class
    df['Age_Pclass'] = df['Age'] * df['Pclass']
    # Scale 'Age_Pclass'
    df['Age_Pclass'] = scale_df(df['Age_Pclass'])

    # Create column 'FarePP' for the fare per person, taking into consideration families
    df['FarePP'] = df['Fare']/(df['Relatives']+1)
    df['FarePP'] = df['FarePP'].astype(int)
    # Scale 'FarePP'
    df['FarePP'] = scale_df(df['FarePP'])
    
    # Drop the columns 'PassengerId' 'Name', 'SibSp', 'Parch', 'Sex'
    #print(df.columns.values)
    #['Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Embarked' 'Sex_MF01' 'Relatives' 'Deck']
    df.drop(columns=['PassengerId','Name', 'SibSp', 'Parch', 'Sex', 'Ticket'], inplace=True)

    # Rename column 'Sex_MF01' to 'Sex'
    df.rename(columns={"Sex_MF01": "Sex"}, inplace=True)

    #print(df.columns.values)
    #['Survived' 'Pclass' 'Age' 'Fare' 'Embarked' 'Sex' 'Relatives' 'Deck' 'Title']

    # Show the missing data
    """
    total = df.isnull().sum().sort_values(ascending=False)
    percent_1 = df.isnull().sum()/df.isnull().count()*100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
    print('Missing data: \n', missing_data.head(5))
    """

    return df


def model_building_donges(df):
    # "Predicting the Survival of Titanic Passengers" by Niklas Donges
    # https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
    # A general rule is that, the more features you have, the more likely your model will 
    # suffer from overfitting and vice versa.

    print('\n', 'model_building_doges()')

    # Split the data into train and test sets using train_test_split().
    from sklearn.model_selection import train_test_split
    X = df[['Pclass', 'AgeGrp', 'Age_Pclass', 'Sex', 'FareGrp', 'FarePP', 'Relatives', 'Deck', 'Embarked', 'Title']]
    # From correlation: 'Sex','Title','Pclass','Age_Pclass','FareGrp','Deck'
    #X = df[['Sex','Title','Pclass','Age_Pclass','FareGrp','Deck']]
    y = df['Survived']
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    """
    X = df[['Pclass', 'Age', 'AgeGrp', 'Age_Pclass', 'Sex', 'Fare', 'FareGrp', 'FarePP', 'Relatives', 'Deck', 'Embarked', 'Title']]
    y = df['Survived']
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #X_train = df.drop('Survived', axis=1)
    """

    # Uncomment the section below to run the model
    """
    # Stochastic Gradient Descent (SGD):
    print('\nStochastic Gradient Descent (SGD):')
    sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
    sgd.fit(X_train, Y_train)
    Y_pred = sgd.predict(X_test)
    sgd.score(X_train, Y_train)
    acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
    #print('\tacc_sgd: ', acc_sgd)
    """

    # Uncomment the section below to run the model
    """
    print('\nRandom Forest:')
    # Random forest builds multiple decision trees and merges them together to get a more accurate and stable prediction.
    # It can be used for both classification and regression problems.
    # For the most part it has all the hyperparameters of a decision-tree classifier and also all the hyperparameters 
    # of a bagging classifier, to control the ensemble itself.
    # Random forest makes it very easy to measure the relative importance of each feature. 
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_prediction = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    #print('\tacc_random_forest: ', acc_random_forest)
    # Evaluate the relative importance of each feature in the model:
    importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    print(importances.head(15))
    # importances.plot.bar()
    # Another way to evaluate a random-forest classifier that is more accurate than the score we used before
    # is the out-of-bag samples to estimate the generalization accuracy.
    # Using the out-of-bag error estimate removes the need for a test set.
    # https://towardsdatascience.com/what-is-out-of-bag-oob-score-in-random-forest-a7fa23d710
    print('\t', 'Random forest oob score: ', round(random_forest.score(X_test, y_test),3))
    """

    # Uncomment the section below to run the models below
    """
    print('\nLogistic Regression:')
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    #print('\tacc_log: ', acc_log)

    print('\nK Nearest Neighbor:')
    knn = KNeighborsClassifier(n_neighbors = 3) 
    knn.fit(X_train, Y_train)  
    Y_pred = knn.predict(X_test)  
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    #print('\tacc_knn: ', acc_knn)

    print('\nGaussian Naive Bayes:')
    gaussian = GaussianNB() 
    gaussian.fit(X_train, Y_train)  
    Y_pred = gaussian.predict(X_test)  
    acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
    #print('\tacc_gaussian: ', acc_gaussian)

    print('\n', "Perceptron:")
    perceptron = Perceptron(max_iter=5)
    perceptron.fit(X_train, Y_train)
    Y_pred = perceptron.predict(X_test)
    acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
    #print('\t, "acc_perceptron: ', acc_perceptron)

    print('\n', "Linear Support Vector Machine:")
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    #print('\t, "acc_linear_svc: ', acc_linear_svc)

    print('\n', "Decision Tree:")
    decision_tree = DecisionTreeClassifier() 
    decision_tree.fit(X_train, Y_train)  
    Y_pred = decision_tree.predict(X_test)  
    acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
    #print('\t, "acc_decision_tree: ', acc_decision_tree)

    results = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
                'Random Forest', 'Naive Bayes', 'Perceptron', 
                'Stochastic Gradient Decent', 
                'Decision Tree'],
        'Score': [acc_linear_svc, acc_knn, acc_log, 
                acc_random_forest, acc_gaussian, acc_perceptron, 
                acc_sgd, acc_decision_tree]})
    result_df = results.sort_values(by='Score', ascending=False)
    result_df = result_df.set_index('Score')
    print('\n', result_df.head(10))
    """

    # Uncomment the section below to run this model
    """
    print('\n', "K-Fold Cross Validation:")
    # K-Fold Cross Validation on our random forest model, using 10 folds (K = 10). 
    from sklearn.model_selection import cross_val_score
    rf = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
    print('\t', "Scores:", scores)
    print('\t', "Mean:", scores.mean())
    print('\t', "Standard Deviation:", scores.std())
    """

    # Uncomment the section below and run it once to obtain optimized
    # parameters for Random Forest.
    """
    print('\n', 'Hyperparameter Tuning (takes very long ~900 sec to run):')
    param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}
    from sklearn.model_selection import GridSearchCV, cross_val_score
    #rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=1, n_jobs=-1)
    clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
    clf.fit(X_train, Y_train)
    print(clf.best_params_)
    # Result:  {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 25, 'n_estimators': 700}
    """
    # Using the above optimized parameters for Random Forest obtained from Hyperparameter Tuning:
    # (manually edit below to reflect: {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 25, 'n_estimators': 700})
    print('\n', 'Optimized Random Forest from the Hyperparameter Tuning results:')
    random_forest = RandomForestClassifier(criterion = "gini", 
                                        min_samples_leaf = 5, 
                                        min_samples_split = 25,   
                                        n_estimators=700, 
                                        max_features='sqrt', 
                                        oob_score=True, 
                                        random_state=1, 
                                        n_jobs=-1)

    random_forest.fit(X_train, Y_train)
    Y_prediction = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    print('\t', "oob score:", round(random_forest.oob_score_, 4)*100, "%")
    # oob score: 83.15 %

    # Confusion Matrix:
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
    confusion_matrix_results = confusion_matrix(Y_train, predictions)
    #print('\n', confusion_matrix_results)   # [[356  36]  [ 79 152]]  
    print('\n', 'Confusion Matrix Results:')
    print('\t', confusion_matrix_results[0][0], 'passengers were correctly classified as not survived (called true negatives)')
    print('\t', confusion_matrix_results[0][1], 'passengers were incorrectly classified as not survived (false positives)')
    print('\t', confusion_matrix_results[1][0], 'passengers were incorrectly classified as survived (false negatives)')
    print('\t', confusion_matrix_results[1][1], 'passengers were correctly classified as survived (true positives).')

    print('\n', 'Precision and Recall:')
    from sklearn.metrics import precision_score, recall_score
    print('\t', "Precision:", round(precision_score(Y_train, predictions), 3), ' (model predicts survival correctly ', round(precision_score(Y_train, predictions)*100, 0), '% of the time)')
    print('\t', "Recall:", round(recall_score(Y_train, predictions), 3), ' (model predicted the survival of ', round(recall_score(Y_train, predictions)*100, 0), '% of the passengers who actually survived)')

    # F-Score:
    # The F-score is the combination of precision and recall into one score.
    # It assigns much more weight to low values.
    from sklearn.metrics import f1_score
    print('\n', 'F-Score: ', round(f1_score(Y_train, predictions),3))
    # F-Score:  0.726

    # Plot of the precision and recall with the threshold:
    from sklearn.metrics import precision_recall_curve
    # getting the probabilities of our predictions
    y_scores = random_forest.predict_proba(X_train)
    y_scores = y_scores[:,1]
    precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
    def plot_precision_and_recall(precision, recall, threshold):
        plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
        plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
        plt.xlabel("threshold", fontsize=19)
        plt.legend(loc="upper right", fontsize=19)
        plt.ylim([0, 1])
    plt.figure(figsize=(14, 7))
    plot_precision_and_recall(precision, recall, threshold)
    # (uncomment below to see the plot)
    #plt.show()
    # The plot shows recall is falling of rapidly at a precision of around 85%
    # (intersection of the recall and precision lines).
    # For this reason, choose the precision/recall tradeoff before 85% precision
    # at about 75% precision (y-axis), and a corresponding threshold of 0.4

    # Plot the precision and recall against each other:
    # (uncomment below to generate and see the plot)
    """
    def plot_precision_vs_recall(precision, recall):
        plt.plot(recall, precision, "g--", linewidth=2.5)
        plt.ylabel("recall", fontsize=19)
        plt.xlabel("precision", fontsize=19)
        plt.axis([0, 1.5, 0, 1.5])
    plt.figure(figsize=(14, 7))
    plot_precision_vs_recall(precision, recall)
    plt.show()
    """

    # ROC AUC Curve
    from sklearn.metrics import roc_curve
    # compute true positive rate and false positive rate
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)
    # plotting them against each other
    def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
        plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'r', linewidth=4)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate (FPR)', fontsize=16)
        plt.ylabel('True Positive Rate (TPR)', fontsize=16)
    plt.figure(figsize=(14, 7))
    plot_roc_curve(false_positive_rate, true_positive_rate)
    # (uncomment below to see the plot)
    #plt.show()    

    # ROC AUC Score
    # A classifiers that is 100% correct would have a ROC AUC Score of 1,
    # and a completely random classiffier would have a score of 0.5.
    from sklearn.metrics import roc_auc_score
    r_a_score = roc_auc_score(Y_train, y_scores)
    if r_a_score <= 0.5:
        print('\n', "ROC-AUC Score: ", round(r_a_score, 3), ' really poor!')
    else:
        print('\n', "ROC-AUC Score:", round(r_a_score, 3))
    # ROC-AUC-Score: 0.921


def model_building_tbu(df):
    # "Building a Machine Learning Model Step By Step With the Titanic Dataset" by Taha Bilal Uyar
    # https://medium.com/swlh/building-a-machine-learning-model-step-by-step-with-the-titanic-dataset-e3462d849387

    print('\n', 'model_building_tbu()', '\n')

    # Split the data into train and test sets using train_test_split().
    from sklearn.model_selection import train_test_split
    X = df[['Pclass', 'AgeGrp', 'Age_Pclass', 'Sex', 'FareGrp', 'FarePP', 'Relatives', 'Deck', 'Embarked', 'Title']]
    # From correlation: 'Sex','Title','Pclass','Age_Pclass','FareGrp','Deck'
    #X = df[['Sex','Title','Pclass','Age_Pclass','FareGrp','Deck']]
    y = df['Survived']
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Random Forest:  
    # Create a model by using sklearn.RandomForestClassifier().
    # Random forest is a tree-based model so it doesn’t require feature scaling.
    # It can soften a non-linear model.
    from sklearn.ensemble import RandomForestClassifier
    rfc=RandomForestClassifier()
    #rfc=RandomForestClassifier(random_state=35)
    rfc.fit(X_train, Y_train)
    rfc.score(X_test,y_test)
    print("\nRandom Forest:")
    delta = round(abs(rfc.score(X_train, Y_train) - rfc.score(X_test,y_test)),3)
    print("\ttrain accuracy: ", round(rfc.score(X_train, Y_train),3))      
    print("\ttest accuracy: ", round(rfc.score(X_test,y_test),3))      
    print("\tDelta: ", delta)   
    """
    Random Forest:
            train accuracy:  0.978
            test accuracy:  0.817
            Delta:  0.16
    """                                        

    # Apply feature scaling to the model.
    """
    print('\nApplying feature scaling:')
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X_train_sc=scaler.fit_transform(X_train)
    X_test_sc=scaler.transform(X_test)
    # Try Random Forest again.
    rfc=RandomForestClassifier()
    #rfc=RandomForestClassifier(random_state=35)
    rfc.fit(X_train_sc, Y_train)
    delta = round(abs(rfc.score(X_train_sc, Y_train) - rfc.score(X_test,y_test)),3)
    print("\ttrain accuracy: ", round(rfc.score(X_train_sc, Y_train),3))      # 0.978
    print("\ttest accuracy: ", round(rfc.score(X_test_sc,y_test),3))          # 0.795
    print("\tDelta", delta)                                                   # 0.163
    """

    # Note the differences between train and test of 0.978 and 0.776, respectively.
    # Try to fit the model to each feature (column), it can give an insight into which columns cause this slight overfitting.
    """
    rfc=RandomForestClassifier(random_state=35)
    print("\nFeature", "\tTrain", "\tTest", "\tDelta")
    for x in X_train.columns:       
        rfc.fit(X_train[[x]], Y_train)
        delta = round(abs(rfc.score(X_train[[x]], Y_train) - rfc.score(X_test[[x]],y_test)),3)
        print(x,'\t\t', round(rfc.score(X_train[[x]], Y_train),3), '\t', round(rfc.score(X_test[[x]],y_test),3), '\t', delta)
    """
    """
    Feature         Train   Test    Delta
    Pclass           0.673   0.694   0.021
    Age              0.684   0.534   0.15
    AgeGrp           0.642   0.593   0.049
    Age_Pclass       0.729   0.679   0.05
    Sex              0.785   0.791   0.006
    Fare             0.737   0.66    0.076
    FareGrp          0.655   0.619   0.035
    FarePP           0.709   0.634   0.075
    Relatives        0.663   0.679   0.016
    Deck             0.701   0.679   0.022
    Embarked         0.652   0.601   0.051
    Title            0.787   0.799   0.012
    """
    # The Sex column has nearly the same train and test accuracy, which is ideal.
    # The Age, Fare, and FarePP columns have the most difference between train and test.  
    # Normally converting them from numerical to label encoded is the solution, but
    # that was already done with Age -> AgeGrp and Fare -> FareGrp.

    """
    print("\nDroping columns Age and Fare:")
    X = df[['Pclass', 'AgeGrp', 'Age_Pclass', 'Sex', 'FareGrp', 'Relatives', 'Deck', 'Embarked', 'Title']]
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Run the Random Forest model again
    rfc=RandomForestClassifier()
    rfc.fit(X_train, Y_train)
    rfc.score(X_test,y_test)
    delta = round(abs(rfc.score(X_train, Y_train) - rfc.score(X_test,y_test)),3)
    print("\ttrain accuracy: ", round(rfc.score(X_train, Y_train),3))   # 0.971     
    print("\ttest accuracy: ", round(rfc.score(X_test,y_test),3))       # 0.799
    print("\tDelta: ", delta)                                           # 0.173
    # Dropping the columns Age, Fare, and FarePP improved the test accuracy from 0.791 to 0.799
    # but the difference between train and test got only slightly better (less).
    """

    # Hyperparameter Tuning
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    rfc_parameters = { 
        'n_estimators': [100,200,500],
        'max_features': ['sqrt', 'log2'],
        'max_depth' : [6,8,10],
        'criterion' :['gini', 'entropy'],
        'min_samples_split': [2, 4, 6]
    }
    # RandomizedSearchCV selects random combinations usually finds the best parameters in a shorter time.
    print("\nHyperparameter Tuning with RandomizedSearchCV:")
    rand_search= RandomizedSearchCV(rfc, rfc_parameters, cv=5)
    rand_search.fit(X_train, Y_train)
    print("\t", rand_search.best_params_)     # {'n_estimators': 500, 'min_samples_split': 4, 'max_features': 'sqrt', 'max_depth': 6, 'criterion': 'entropy'}
    print("\tbest accuracy :",round(rand_search.best_score_,3))    # 0.830

    # GridSearchCV is an exhaustive search over specified parameters. 
    # It tries each combination in the grid of hyperparameter values,
    # and may take a long time with a large data set and many hyperparameters. 
    """
    print("\nHyperparameter Tuning with GridSearchCV:")
    grid_search= GridSearchCV(rfc, rfc_parameters, cv=5)
    grid_search.fit(X_train, Y_train)
    print("\n", grid_search.best_params_)     # {'criterion': 'entropy', 'max_depth': 6, 'max_features': 'log2', 'min_samples_split': 4, 'n_estimators': 500
    print("\tbest accuracy :",round(grid_search.best_score_,3))    # 0.833
    """
    

#@profile    # instantiating the decorator for the function/code to be monitored for memory usage by memory-profiler
def main():

    # ----------------------------------------------------------------------------
    # Data import, cleaning, EDA, pre-processing.

    # Load the Titanic dataset
    df = pd.read_csv('data/titanic.csv')    # Download the CSV file to a subfolder 'data' under the folder where this script resides.

    # Data Cleaning
    df = data_cleaning(df)

    # Data pre-processing
    df = data_preprocessing(df)
    
    show_df_contents(df)

    # Exploratory Data Analysis (EDA)
    # (Comment out modeling that follows and then uncomment e_d_a() to work with it)
    #e_d_a(df)

    """
    # Create a heatmap to visualize the correlation between the features in DataFrame df.
    sns.heatmap(df.corr(), cmap='RdYlGn', annot=True)
    plt.show()
    # From prior correlation the following features are dominate: 'Sex','Title','Pclass','Age_Pclass','FareGrp','Deck'
    """

    # Below are the specific model building activities from each author:
    #   model_building_tbu()    Taha Bilal Uyar
    #   model_building_donges   Niklas Donges
    # Uncomment the section in order to run the model

    # ----------------------------------------------------------------------------  
    # model_building_donges()

    # From prior correlation the following dominate features in the dataframe 'df'
    # will be used by model_building_doges(): 'Sex','Title','Pclass','Age_Pclass','FareGrp','Deck'   
    model_building_donges(df)

    # ----------------------------------------------------------------------------
    # model_building_tbu()

    # From prior correlation the following dominate features in the dataframe 'df'
    # will be used by model_building_tbu(): 'Sex','Title','Pclass','Age_Pclass','FareGrp','Deck'   
    #model_building_tbu(df)

    # Report the script execution time
    t_stop_sec = time.perf_counter()
    print('\nElapsed time {:6f} sec'.format(t_stop_sec-t_start_sec))


if __name__ == '__main__':
    main()		#only executes when NOT imported from another script
