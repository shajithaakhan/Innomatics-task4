# Innomatics-task4
Performing Exploratory Data Analysis (EDA) on the data considering Salary as a target variable.
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
#Import the data
df= pd.read_csv("Book1.csv")
#display the head
df.head()
#shape of the data.
df.shape
#description of the data.
df.info()
#univariate analysis
df['DOB'] = pd.to_datetime(df['DOB'])
df['DOJ'] = pd.to_datetime(df['DOJ'])
df.isnull().sum().sort_values(ascending=False)
#univariate inculcating numerical and categorical features
from sklearn.impute import SimpleImputer
df['Salary'].mean()
df['Salary'].min()
df['Salary'].skew() #right skewed
df['Salary'].plot(kind='kde')
df['Salary'].plot(kind='box')
from scipy import stats
import matplotlib.pyplot as plt
stats.probplot(df['Salary'], dist='norm', plot=plt)
plt.grid()
clean = df[df.Salary >= 1000000]
clean['Salary'].plot(kind='box')
clean['Salary'].plot(kind='kde')
stats.probplot(clean['Salary'] ,dist="norm", plot=plt)
plt.grid()
sns.boxplot(data=df, x= 'Gender', y='Salary')
df.columns = df.columns.str.strip()
df_dropna = df.dropna()
df_dropna.info()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'most_frequent')a
df_mode_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df_mode_imputed.describe()
df_median_imputed = df.fillna(df.median())
df_median_imputed.describe()
numerical_features = list(df.select_dtypes(include=['number']).columns)
categorical_features = list(df.select_dtypes(include=['object']).columns)

num_discrete_features = ['Salary', '10percentage', '12percentage',  'collegeGPA', 'Domain','conscientiousness', 'agreeableness', 'extraversion','nueroticism', 'openess_to_experience']
numerical_features = [feature for feature in numerical_features if feature not in num_discrete_features]
print(numerical_features)
from sklearn.impute import KNNImputer
cat_encoded_df = pd.get_dummies(df[categorical_features])
combined_df =pd.concat([df[numerical_features],df[num_discrete_features],cat_encoded_df],axis=1)
knn_imputer = KNNImputer(n_neighbors=1)
df_knn_imp = pd.DataFrame(knn_imputer.fit_transform(combined_df),
                          columns=combined_df.columns,
                          index=combined_df.index)
df_knn_imp.describe()
q1= df['Salary'].quantile(0.25)
q3= df['Salary'].quantile(0.75)
IQR =q3-q1
income_lower_bound = q1-1.5*IQR
income_upper_bound = q3-1.5*IQR
clean =df[(df['Salary']>=income_lower_bound)&(df['Salary']<=income_upper_bound)]
clean['Salary'].plot(kind='box')
clean['Salary'].plot(kind='kde')
stats.probplot(clean['Salary'],dist="norm",plot=plt)
plt.grid()
def numerical_univariate_analysis(numerical_data):
  for col_name in numerical_data:
    print("*"*10, col_name, "*"*10)
    print(numerical_data[col_name].agg(['min','max','mean','median','std','skew','kurt']))
    print()
numerical_univariate_analysis(df[['Salary']])
df['JobCity'].nunique()
df['JobCity'].unique()
df['JobCity'].value_counts(normalize=True)
df['JobCity'].value_counts().plot(kind='barh')
def discrete_univariate_analysis(discrete_data):
  for col_name in discrete_data:
    print("*"*10, col_name, "*"*10)
    print(discrete_data[col_name].agg(['count','nunique','unique']))
    print('Value Counts: \n', discrete_data[col_name.value_counts(normalize=True)])
    print()
  discrete_univariate_analysis(df[['GraduationYear']])
#all at once
df[numerical_features].plot(kind='hist',subplots=True,layout=(9,2), figsize = (17,17))
df[numerical_features].plot(kind='box' ,subplots=True,layout=(9,2), figsize = (17,17))
df[numerical_features].plot(kind='barh' ,subplots=True,layout=(9,2), figsize = (17,17))
#bivariate analysis
df.plot(kind='scatter',x='Gender',y='Salary')
clean.plot(kind='scatter',x='Gender',y='Salary')
tab = pd.crosstab(df['Gender'], df['JobCity'], normalize='index')
tab.plot(kind='bar')
df.boxplot(by='Gender', column='Salary')
Question
After doing your Computer Science Engineering if you take up jobs as a Programming Analyst, Software Engineer, Hardware Engineer and Associate Engineer you can earn up to 2.5-3 lakhs as a fresh graduate.
df['Specialization'].value_counts()["computer engineering"]
tab = pd.crosstab(['Specialization"computer engineering"'],df['Designation'],normalize='index')
tab
tab.plot(kind='bar',figsize=(100,100))
df1=tab
df1.ranspose()
df2= df1.append(df['Salary'],ignore_index=True)
df2.transpose()
Question 2
Is there a relationship between gender and specialization? (i.e. Does the preference of Specialisation depend on the Gender?
#relationship between gender and specialization
tab = pd.crosstab(df['Specialization'], df['Gender'], normalize='index')
tab.plot(kind='box')
tab = pd.crosstab(df['Gender'], df['Specialization'], normalize='index')
tab.plot(kind='box', figsize=(100,100))
