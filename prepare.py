import acquire as a 
import pandas as pd
import numpy as np

def prep_iris():
    iris = a.get_iris_data()
    iris.columns.str.replace('.','').str.lower()
    iris = iris.reset_index(drop=True)
    iris = iris.drop(['measurement_id'], axis=1)
    iris = iris.rename(columns={'species_name':'species'})
    return iris

def prep_titanic():
    titanic=a.get_titanic_data()
    titanic = titanic.drop(['class'], axis=1)
    titanic = titanic.drop(['embarked'], axis=1)
    return titanic

def prep_telco():
    telco=a.get_telco_data()
    telco= telco.drop(['payment_type_id'],axis =1)
    telco=telco.drop(['contract_type_id'],axis =1)
    telco=telco.drop(['internet_service_type_id'],axis =1)
    telco.internet_service_type.value_counts(dropna=False)
    return telco


















from sklearn.model_selection import train_test_split
def split(df,target_variable):

    #first split
    train, validate_test = train_test_split(df, 
                 train_size=0.60, #size of the train df, and the test size will default to 1-train_size
                random_state=123, #set any number here for consistency
                 stratify=df[target_variable] #need to stratify on target variable
                )
    
    #second split
    validate, test = train_test_split(validate_test, #this is the df that we are splitting now
                test_size=0.50, #set test or train size to 50%
                 random_state=123, #gotta send in a random seed
                stratify=validate_test[target_variable]#still got to stratify
                )
    
    return train, validate, test

import acquire as a 
import pandas as pd
import numpy as np

