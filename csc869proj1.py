import pandas as pd
import numpy as np
from copy import deepcopy
import time


    
"""
Handling of missing values
1.Take mean for continuous values and mode
for discrete values.
2.Remove null values
"""
def num_missing(x):
  return sum(x == '?')

def check_missing_values(data):
    print("Missing values per column:")
    print(data.apply(num_missing, axis=0))
    

#divide data in k_folds
def kfolds(df, no_folds):
    df_list = []
    n1 = 0
    bin_size = len(df) // no_folds
    n2 = bin_size
    for i in range(0,no_folds):
        if i==0:
            df_list.append(df.loc[:n2])
        if i==no_folds-1:
            df_list.append(df.loc[n1:])
        else:
            df_list.append(df.loc[n1:n2])
        #print(n1,n2)
        n1+=bin_size
        n2 = n1 + bin_size
    return df_list


def replace_missing_nonnumeric_values_with_mode(data):    
    # list of column names
    # -1 because we dont want to replace values in the last column of salary_income
    index = data.columns[:-1]

    # series of column names and types
    index_types = data.dtypes[:-1]
    
    for i in index:
        if index_types[i] == np.object_:
            data[i].replace('?',data[i].mode()[0], inplace=True)
            
def remove_record_with_missing_value(data):
    # list of column names
    index = data.columns[:-1]
    
    for i in index:
        if type(data[i][0]) != np.int64:
            data = data.drop(data.loc[data[i] == '?', (i)].index)
    return data

def classes_using_equi_width(data):
    
    # list of column names
    index = data.columns

    # series of column names and types
    index_types = data.dtypes
    classes = {}   

    for i in index:
        unique_values = None
    
        if i == 'age':
            # defining the bin-size as 5 based on the observations in tableau for 'age' column
            unique_values = data[i].value_counts(bins = (np.max(data[i])-np.min(data[i]))/5)
        elif i == 'hours-per-week':
            # defining the bin-size as 5 based on the observations in tableau for 'hours-per-week' column
            unique_values = data[i].value_counts(bins = (np.max(data[i])-np.min(data[i]))/10)
        elif i == 'capital-gain':
            # defining the no. of bins as 3 based on the observations in tableau for 'capital-gain' column
            unique_values = data[i].value_counts(bins=3)
        elif index_types[i] == np.int64:
            unique_values = data[i].value_counts(bins=8)
        else:
            unique_values = data[i].value_counts()
        classes.update({i: unique_values})
    
    return classes

def classes_using_gaussian(data):
    
    # list of column names
    index = data.columns

    # series of column names and types
    index_types = data.dtypes
    classes = {}
    
    for i in index:
        unique_values = None
    
        if index_types[i] == np.int64:
            temp = [1]*6
            u = data[i].mean()
            d = data[i].std()
            unique_values = pd.Series(data=temp, index=pd.interval_range(start = u -(3*d), end = u + (3*d), periods=6))
        else:
            unique_values = data[i].value_counts()
        classes.update({i: unique_values})
    
    return classes
            
"""
creating a datastructure with 
the count of each unique class of every column
for continues values continuous equi-width
classes have been created using pandas
"""
def naive_bayes_probability(data, test):
    
    # removing the columns 'Education-Num' as it is just a numeric representation of column 'Education' and hence is redundant.
    data = data.drop(['education-num'],axis=1)
    
    # list of column names
    index = data.columns

    # series of column names and types
    index_types = data.dtypes
    
    classes = classes_using_equi_width(data)
    flag = True
    #classes = classes_using_gaussian(data)
    #flag = False
    
    probability_of_yes = classes['income']['>50K'] / len(data)
    probability_of_no = classes['income']['<=50K'] / len(data)
    
    """
    data structure which maintains the probabilities of all the classes of all the columns given that the income is greater than 50K
    """
    classes_with_probability_given_yes =  deepcopy(classes)
    for i in index:
        k = 0
        for j in classes_with_probability_given_yes[i].index:
            if type(j) == pd.Interval:
                v1 = len(data.loc[(data[i] > j.left) & (data[i] <= j.right) & (data['income'] == ">50K"), [i,'income']])
                if v1 == 0:
                    v1 = 0.01
                v2 = classes['income'].loc['>50K']
                classes_with_probability_given_yes[i].iloc[k] = v1/v2
            else:
                v1 = len(data.loc[(data[i] == j) & (data['income'] == ">50K"), [i,'income']])
                if v1 == 0:
                    v1 = 0.01
                v2 = classes['income'].loc['>50K']
                classes_with_probability_given_yes[i].iloc[k] = v1 / v2
            k+=1
    
    """
    data structure which maintains the probabilities of all the classes of all the columns given that the income is less than 50K
    """        
    classes_with_probability_given_no =  deepcopy(classes)
    for i in index:
        k = 0
        for j in classes_with_probability_given_no[i].index:
            if type(j) == pd.Interval:
                v1 = len(data.loc[(data[i] > j.left) & (data[i] <= j.right) & (data['income'] == "<=50K"), [i,'income']])
                if v1 == 0:
                    v1 = 0.01
                v2 = classes['income'].loc['<=50K']
                classes_with_probability_given_no[i].iloc[k] = v1 / v2
            else:
                v1 = len(data.loc[(data[i] == j) & (data['income'] == "<=50K"), [i,'income']])
                if v1 == 0:
                    v1 = 0.01
                v2 = classes['income'].loc['<=50K']
                classes_with_probability_given_no[i].iloc[k] = v1 / v2
            k+=1
            
    predict(test, classes_with_probability_given_yes, classes_with_probability_given_no, probability_of_yes, probability_of_no,flag) 


"""
method that uses Naive Bayes formula to predict the income on test data
"""        
def predict(testset, yes_classes, no_classes,pyes,pno,flag):
    
    col_list = testset.columns[:-1]
    col_type_list = testset.dtypes[:-1]
    prediction = []
    for i in range(0,len(testset)):
        yes = 1
        no = 1
        for j in range(0,len(col_list)):
            #print(j,col_list[j])
            try:
                if col_type_list[j] == np.int64:
                    #if col_list[j] == 'fnlwgt':
                        #yes_classes['age'].iloc[yes_classes['age'].index.get_loc(0))]
                    tempY = yes_classes[col_list[j]].iloc[yes_classes[col_list[j]].index.get_loc(testset[col_list[j]].iloc[i])]
                    tempN = no_classes[col_list[j]].iloc[no_classes[col_list[j]].index.get_loc(testset[col_list[j]].iloc[i])]
                    if type(tempY) != np.float64:
                        tempY = yes_classes[col_list[j]].iloc[yes_classes[col_list[j]].index.get_loc(testset[col_list[j]].iloc[i])[0]]
                    yes = yes * tempY
                    
                    if type(tempN) != np.float64:
                        tempN = no_classes[col_list[j]].iloc[no_classes[col_list[j]].index.get_loc(testset[col_list[j]].iloc[i])[0]]
                    no = no * tempN
                else:                
                    yes = yes * yes_classes[col_list[j]][testset[col_list[j]].iloc[i]]                
                    no = no * no_classes[col_list[j]][testset[col_list[j]].iloc[i]]
                
                
            except KeyError as e:
                yes = yes * 0.00001
                no = no * 0.00001
                continue            
            
        yes = yes * pyes
        no = no * pno
        
        try:
            if yes > no:
                prediction.append('>50K')
            else:
                prediction.append('<=50K')
        except ValueError as e:
            print(e)
    
    testset.loc[:,'prediction'] = prediction
    
    #return test_data

"""
method which evaluates the accuracy of the predicted data on the test set
"""
def evaluate(t_data, evaluation_results,i):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    TP = len(t_data.loc[(t_data['income'] == ">50K") & (t_data['prediction'] == ">50K"), ['income','prediction']])
    TN = len(t_data.loc[(t_data['income'] == "<=50K") & (t_data['prediction'] == "<=50K"), ['income','prediction']])
    FP = len(t_data.loc[(t_data['income'] == "<=50K") & (t_data['prediction'] == ">50K"), ['income','prediction']])
    FN = len(t_data.loc[(t_data['income'] == ">50K") & (t_data['prediction'] == "<=50K"), ['income','prediction']])
    
    print("TP:",TP,"\nTN:",TN,"\nFP:",FP,"\nFN:",FN)
    accuracy = (TP+TN)/(TP+TN+FP+FN)*100
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = (2*(precision*recall))/(precision+recall)
    
    
    print("Evaluation Measures of fold:",i)
    print("Accuracy:",accuracy,"%")
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1 measure:",f1)
    
    evaluation_results = evaluation_results.append(pd.DataFrame(data=[[accuracy,precision,recall,f1]], columns=['accuracy','precision','recall', 'f1']), ignore_index=True)
    return evaluation_results
    


def main():
    
    # reading data from CSV
    data = pd.read_csv('adult.txt',sep=", ")
    
    check_missing_values(data)
    
    #Method 1 for handling missing values: Replace with mean/mode for continuous/discrete columns
    replace_missing_nonnumeric_values_with_mode(data)
    
    #Method 2 for handling missing values: Remove records with null value in any column
    #data = remove_record_with_missing_value(data)
    
    check_missing_values(data)
    
    #shuffle data randomly before kfold
    data = data.sample(frac=1).reset_index(drop=True)
    
    no_folds = 10
    df_list = kfolds(data, no_folds)
    df_len = len(df_list)
    
    evaluation_results = pd.DataFrame(data=None, columns=['accuracy','precision','recall', 'f1'])
    
    for i in range(0,df_len):
        if i == 0:
            train_data = pd.concat(df_list[1:])
            test_data = df_list[0]
        elif i == df_len:
            train_data = pd.concat(df_list[:-2])
            test_data = df_list[-1]
        else:
            train_data = pd.concat(df_list[:i]+df_list[i+1:])
            test_data = df_list[i]
            
        naive_bayes_probability(train_data, test_data)
        evaluation_results = evaluate(test_data, evaluation_results,i)
    
    print("\n\n\n")
    print("-------Average Evaluation Results of the K-folds:--------")
    print("Avg accuracy:",evaluation_results['accuracy'].mean(),"%")
    print("Avg precision:",evaluation_results['precision'].mean())
    print("Avg recall:",evaluation_results['recall'].mean())
    print("Avg f1:",evaluation_results['f1'].mean())
    print("\n\n\n")

if __name__== "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    