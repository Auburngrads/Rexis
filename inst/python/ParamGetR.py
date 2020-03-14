# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:21:17 2020
ParamGet
@author: mchale
"""
filename_str="Datasets/heart.csv"


def ParamGetR(filename_str="Datasets/heart.csv", filesave_str = "Current Results.xlsx", pltsave_str = "Current CI plt.png"):
    import pandas as pd
    #import tensorflow.keras as K
    import numpy as np
    #import os
    from sklearn import preprocessing
    from numpy import linalg as LA    
    
    
    
    #path = "C:/Users/mchale/OneDrive/Documents/AFIT/Research/Thesis/Thesis Code"
    
    #os.chdir(path)
    
    #Import theas pandas df
    #filename="data_imputed.csv" default during debugging
    filename=filename_str
    data = pd.read_csv(filename)
   
    firstcolname=list(data)[0]
    #y=data[[firstcolname]]
    del data[firstcolname]

    #Minimax normalization of data
    min_max_scaler = preprocessing.MinMaxScaler()
    final_data = min_max_scaler.fit_transform(data)

    #Get n and m
    n,m =np.shape(final_data)

    if n>= 10^3:
        big_set=True
    else:
        big_set=False
        
    if m>= 10:
        many_vars=True
    else:
        many_vars=False


    #Get MajVarsCat
    type_vect=np.zeros((1, m))
    
    for i in range(0, m): #weird python indexing will generate num_reps iterations
        
        if data.iloc[:,i].nunique()>=12:
            type_vect[0,i]=1

        if np.mean(type_vect) >=.5: #tests whether the majority of columns have many levels
            data_categorical = True
        else:
            data_categorical = False
                

    #Get Condition
    if LA.cond(final_data)>=10^5:
        ill_cond=True
    else:
        ill_cond=False


    #Generate Recommendation
    #Big set (Right side of tree)
    if big_set==True and many_vars==True and data_categorical==True  and ill_cond == True:
        Ranks=[3,1,2,4,5]
        
        
        
    elif big_set==True and many_vars==True  and data_categorical==False and ill_cond == True:
        Ranks=[4,3,2,5,1]
    
    elif big_set==True and many_vars==True and data_categorical==False and ill_cond == True:
        Ranks= [3, 1, 2, 4, 5]
        
    elif big_set==True and many_vars==True and data_categorical==False and ill_cond == False:
        Ranks= [3, 2, 1, 4, 5]
        
        #Small set (Left side of tree)
    elif big_set==False and data_categorical==True:# and many_vars==True #and ill_cond == False:
        Ranks= [3, 2, 5, 1, 4]
        
    elif big_set==False and data_categorical==False: # and many_vars==True #and ill_cond == False:
        Ranks= [4, 3, 5, 2, 1]  #SVR SVM  RF DT NB
            
    else:
        Ranks= [4, 2, 1, 3, 5]
    
    
    rank_array = Ranks
    
    #The following several lines may be used to create a dataframe of rank info. We find an object is much easier to format in Thesis_func_execute
    #Dictionary to assign a value to each indx of future df
    rankdata = {'Decision Tree': Ranks[0], 'Random Forest': Ranks[1], 'Naive Bayes' : Ranks[2], 'SVM' : Ranks[3], 'SVR': Ranks[4]}
    Ranks=pd.DataFrame.from_dict(rankdata, orient='index')
    Ranks.columns= ['Ranks']
    
    #df = {'Big Set': big_set, 'Many Vars': many_vars, 'Categorical Data': data_categorical, 'Ill conditioned': ill_cond, 'Ranks': Ranks}
    
    
    
     
    class result:
        def __init__(self, ranks, ranks_df, bigset, manyvars, categorical, illcond):
            self.ranks = ranks
            self.ranksdf = Ranks
            self.bigset = bigset
            self.manyvars = manyvars
            self.categorical = categorical
            self.illcond = illcond
    
    
    result_obj = result(rank_array, Ranks, big_set, many_vars, data_categorical, ill_cond)
    
    
    
    return result_obj



























