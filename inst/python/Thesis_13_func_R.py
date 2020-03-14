
    # -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:37:41 2019
Thesis Function v 9. Compares classifying performance and time for each ML method
    1) Decision Trees
    2) Random Forest
    3) Naive Bayes
    4) SVM with PCA
    5) SVR

Required arguments: 
    1) name of dataset (string). Dataset must inlucde .csv label. Must include headers
and a binary target in the first column. Other variables may be real numerical with no
missing values.

Optional arguments: 
    1) name to save recall and time info (string). Must include .xlsx label
    2) name to save graphic of results

Returns: Python object of the custom class "Results"


Future Revisions to Function
-Automatically iterate through all datasets in a folder

NOTE: LINE 312 WRITE TO CSV IS COMMENTED OUT!
@author: Marc Chale

"""
# %% Debug courtesy code


#filepath_str="C:/Users/mchale/OneDrive/Documents/AFIT/WInter 20/OPER 782 Data Science Programs/Rexis/data/heart.csv"
#save_path="C:/Users/mchale/Desktop"
#filesave_str = "Heart"#pltsave_str = "Heart"
#loan=Thesis_12_func("Datasets/Bank_Personal_Loan.csv", "Loan_"+timestamp,  "Loan")

# %% Begin Function
def Thesis_13_func_R(filepath_str="C:/Users/mchale/OneDrive/Documents/AFIT/WInter 20/OPER 782 Data Science Programs/Rexis/data/heart.csv", save_path="C:/Users/mchale/Desktop", filesave_str = "Current Results", pltsave_str = "Current CI plt"):
#    """Input the file name of input data as string. Outputs recall"""
    
    # %%DEFINE
    print("Code has started running!", flush=True)
    print('BEGIN DEFINE', flush=True)
    
    
    import pandas as pd
    #import tensorflow.keras as K
    import numpy as np
    import time
    import os
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import  recall_score #,precision_score
    from sklearn import svm
    from sklearn.svm import SVR
    from scipy.stats import t
    from scipy.stats import spearmanr
    from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier  
    from ParamGetR import ParamGetR
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from pareto import identify_pareto
    #from adjustText import adjust_text
    from matplotlib.text import OffsetFrom
   # import rpy2
    #from rpy2.robjects import r, pandas2ri
    # %% Minor pre-processing
    #Start timer
    num_reps=20
    start=time.time()

    recall_mat=np.zeros((5,num_reps))
    duration_mat=np.zeros((5,num_reps)) 
    
    ##IMPORT
    print('BEGIN IMPORT', flush=True)
    
    #Set Wroking Directory
#    os.chdir("C:/Users/mchale/OneDrive/Documents/AFIT/Research/Thesis/Thesis Code")
#    path = "C:/Users/mchale/OneDrive/Documents/AFIT/Research/Thesis/Thesis Code"
    
#    os.chdir(path)
    
    #Import theas pandas df
    #filename="data_imputed.csv" default during debugging
    #filename=filename_str
    data = pd.read_csv(filepath_str)
   
    firstcolname=list(data)[0]
    y=data[[firstcolname]]
    del data[firstcolname]

    #Minimax normalization of data
    min_max_scaler = preprocessing.MinMaxScaler()
    final_data = min_max_scaler.fit_transform(data)

    # %% Enter loop for each rep
    for i in range(1, num_reps+1): #weird python indexing will generate num_reps iterations
        seed = 18+i # fix random seed for reproducibility
        np.random.seed(seed)
    

        #Split the final data into train/test
        x_final_train, x_final_test, y_final_train, y_final_test =\
        train_test_split(final_data, y, test_size=0.2, random_state=seed, stratify=y)
        y_final_train=y_final_train.values.ravel()
    
    
    
        # %% MODEL
        print('BEGIN MODEL', flush=True)
        print("ITERATION", "%1.1d" % (i), flush=True)
    
        ##MODEL DECISION TREE USING THE FINAL TRAINING DATASET
    
        print('1/5: Creating Decision Tree Classifier', flush=True)
        r=1 #Index number of technique
        start_dt = time.time() #Record time Decision Tree begins

        # Instantiate a DecisionTreeClassifier 
        dt_final = DecisionTreeClassifier(random_state=seed)
        #defaults: max_depth default is until pure. default criterion is gini
        
        # Fit dt to the training set
        dt_final.fit(x_final_train, y_final_train)

        #Predict the class of each observation of a dataset
        y_pred_final_DT = dt_final.predict(x_final_test)
        
        #Record time decision tree completes
        now=time.time() 
        durationmin_dt = round((now-start_dt)/60)
        durationsec_dt = round((now-start_dt)%60)
        duration_mat[r-1,i-1]=(now-start_dt) #seconds
        
        print("The Decision Tree model and predictions have been generated in", "%2.2d:%2.2d" % (durationmin_dt, durationsec_dt), flush=True)
    

        ##MODEL RANDOM FOREST USING THE FINAL TRAINING DATASET
    
        print('2/5 Creating Random Forest Classifier From Final Data Set', flush=True)
        start_rf = time.time() #Record time Random Forest begins
        r=2 #Index number of technique
  
        #Instantiate RF    
        RF_final=RandomForestClassifier(random_state=seed)
    
        #Fit Model
        RF_final.fit(x_final_train, y_final_train)
        RandomForestClassifier(bootstrap=True, random_state=seed)
        #we use default settings
    
        #Predict
        y_pred_final_RF = RF_final.predict(x_final_test)    

        #Record time Random Forest Completes
        now=time.time() 
        durationmin_rf = round((now-start_rf)/60)
        durationsec_rf = round((now-start_rf)%60)
        duration_mat[r-1,i-1]=(now-start_rf) #seconds
    
        print("The Random Forest model and predictions have been generated in", "%2.2d:%2.2d" % (durationmin_rf, durationsec_rf), flush=True)
    
    
        ##MODEL NAIVE BAYES USING FINAL DATASET
        print('3/5: Creating Naive Bayes Model', flush=True)
        r=3 #Index number of technique
        start_nb = time.time() #Record time Naive Bayes begins
    
        #Instantiate Naive Bayes
        gnb = MultinomialNB()
        #Fit model and predict class
        y_pred_final_NB = gnb.fit(x_final_train, y_final_train).predict(x_final_test)
        
        #Record time Naive Bayes completes
        now = time.time() 
        durationmin_nb = round((now-start_nb)/60)
        durationsec_nb = round((now-start_nb)%60)
        duration_mat[r-1,i-1]=(now-start_nb) #seconds
    
        print("The Naive Bayes model and predictions have been generated in", "%2.2d:%2.2d" % (durationmin_nb, durationsec_nb), flush=True)
    

        print('4/5: Creating Support Vector Machine Model', flush=True)
        r=4 #Index number of technique
        start_svm = time.time() #Record time SVM begins
        
        #Instantiate SVM Classifier
        clf = svm.SVC(kernel='rbf', probability=False, class_weight='balanced')
        #Fit SVM Model
        #svm_y_scores = clf.fit(x_final_train, y_final_train).decision_function(x_final_test)
        clf.fit(x_final_train, y_final_train).decision_function(x_final_test)
        #Predict Class of test data
        y_pred_final_SVM = clf.predict(x_final_test)
    
    
        #Record time SVM completes
        now = time.time()
        durationmin_svm = round((now-start_svm)/60)
        durationsec_svm = round((now-start_svm)%60)
        duration_mat[r-1,i-1]=(now-start_svm) #seconds
    
        print("The SVM model and predictions have been generated in", "%2.2d:%2.2d" % (durationmin_svm, durationsec_svm), flush=True)
    
    
        ##MODEL SUPPORT VECTOR REGRESSION
        print('5/5 pt: Creating Support Vector Regression Model', flush=True)
        r=5#Index number of technique    
        start_svr = time.time() #Record time SVR begins

        #Instantiate SVM Classifier
        svr = SVR(C=1.0, epsilon=0.1) #default settings
        
        #Fit SVR Model
        svr.fit(x_final_train, y_final_train)
        
        #Predict class of test data with probability based on distance to decision boundary
        y_pred_final_SVR_prob = svr.predict(x_final_test)
        #Discretize the distance to decision boundary to a binary decision
        y_pred_final_SVR=np.absolute(y_pred_final_SVR_prob) #determine magnitude
        y_pred_final_SVR=np.around(y_pred_final_SVR,decimals=0) #round to neareast val
        y_pred_final_SVR= y_pred_final_SVR.astype(int)    
        y_pred_final_SVR= np.sign( y_pred_final_SVR)
    
        #Record time SVR completes
        now=time.time()
        durationmin_svr = round((now-start_svr)/60)
        durationsec_svr = round((now-start_svr)%60)
        duration_mat[r-1,i-1]=(now-start_svr) #seconds
    
        print("The SVR model and predictions have been generated in", "%2.2d:%2.2d" % (durationmin_svr, durationsec_svr), flush=True)
    
        # %% EVALUATE
        print('BEGIN EVALUATE', flush=True)
        print('>>>', flush=True)
    
    
        ##EVALUATE DECISION TREE
        print('1/5: Decision Tree', flush=True)
        r=1
        recall_mat[r-1,i-1]=recall_score(y_final_test, y_pred_final_DT)
        
    
        ##EVALUATE RANDOM FOREST
        print('2/5 Random Forest', flush=True)
        r=2
        recall_mat[r-1,i-1]=recall_score(y_final_test, y_pred_final_RF)
    
        ##EVALUATE NAIVE BAYES
        print('3/5: Naive Bayes', flush=True)
        r=3
        recall_mat[r-1,i-1]=recall_score(y_final_test, y_pred_final_NB)
        
        ##EVALUATE SUPPORT VECTOR MACHINE
        print('4/5: Support Vector Machine', flush=True) 
        r=4
        recall_mat[r-1,i-1]=recall_score(y_final_test, y_pred_final_SVM)
    
        ##EVALUATE SUPPORT VECTOR REGRESSION
        r=5
        print('5/5: Support Vector Regression', flush=True) 
        recall_mat[r-1,i-1]=recall_score(y_final_test, y_pred_final_SVR)  
    
        #Time stamp the end time of the loop
        now = time.time()
        durationmin = round((now-start)/60)
        durationsec = round((now-start)%60)
    
    
        print("Evaluation of Base Learners Complete", flush=True)
        print("Total ellapsed time is", "%2.2d:%2.2d, flush=True" % (durationmin, durationsec), flush=True)
    
    
        print("Generating Bonferroni Confidence Intervals", flush=True)
    
        mean_vect=np.mean(recall_mat, axis = 1) #vector of mean for each method
        
        print("MenVect Produced", flush=True)
        
        time_vect=np.mean(duration_mat, axis = 1) #vector of mean for each method
     
        print("Time Vect Worked", flush=True)
        
        alpha=0.05 #statistical significance
        
        rows = np.shape(mean_vect)[0] #number of samples being compared
        
        print("Rows Worked", flush=True)
        
        t_crit=t.ppf(1-alpha/rows, num_reps-1) #same t_crit for each HW
        
        S=np.std(recall_mat, axis=1)    #need to generate a vector of sample SDs
        
        half_width=t_crit*S/np.power(num_reps,0.5) #need to generate a vector of HWs
        
        print("HW Worked", flush=True)
        
        UB= mean_vect+half_width
        
        LB= mean_vect-half_width
        
        print("Completed Analysis (included intervals)", flush=True)
      
        # %%Create array of everything we wish to record
        #array2 is a dictionary that assigns a name to each list of data, which become df columns
        print("Begin Create Result Array", flush=True)

        
        array2 = {'Rep 1': recall_mat[:,0], 'Rep 2': recall_mat[:,1], 'Rep 3': recall_mat[:,2], 'Rep 4': recall_mat[:,3] , 'Rep 5': recall_mat[:,4], 'means': mean_vect, 'avg time': time_vect, 'SD': S, 'LB': LB, 'UB': UB }
        #Creates a data frame whos columns are each named list in array 2
        recall_df = pd.DataFrame(data=array2, index=['Decision Tree', 'Random Forest', 'Naive Bayes', 'SVM', 'SVR'])
        #Assign a new colum to the dataframe. This column is the ranking of the values in "means"
        recall_df['Ranks'] = list(recall_df["means"].rank(method='min' ,ascending = False).values)
        obs_ranks_avg = list(recall_df["means"].rank(method='average' ,ascending = False).values)
        
        #recall_df=pd.DataFrame.append(recall_df, rank_vect, ignore_index=True)
        
        recall_df = recall_df[['Rep 1' , 'Rep 2', 'Rep 3', 'Rep 4', 'Rep 5', 'means', 'avg time', 'SD', 'LB', 'UB', 'Ranks']]
        
        #Write results of recall and recall CI's to Excel
        
        #filesave_str = "Current Results.xlsx", pltsave_str = "Current CI plt.png
        #recall_df.to_excel(filesave_str+'.xlsx', engine='xlsxwriter')

        print("Result Array Completed", flush=True)
     
    #END OF LOOP!
    # %% Run Recomender Function
    print("Loop has completed", flush=True)
    print("Begin Creating Objects", flush=True)

    
    
    rec=ParamGetR(filepath_str)
    #Generate summary data of recommend
    rec_best_indx=int(np.where(rec.ranksdf==1)[0])
    rec_best_name=list(recall_df.index)[rec_best_indx]
    rec_best_recall=list(recall_df.means)[rec_best_indx]
    rec_best_time=list(recall_df['avg time'])[rec_best_indx]
    
    print("The slicing worked", flush=True)

    #Create a new class of objects with summary data of prediction
    class recommend:
        def __init__(self, big_set, categorical, illcond, many_vars, ranks, ranksdf, best_indx, best_name, best_recall, best_time):
            self.bigset = big_set
            self.categorical = categorical
            self.illcond = illcond
            self.manyvars = many_vars
            self.ranks = ranks
            self.ranksdf = ranksdf
            self.bestindx = best_indx
            self.bestname= best_name
            self.bestrecall = best_recall
            self.besttime = best_time

    rec_obj= recommend(rec.bigset, rec.categorical, rec.illcond, rec.manyvars, rec.ranks, rec.ranksdf, rec_best_indx, rec_best_name, rec_best_recall, rec_best_time)
    
    print("First Object Created", flush=True)

 
    
    # %%Generate the most important summary data
    print("Begin Creating Second Object", flush=True)

    
    true_best_indx=np.where(recall_df.Ranks==1)#dataframe still has obj 'Ranks'
    true_best_name=(recall_df.index)[true_best_indx]
    true_best_recall=max(recall_df['means'])
    
    count_true_bests=(true_best_indx[0].shape)[0]
    true_best_time=[] #preallocate list for the times of all the best algorithms
        
    
    for k in range(0, count_true_bests):
        true_best_time.append(recall_df['avg time'].iloc[ true_best_indx[0][k]]) #store the recalls for each member of best_indx

    #Create a new class of objects with summary data
    class true:
      def __init__(self, source, complete, mean_recall, avg_times, SD, LB, UB, Ranks, Ranks_avg, best_indx, best_name, best_recall, best_time):#, best_recall_indx, bestRecallVal):
        self.source = source
        self.complete = complete
        self.means = mean_recall
        self.avg_times = avg_times
        self.SD = SD
        self.LB = LB
        self.UB = UB
        self.ranks = Ranks
        self.ranks_avg= Ranks_avg
        self.bestindx = best_indx
        self.bestname= best_name
        self.bestrecall = best_recall
        self.besttime = best_time
    
    #Populate object with summary data of true results
    true_obj = true(filepath_str, recall_df, recall_df['means'], recall_df['avg time'], recall_df['SD'], recall_df['LB'], recall_df['UB'], recall_df['Ranks'], obs_ranks_avg, true_best_indx, true_best_name , true_best_recall, true_best_time)
    
    print("Finished Creating Second Object", flush=True)

    print("Begin Creating Third Object", flush=True)

    
diff_time=rec_obj.besttime-true_obj.besttime[0]
    diff_recall=rec_obj.bestrecall-true_obj.bestrecall
    #correlation=rec_obj.Ranks.corrwith(true_obj.Ranks)[0] ###########spearman!!
    correlation = spearmanr(rec_obj.ranks, obs_ranks_avg)
    hit=False
    
    #Hit check. Iterates through all true bests, logical test if any match index of rec. 
    #Could alternatively perform this check by if bestrecall-true_obj.bestrecall==0
    for k in range(0, count_true_bests):
        if  rec_obj.bestindx==true_obj.bestindx[0][k]:
            hit=True
            
            
    
    class compare:
        def __init__(self, diff_time, diff_recall, correlation, pcorr):
            self.time = diff_time
            self.recall = diff_recall
            self.corr = correlation
            self.pcorr = pcorr
            
    
    compare = compare(diff_time, diff_recall, correlation[0], correlation[1])



    
    class Results:
      def __init__(self, recommend_results, true_results, compare, hit):
        self.rec = recommend_results
        self.obs = true_results
        self.diff = compare
        self.hit = hit
    
    results = Results(rec_obj, true_obj, compare, hit)


    print("THird object all done done", flush=True)

    print('Return Objects All Completed', flush=True)

  # %%Plot Recall Bars      
    print('Creating Data Frame to Return', flush = True)
    
   # pandas2ri.activate()
    results_df=recall_df
    results_df=results_df.rename(columns = {'Ranks':'ObservedRanks', 'means': 'MeanRecall'}) 
    numcol=len(results_df.columns)
    results_df.insert(numcol, "RecRanks", list(np.float_(results.rec.ranks)) , True)
    
  #  R_obj=pandas2ri.py2ri(results_df)
    
    return results_df

