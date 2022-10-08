from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools

def split_dataset(inp_df):
    grain_index = inp_df.index 
    train_len = math.floor(len(grain_index)  * 0.9)
    train_index = grain_index[:train_len]
    val_index = grain_index[train_len:]
    train_data = inp_df.loc[train_index,:]
    val_data = inp_df.loc[val_index,:]
    
    train_x = train_data.drop(columns = ['units'])
    train_y = train_data['units']
    val_x = val_data.drop(columns = ['units'])
    val_y = val_data['units']
    return train_x, train_y, val_x, val_y


def calc_rmsle(predictions, observations):
    total = 0 
    ## for all the predicted and actual values
    for i in range(len(predictions)):
        
        ## Take log of predicted values and add 1 to prevent underflow
        pred= np.log1p(predictions[i]+1)
        ## Take log of actual values and add 1 to prevent underflow
        targ = np.log1p(observations[i] + 1)
        
        ## if predicted and actual values are not a number then don't go 
        if not (math.isnan(pred)) and  not (math.isnan(targ)): 
            ## summation of squares of difference of logarithmic predicted and target value
            total = total + ((pred-targ) **2)
        else:
            raise Exception("predicted or actual value is not a number! | predicted : {}, target : {}".format(pred,targ))
    
    ## taking mean of the summated value
    total = total / len(predictions)        
    ## Performing square root of that value
    return np.sqrt(total)


def tune_decision_tree_regressor(hyper_params, train_x, train_y, val_x, val_y):
    min_rmsle = np.inf
    best_depth = -1
    best_min_samples = -1
    best_max_leaf_nodes = -1
    best_hyper_params = dict()
    
    max_depth_params = hyper_params['max_depth_params']
    min_samples_split_params = hyper_params['min_samples_split_params']
    max_leaf_nodes_params = hyper_params['max_leaf_nodes_params']
    
    for depth in max_depth_params:
        for min_samples in min_samples_split_params:
            for max_leafs in max_leaf_nodes_params:
                regressor = DecisionTreeRegressor(max_depth=depth,
                                                  min_samples_split = min_samples,
                                                  max_leaf_nodes = max_leafs)
                regressor.fit(train_x, train_y)
                val_pred = regressor.predict(val_x)
                rmsle = calc_rmsle(val_y.values,val_pred)
                if rmsle < min_rmsle:
                    best_hyper_params['best_depth'] = depth
                    best_hyper_params['best_min_samples'] = min_samples
                    best_hyper_params['best_max_leaf_nodes'] = max_leafs
                    min_rmsle = rmsle
    
    print("Best depth to train decision tree regressor : ",best_hyper_params['best_depth'])
    print("Best minimum number of samples to train decision tree regressor : ",best_hyper_params['best_min_samples'])
    print("Best maximum leaf nodes to train decision tree regressor : ",best_hyper_params['best_max_leaf_nodes'])
    print("Best RMSLE observed while training decision tree regressor : ",min_rmsle)
    regressor = DecisionTreeRegressor(max_depth = best_hyper_params['best_depth'],
                                      min_samples_split = best_hyper_params['best_min_samples'],
                                      max_leaf_nodes = best_hyper_params['best_max_leaf_nodes'])
    regressor.fit(train_x, train_y)
    predictions = regressor.predict(val_x)
    return best_hyper_params, min_rmsle, regressor, predictions


def get_feature_importances(inp_x, regressor):
    feature_columns = inp_x.columns
    feature_importances = regressor.feature_importances_
    feat_imp_dict = dict(zip(feature_columns,feature_importances))
    feat_imp_df = pd.DataFrame()
    feat_imp_df['cols'] = feat_imp_dict.keys()
    feat_imp_df['imp_vals'] = feat_imp_dict.values()
    fig,ax1 = plt.subplots(nrows = 1,ncols = 1,figsize=(20,10))
    plt.xticks(rotation = 45)
    sns.barplot(x = 'cols', y = 'imp_vals', data = feat_imp_df,ax=ax1)

    
def check_weather_conditions(x,is_mild=True):
    if is_mild:
        mild_weather_cond = ['DZ','RA','SN','SG','BR','HZ']
        for weather_cond in mild_weather_cond:
            if weather_cond in x:
                return 1
            else:
                return 0 
    else:
        extreme_weather_cond = ['TS','FC','SS','DS','FG','VA','GR','FU','DU']
        for weather_cond in extreme_weather_cond:
            if weather_cond in x:
                return 1
            else:
                return 0
            
            
def plot_observations_predictions(observations, predictions):
    plt.rc("figure", figsize=(25, 8))
    plt.rc("font", size=12)

    plt.plot(observations,color="blue",label='observations')
    plt.plot(predictions, color='red',label='predictions')
    plt.title("Plot of Predictions and Observed values")
    plt.legend(loc="upper right")
    plt.xlabel("time")
    plt.ylabel("units")
    
    plt.show()
    
    
def evaluate_rf_regressor(dict_of_hyper_params, train_x, train_y, val_x, val_y, index):
    random_forest_regressor = RandomForestRegressor(**dict_of_hyper_params)
    random_forest_regressor.fit(train_x, train_y)
    val_pred = random_forest_regressor.predict(val_x)

    rmsle = calc_rmsle(val_y.values,val_pred)
    print("Completed execution for index : ",index)
    return (index,rmsle)


def tune_rf_regressor(rf_hyper_params,train_x, train_y, val_x, val_y):
    min_rmsle = np.inf
    best_hyper_params = None

    hp_list = [val for val in rf_hyper_params.values()]
    hp_combination = list(itertools.product(*hp_list))
    hp_keys = tuple(rf_hyper_params.keys())


    rmsle_list = Parallel(n_jobs=6,verbose=1)(delayed(evaluate_rf_regressor)
              (dict(zip(hp_keys,hp_combination[hp_index])),
               train_x, train_y, val_x, val_y, hp_index) 
               for hp_index in range(len(hp_combination)))

    for index in range(len(rmsle_list)):
        if rmsle_list[index][1] < min_rmsle:
            min_rmsle = rmsle_list[index][1]
            best_hyper_params = dict(zip(hp_keys,hp_combination[rmsle_list[index][0]]))

    rf_regressor = RandomForestRegressor(**best_hyper_params)
    rf_regressor.fit(train_x, train_y)
    predictions = rf_regressor.predict(val_x)  
    print("Best Hyper Params : ", best_hyper_params)
    print("Min RMSLE : ", min_rmsle)
    return best_hyper_params, min_rmsle, rf_regressor, predictions


def calc_new_preds(preds,diff):
    mean_prediction = np.mean(preds)
    new_preds = preds.copy()
    for i in range(len(preds)):
        if preds[i] < mean_prediction:
            new_preds[i] = preds[i]-diff
    return new_preds


def adjust_predictions(grain, preds, val_y):
    mean_prediction = np.mean(preds)
    diff_param = [i for i in range(5,20)]
    best_diff_param = 0
    min_rmsle = np.inf
    for diff in diff_param:
        try:
            new_predictions = calc_new_preds(preds,diff)
            rmsle = calc_rmsle(new_predictions, val_y.values)
            if rmsle < min_rmsle :
                best_diff_param = diff
                min_rmsle = rmsle
        except:
            print("Couldn't find best_diff_param for grain : {} and diff_value : {}".format(grain, diff))
            continue
        

    print("Minimum RMSLE : ", min_rmsle)
    print("Best difference parameter : ", best_diff_param)
    new_predictions = calc_new_preds(preds,best_diff_param)
    
    return new_predictions, best_diff_param, min_rmsle