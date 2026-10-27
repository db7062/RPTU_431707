"""
Created by: Rahul Sharma
Created on: 22.09.2023 at 15:46
Description: This script houses the feature importance algorithms
"""

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FeatureImportance():

    @staticmethod
    def get_RF_feature_importance(classifier:RandomForestClassifier, featureNames, X_test, y_test, topk = -1)->tuple:
        
        perm_importance = permutation_importance(classifier, X_test, y_test)
        mean_importance = perm_importance.importances_mean
        sorted_idx = perm_importance.importances_mean.argsort() #this is sorted in ascending order. 
        sorted_feature_names = featureNames[sorted_idx]
        sorted_mean_importance = mean_importance[sorted_idx]
        sorted_feature_and_importance = [(sorted_feature_names[i], sorted_mean_importance[i]) for i in range(len(sorted_mean_importance))]

        #select the top k features by importance
        topk_features_by_importance = sorted_feature_and_importance.reverse() #this is sorted in descending order

        if topk != -1:
            topk_features_by_importance = sorted_feature_and_importance[0:topk]
        else:
            topk_features_by_importance = sorted_feature_and_importance

        return (topk_features_by_importance, perm_importance, mean_importance, sorted_idx)
    

    @staticmethod
    def get_abs_SHAP_feature_importance(classifier, X_train, X_test, X_cat, original_featureNames, original_catFeatureNames, shap_allfeatureNames, shap_catFeatureNames):
        
        #explainer = shap.TreeExplainer(classifier)
        #shap_values = explainer.shap_values(X_test, y_test)
        explainer = shap.KernelExplainer(classifier.predict_proba, shap.sample(X_train, 100), keep_index=True, keep_index_ordered=True)
        #shap_values = explainer.shap_values(shap.sample(X_test, 50), nsamples="auto")
        TEST_SAMPLE_SIZE = 100
        X_test_sample = shap.sample(
                    X_test,
                    TEST_SAMPLE_SIZE,
                    random_state=42
)

        shap_values = explainer.shap_values(X_test_sample, nsamples="auto",silent=True)
        
        
        """
        The SHAP package will give a list of list of shape (num_classes, num_rows_in_testData, num_features)
        The num features will represent the number of features in the dataset prepared for SHAP calculation.
        """
        
        n_categories = []
        for feat in original_catFeatureNames:
            #finding out the number of unique categories in each categorical variable
            n = X_cat[feat].nunique()
            n_categories.append(n)

        
        """
        The strategy to calculate the SHAP values for finding out the top k features is as follows:
        1. Get the shap values for the features
        2. For each class there will be a shap value. What I am thinking is that get the corresponding absolute
        SHAP value from each class for the same data point and average it. That will be the final absolute
        mean shap for the feature and plot it. 
        3. Alternatively I can do the absolute mean shap calculation for each class separately and see for each
        class how the abs mean SHAP value is. 

        """
        cat_summed_SHAP_by_Class_with_remaining = []

        #for shap values based on class categories
        for classShap in shap_values:

            summed_categorical_shap = []
            categorical_shap_values = classShap[:, :len(shap_catFeatureNames)]
            remaining_shap_values   = classShap[:, len(shap_catFeatureNames):]

            for values in categorical_shap_values:
                values_split = np.split(values, np.cumsum(n_categories))[:-1]
                values_sum = [sum(l) for l in values_split]
                summed_categorical_shap.append(values_sum)
            
            summed_categorical_shap = np.array(summed_categorical_shap)

            all_shap_values_for_this_class = np.concatenate((summed_categorical_shap, remaining_shap_values), axis=1)
            cat_summed_SHAP_by_Class_with_remaining.append(all_shap_values_for_this_class)
        

        ############ dataframe for shap values by class for each feature ###############

        """Here I save two data frames where 
        1. cat_summed_SHAP_by_class will have a summed value of shap for one-hot encoded sub-category for each category
        2. SHAP_by_class will have not summed values of shap for the categorical features (original shap values)
        3. ***Right now I will return the abs summed shap values over all the classes for each data point for each feature***
        """

        cat_summed_SHAP_by_Class_np = np.array(cat_summed_SHAP_by_Class_with_remaining)

        #next line gets abs sums all the values of shap through all the classes and averages it
        all_classes_shap = cat_summed_SHAP_by_Class_np.reshape((-1, cat_summed_SHAP_by_Class_np.shape[-1]))
        abs_mean_all_classes_shap = np.abs(all_classes_shap).sum(axis=0) / len(X_test_sample)
        abs_mean_shap_over_all_values_by_feature = {}

        for i in range(len(original_featureNames)):
            abs_mean_shap_over_all_values_by_feature[original_featureNames[i]] = abs_mean_all_classes_shap[i]

        #arrange the features by decreasing abs mean of shap values over the test data
        abs_mean_shap_over_all_values_by_feature = sorted(abs_mean_shap_over_all_values_by_feature.items(), key=lambda x:x[1], reverse=True)



        ################################################## Sum all classes shap no absolute for one class
        one_class_shap = cat_summed_SHAP_by_Class_np[0]
        sum_one_class_shap = one_class_shap.sum(axis=0) / len(X_test_sample)

        sum_one_class_shap_by_feature = {}
        for i in range(len(original_featureNames)):
            sum_one_class_shap_by_feature[original_featureNames[i]] = sum_one_class_shap[i]
        
        #arrange the features by decreasing abs mean of shap values over the test data
        sum_one_class_shap_by_feature = sorted(sum_one_class_shap_by_feature.items(), key=lambda x:x[1], reverse=True)
        


        ################################################## getting the features sorted by shap value without any summation across the sub-category for a categorical feature
        abs_mean_shap_over_all_values_by_feature_with_sub_cat = {}

        all_classes_shap_with_sub_cat = np.array(shap_values).reshape((-1, np.array(shap_values).shape[-1]))
        abs_mean_all_classes_shap_with_sub_cat = np.abs(all_classes_shap_with_sub_cat).sum(axis=0) / len(X_test_sample)
        for i in range(len(shap_allfeatureNames)):
            abs_mean_shap_over_all_values_by_feature_with_sub_cat[shap_allfeatureNames[i]] = abs_mean_all_classes_shap_with_sub_cat[i]
        
        abs_mean_shap_over_all_values_by_feature_with_sub_cat = sorted(abs_mean_shap_over_all_values_by_feature_with_sub_cat.items(), key=lambda x:x[1], reverse=True)



        #TODO: Sum all classes shap with sub categories no absolute for one class (done)
        sum_one_class_shap_by_feature_with_sub_cat = {}

        one_class_shap_with_sub_cat = np.array(shap_values[0]).reshape((-1, np.array(shap_values).shape[-1]))
        sum_one_class_shap_with_sub_cat = np.abs(one_class_shap_with_sub_cat).sum(axis=0) / len(X_test_sample)
        for i in range(len(shap_allfeatureNames)):
            sum_one_class_shap_by_feature_with_sub_cat[shap_allfeatureNames[i]] = sum_one_class_shap_with_sub_cat[i]
        
        sum_one_class_shap_by_feature_with_sub_cat = sorted(sum_one_class_shap_by_feature_with_sub_cat.items(), key=lambda x:x[1], reverse=True)
        return abs_mean_shap_over_all_values_by_feature, sum_one_class_shap_by_feature, abs_mean_shap_over_all_values_by_feature_with_sub_cat, sum_one_class_shap_by_feature_with_sub_cat
    
        

        
        
        #return shap.summary_plot(shap_values, X_test, class_names=classes, plot_type="bar")
        #return shap.summary_plot(cat_summed_SHAP_by_Class, X_test_rf, class_names=classes, plot_type="bar")

        
        






        



        

        
        
    

    def calculateSHAP():
        pass
    

    @staticmethod
    def plot_perm_feature_importance(feature_names, perm_importance, sorted_idx):

        plt.barh(feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
        plt.xlabel("Permutation Importance")
        plt.show()