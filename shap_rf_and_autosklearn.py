# uncompyle6 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.20 (default, Sep  7 2024, 18:35:08) 
# [GCC 11.4.0]
# Embedded file name: /home/dbanerjee/example_for_debjeet/shap_rf_and_autosklearn.py
# Compiled at: 2026-06-29 11:57:20
# Size of source mod 2**32: 10820 bytes
"""
Created by: Rahul Sharma
Created on: 20.09.2023
Description: get the feature importance of all the features in a dataset and catalog them
This will be done using the basic sklearn classifier, an autosklearn RF classifier and the RF feature importance and the
SHAP feature importance. 
"""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.stats import randint
from matplotlib import pyplot as plt
import shap
from Importance import FeatureImportance
import openml, json, sys, autosklearn.classification, pickle, os

class ClassificationAlgorithms:

    @staticmethod
    def train_default_RandomForest_Classifier(datasetId, randomState, X_train, y_train, X_test, y_test) -> tuple:
        rf = RandomForestClassifier(random_state=randomState)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return (
         rf, accuracy, datasetId)

    @staticmethod
    def prepare_data_for_RF_importance(datapath, randomState):
        features = pd.read_csv((datapath + "/features.csv"), header=0, index_col=False)
        targets = pd.read_csv((datapath + "/targets.csv"), header=0, index_col=False)
        nan_indices, _ = np.where(features.isna())
        features.drop(nan_indices, inplace=True)
        targets.drop(nan_indices, inplace=True)
        targetColumnName = targets.columns[0]
        unique_targets = list(targets[targetColumnName].unique())
        le = LabelEncoder()
        targets = le.fit_transform(targets[targetColumnName])
        with open(datapath + "/categorical_indicator.json", "r") as jsonFile:
            cat_indicators = json.load(jsonFile)
        cat_cols = []
        cont_cols = []
        for key, value in cat_indicators.items():
            if value == True:
                cat_cols.append(key)
            else:
                cont_cols.append(key)
        else:
            for cat in cat_cols:
                features[cat] = features[cat].astype("category").cat.codes
            else:
                featureNames = np.array(list(features))
                X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=randomState)
                return (X_train, X_test, y_train, y_test, featureNames, nan_indices)

    @staticmethod
    def prepare_data_for_SHAP_importance(X, y, datapath, cat_indicators, randomState, datasetId=23, suite_id=99):
        originalFeatureNames = X.columns
        targetColumn = y.columns
        nan_indices, _ = np.where(X.isna())
        X.drop(nan_indices, inplace=True)
        y.drop(nan_indices, inplace=True)
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        assert len(targetColumn) == 1, f"Target variables > 1 for {datapath}"
        numClasses = y[targetColumn[0]].nunique()
        classes = y[targetColumn[0]].unique()
        cat_cols = []
        cont_cols = []
        for key, value in cat_indicators.items():
            if value == True:
                cat_cols.append(key)
            else:
                cont_cols.append(key)
        else:
            for cat in cat_cols:
                X[cat] = X[cat].astype("category").cat.codes
            else:
                y[targetColumn[0]] = y[targetColumn[0]].astype("category").cat.codes
                X_categorical = X.drop(cont_cols, axis=1)
                enc = OneHotEncoder()
                enc.fit(X_categorical)
                X_encoded = enc.transform(X_categorical).toarray()
                cat_feature_names = X_categorical.columns
                new_cat_feature_names = enc.get_feature_names(cat_feature_names)
                X_encoded = pd.DataFrame(X_encoded, columns=new_cat_feature_names)
                if not X_encoded.empty:
                    X_for_SHAP = pd.concat([X_encoded, X.drop(cat_cols, axis=1)], axis=1)
                else:
                    X_for_SHAP = X
                y = np.ravel(y)
                X_train, X_test, y_train, y_test = train_test_split(X_for_SHAP, y, test_size=0.2, random_state=randomState)
                print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
                return (X_train, X_test, y_train, y_test, X_categorical, originalFeatureNames, cat_cols, X_for_SHAP.columns, new_cat_feature_names, classes, nan_indices)

    @staticmethod
    def compute_and_save_autosklearn_models(datapath, did, randomState):
        """We get all the model configurations for every dataset.
        The following steps seem to make sense. 

        1. For each binary classification dataset
            - make the train test split using randomState = 42
            - make the folder for the current dataset where all the pipelines data will be saved 
            - run autosklearn classifier on the dataset for an hour and save all the runs
        """
        datasetPath = datapath + str(did)
        X_train, X_test, y_train, y_test, X_cat, original_featureNames, original_catFeatureNames, shap_featureNames, shap_catfeatureNames, classes, nan_indices_ = ClassificationAlgorithms.prepare_data_for_SHAP_importance(datasetPath, randomState, datasetId=(int(did)))
        mainLogDir = "/scratch/rsharma/p-hacking/src/rahulsStuff/multiobjectiveOptim/Thesis/uai_rebuttal/rebuttal_pipelines/"
        logFolder = mainLogDir + str(did)
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=43200, per_run_time_limit=3600,
          tmp_folder=logFolder,
          delete_tmp_folder_after_terminate=False,
          n_jobs=1,
          initial_configurations_via_metalearning=25,
          ensemble_kwargs={"ensemble_size": 1},
          memory_limit=10240,
          seed=randomState)
        automl.fit(X_train, y_train)

    @staticmethod
    def get_shap_for_all_autosklearn_models(datasetPath, pipelines_for_did, did, randomState):
        """
        1. For all the datasets
            -> for all the models configs for that dataset
                - Take the pipeline and fit the data to that pipeline
                - Get the accuracy of that model
                - run the shap value calculation for each of them with nsamples of 50
                - get the sum of shap (not in an absolute manner) and see for the top-k values which have been flipped.

        """
        for pipeline in pipelines_for_did:
            try:
                X_train, X_test, y_train, y_test, X_cat, original_featureNames, original_catFeatureNames, shap_featureNames, shap_catfeatureNames, classes, nan_indices_ = ClassificationAlgorithms.prepare_data_for_SHAP_importance(datasetPath, randomState, datasetId=(int(did)))
                with open(pipeline, "rb") as fh:
                    model = pickle.load(fh)
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)
                accuracy = accuracy_score(y_hat, y_test)
                origFeature_abs_mean_shap, sum_one_class_shap, origFeature_abs_mean_shap_subcat, sum_one_class_shap_subcat = FeatureImportance.get_abs_SHAP_feature_importance(model, X_train, X_test, X_cat, original_featureNames, original_catFeatureNames, shap_featureNames, shap_catfeatureNames)
                saveMainDir = "/netscratch/rsharma/p-hacking/src/rahulsStuff/experiments/source/data/shap_values_autosklearn_configs_Xtest_50/"
                savePath = saveMainDir + str(did)
                config_id = pipeline[(pipeline.rfind("/") + 1)[:None]]
                if not os.path.exists(savePath):
                    os.mkdir(savePath)
                featureImportances = {'configPath':pipeline, 
                 'config_id':config_id, 
                 'accuracy':accuracy, 
                 'origFeature_abs_mean_shap':origFeature_abs_mean_shap, 
                 'sum_one_class_shap':sum_one_class_shap, 
                 'origFeature_abs_mean_shap_subcat':origFeature_abs_mean_shap_subcat, 
                 'sum_one_class_shap_subcat':sum_one_class_shap_subcat, 
                 'nan_indices':(nan_indices_.tolist)()}
                with open(savePath + "/" + str(config_id) + ".json", "w+") as savejson:
                    json.dump(featureImportances, savejson)
            except Exception as e:
                try:
                    print(e)
                    print(f"Error encountered for dataset id: {did}")
                finally:
                    e = None
                    del e

# okay decompiling __pycache__/shap_rf_and_autosklearn.cpython-38.pyc
