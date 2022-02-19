

# --------------------
# Libraries
# --------------------

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycaret.datasets import get_data
from pycaret.regression import *

import requests

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "browser"


import numerapi
import sklearn.linear_model

import mlflow

mlflow.set_tracking_uri("file:///C:Users/marianota/Projects/numerai_framework/mlruns")


# ----------------
# Load data
# ----------------

# download the latest training dataset (takes around 30s)
# training_data = pd.read_csv("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz")
training_data = pd.read_csv("numerai_training_data.csv")
#training_data.head()

# download the latest tournament dataset (takes around 30s)
#  tournament_data = pd.read_csv("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz")


# -----------------------
# Exploratory analysis
# -----------------------
exploration = False
if exploration:
    df = training_data.copy()

    print(df.columns)

    print(df.shape)
    # Check number of eras
    eras_list = df.era.unique().tolist()
    print(eras_list)
    number_of_eras = len(eras_list)
    print(number_of_eras)

    # check number of ids
    ids_list = df.id.unique().tolist()
    print(ids_list)
    count_ids = len(ids_list)
    print(count_ids)

    # check number of entries by era and id
    df['Key'] = df['era'].astype(str) + '-' + df['id'].astype(str)
    Key_list = df.Key.unique().tolist()
    print(Key_list)
    count_Key = len(Key_list)
    print(count_Key)


    # number eras by id
    df['count_era'] =df.groupby(['era']).id.transform('count')


    # analize eras
    df['era_num'] = df['era'].str[3:].astype(int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.era_num,
                             y=df.count_era,
                             name='Era Analysis', mode='lines'))
    fig.show()

    df = df[df.era_num>90]


    from pycaret.datasets import get_data
    dataset = get_data('training_data', profile=True)


#-------------------------------------------------------
# Feature Engineering using feature engine library
#-------------------------------------------------------
feature_engine = False
if feature_engine:
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import AdaBoostRegressor
    import xgboost as xgb
    from feature_engine.selection import (
        DropConstantFeatures,
        DropDuplicateFeatures,
        DropCorrelatedFeatures,
        SmartCorrelatedSelection,
        SelectByShuffling,
        SelectBySingleFeaturePerformance,
        SelectByTargetMeanPerformance,
        RecursiveFeatureElimination,
        RecursiveFeatureAddition)
    from feature_engine.imputation import(
        CategoricalImputer,
        MeanMedianImputer)
    from feature_engine.encoding import (
        OneHotEncoder,
        CountFrequencyEncoder,
        OrdinalEncoder)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV


    # Define estimator
    estimator_xgb = xgb.XGBRegressor(
                    objective            = "reg:squaredlogerror",
                    booster              = "gbtree",
                    eval_metric          = "rmsle",
                    eta                  = 0.01,
                    max_depth            = 4,
                    subsample            = 0.5,
                    colsample_bytree     = 0.7,
                    n_estimators         = 100,
                    nrounds              = 560,
                    verbose              = 1
                        )

    estimator_rf = RandomForestRegressor(random_state=1)

    estimator_ada = AdaBoostRegressor(random_state=1)

    estimator = estimator_rf

    estimator2 = estimator_rf


    # Drop correlated
    correlated = DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.95)

    # Drop by shuffling
    drop_shuffling = SelectByShuffling(estimator = estimator,
                                scoring="neg_mean_squared_log_error", cv=10)

    # Smart correlation
    smart_correlation = SmartCorrelatedSelection(
            threshold        = 0.95,
            selection_method = "model_performance",
            estimator        = estimator,
            missing_values   = "raise",
            # selection_method = "variance",
            method           = "pearson"
            )

    # Select by single feature performance
    single_feature_performance = SelectBySingleFeaturePerformance(estimator = estimator2,
                                                           scoring="r2",
                                                           cv=10)

    # Select by target feature performance
    target_performance = SelectByTargetMeanPerformance(
                        scoring      = "r2_score",
                        cv           = 10,
                        random_state = 1)

    # Select by recursive feature elimination
    estimator = RandomForestRegressor(random_state=1)
    recursive = RecursiveFeatureElimination(estimator = estimator,
                                            scoring="neg_mean_squared_log_error", cv=10)

    # Select by feature addition
    addition = RecursiveFeatureElimination(estimator=estimator,
                                           scoring="neg_mean_squared_log_error", cv=10)

    # Feature selection pipeline
    pipe = Pipeline([
                     #('correlated', correlated),
                    #('smart_correlation'   , smart_correlation)
                     ('drop_shuffling'      , drop_shuffling)
                    #('single_feature_performance' , single_feature_performance)
                    #('target_performance'   , target_performance)
                    #('recursive'           , recursive)
                    #('addition'            , addition)
                   ])

# Apply pipeline tot he features
training_data = df.copy()
feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
features = training_data[feature_cols]
target = training_data['target']

# Find features using pipeline
#pipe.fit(features, target)
#features = pipe.transform(features)

data = pd.concat([features,target], axis=1)

data.head()


#-------------------------------------
# Pycaret feature engineering
#-------------------------------------

# This step includes feature engineering, cleaning data, prepare data for training
exp = setup(data,
                target = 'target',
                feature_selection = True,
                ignore_low_variance = True,
                remove_multicollinearity = True,
                log_experiment = True,
                pca = True,
                silent = True,
                experiment_name = 'sub2')

model = create_model('lightgbm')


#----------------------------
# Train model using pycaret
#----------------------------

# Find best model
best = compare_models()
best_result = pull()
print(best_result)

model = create_model('lightgbm')

best_specific = compare_models(include = ['dt','rf','xgboost'])

# The best model
tuned_model = tune_model(model)
model_tuned = pull()
print(model_tuned)

# Train a bagging classifier
bagged_dt = ensemble_model(tuned_model, method = 'Bagging', n_estimators = 100)

# Train a adaboost classifier on dt with 100 estimators
boosted_dt = ensemble_model(tuned_model, method = 'Boosting', n_estimators = 100)

# Train a voting classifier dynamically
blender_specific = blend_models(estimator_list = compare_models(sort = 'R2', n_select = 5))

# Final model using automl
final_model = automl()
results_final_model = pull()
print(results_final_model)


# ----------------------------
# Advance training models
# ----------------------------

huber = create_model('huber', verbose = False)
dt = create_model('dt', verbose = False)
lightgbm = create_model('lightgbm', verbose = False)
ridge = create_model('ridge', verbose = False)

blend_specific = blend_models(estimator_list = [huber,dt,lightgbm,ridge])


#-----------------------
# Model evaluation
#-----------------------

pred_test = predict_model(final_model)
holdout_score = pull()
print(holdout_score)

# rmsle
from math import sqrt
from sklearn.metrics import mean_squared_log_error

rmsle = sqrt(mean_squared_log_error(pred_test.target.values, pred_test.Label.values))
print('rmsle test: ', rmsle)


# Training data pred results
features = training_data[feature_cols]

# Predictions
pred_train = predict_model(model, data = training_data)

plt.hist(training_data.target)
plt.hist(pred_train.Label)


#---------------------------------
# Prepare submission
#---------------------------------
tournament_data = pd.read_csv("numerai_tournament_data.csv")

example_predictions = pd.read_csv("example_predictions.csv")

features = tournament_data[feature_cols]

# Predictions
pred_sub = predict_model(final_model, data = features)

pred_sub.rename(columns={'Label': 'prediction'}, inplace=True)

df_submission = pd.concat([tournament_data.id,pred_sub.prediction], axis=1)

df_submission.to_csv("submission.csv", index=False)


# -----------------
# Analyse results
# -----------------

TARGET_NAME = f"target"
PREDICTION_NAME = f"prediction"

feature_names = [
        f for f in tournament_data.columns if f.startswith("feature")
    ]
print(f"Loaded {len(feature_names)} features")


# Submissions are scored by spearman correlation
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


# convenience method for scoring
def score(df):
    return correlation(df[PREDICTION_NAME], df[TARGET_NAME])

# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)

# Check the per-era correlations on the training set (in sample)
train_correlations = pred_train.groupby("era").apply(score)
print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std(ddof=0)}")
print(f"On training the average per-era payout is {payout(train_correlations).mean()}")


"""Validation Metrics"""
# Check the per-era correlations on the validation set (out of sample)
validation_data = tournament_data[tournament_data.data_type == "validation"]

# filter closes values to train data

validation_data['era_value'] = validation_data['era'].str[3:].astype(int)

validation_data = validation_data[validation_data['era_value']<180]

validation_correlations = validation_data.groupby("era").apply(score)
print(f"On validation the correlation has mean {validation_correlations.mean()} and "
      f"std {validation_correlations.std(ddof=0)}")
print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")

# Check the "sharpe" ratio on the validation set
validation_sharpe = validation_correlations.mean() / validation_correlations.std(ddof=0)
print(f"Validation Sharpe: {validation_sharpe}")

print("checking max drawdown...")
rolling_max = (validation_correlations + 1).cumprod().rolling(window=100,
                                                              min_periods=1).max()
daily_value = (validation_correlations + 1).cumprod()
max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
print(f"max drawdown: {max_drawdown}")

# Check the feature exposure of your validation predictions
feature_exposures = validation_data[feature_names].apply(lambda d: correlation(validation_data[PREDICTION_NAME], d),
                                                         axis=0)
max_per_era = validation_data.groupby("era").apply(
    lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max())
max_feature_exposure = max_per_era.mean()
print(f"Max Feature Exposure: {max_feature_exposure}")


# tournament metrics

# Load example preds to get MMC metrics
example_preds = pd.read_csv("example_predictions.csv")
example_preds['Expred'] = example_preds.prediction

# merge
validation_data = pd.merge(left=validation_data,
                           right=example_preds[['id','Expred']],
                           how='left',
                           left_on='id',
                           right_on='id')


validation_data["ExamplePreds"] = validation_data.target
validation_data["ExamplePreds"] = validation_data.Expred


# ----------------------------
# Advance Metrics
# ----------------------------

""" 
functions used for advanced metrics
"""

# to neutralize a column in a df by many other columns on a per-era basis
def neutralize(df,
               columns,
               extra_neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    # need to do this for lint to be happy bc [] is a "dangerous argument"
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (pd.Series(x).rank(method="first").values - .5) / len(x)
                scores2.append(x)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


# to neutralize any series by any other series
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def get_feature_neutral_mean(df):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME],
                                          feature_cols)[PREDICTION_NAME]
    scores = df.groupby("era").apply(
        lambda x: correlation(x["neutral_sub"], x[TARGET_NAME])).mean()
    return np.mean(scores)

# Check feature neutral mean
print("Calculating feature neutral mean...")
feature_neutral_mean = get_feature_neutral_mean(validation_data)
print(f"Feature Neutral Mean is {feature_neutral_mean}")


print("calculating MMC stats...")
# MMC over validation
mmc_scores = []
corr_scores = []
for _, x in validation_data.groupby("era"):
    series = neutralize_series(pd.Series(unif(x[PREDICTION_NAME])),
                               pd.Series(unif(x["ExamplePreds"])))
    mmc_scores.append(np.cov(series, x[TARGET_NAME])[0, 1] / (0.29 ** 2))
    corr_scores.append(correlation(unif(x[PREDICTION_NAME]), x[TARGET_NAME]))

val_mmc_mean = np.mean(mmc_scores)
val_mmc_std = np.std(mmc_scores)
val_mmc_sharpe = val_mmc_mean / val_mmc_std
corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
corr_plus_mmc_mean = np.mean(corr_plus_mmcs)
corr_plus_mmc_sharpe_diff = corr_plus_mmc_sharpe - validation_sharpe

print(
    f"MMC Mean: {val_mmc_mean}\n"
    f"Corr Plus MMC Sharpe:{corr_plus_mmc_sharpe}\n"
    f"Corr Plus MMC Diff:{corr_plus_mmc_sharpe_diff}"
)

# Check correlation with example predictions
full_df = pd.concat([validation_data.Expred, validation_data[PREDICTION_NAME], validation_data["era"]], axis=1)
full_df.columns = ["example_preds", "prediction", "era"]
per_era_corrs = full_df.groupby('era').apply(lambda d: correlation(unif(d["prediction"]), unif(d["example_preds"])))
corr_with_example_preds = per_era_corrs.mean()
print(f"Corr with example preds: {corr_with_example_preds}")





