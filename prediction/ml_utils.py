import pickle
import pandas as pd
import numpy as np
import sklearn
import collections
from sklearn.linear_model import Ridge, RidgeCV, RidgeClassifier, LogisticRegression, Lasso
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, normalize

data_path = '../resources/prediction_answer.csv'
ml_model_path = '../models/SVC_rbf_model'
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
float_to_engagement = {
    0: "Disengaged",
    0.33: "Barely engaged",
    0.66: "Engaged",
    1: "Highly engaged"
}


def make_data_mean_agg(data_path=data_path, emotions=emotions):
    df = pd.read_csv(data_path, names=emotions).mean()
    return df


def define_engagement(x, model=ml_model_path):
    model = pickle.load(open(model, 'rb'))
    try:
        res = model.predict(x.to_numpy().reshape(1, -1))
        # res_round = np.argmin(abs(res - deg) for deg in float_to_engagement.keys())
        res_round = [i for i in float_to_engagement.keys()
                     if abs(res - i) == min(abs(res - j) for j in float_to_engagement.keys())][0]
        with open('../flask_app/model_results/result.txt', 'w') as f:
            f.write(float_to_engagement[res_round])
        return float_to_engagement[res_round]
    except:
        res = 'It seems the video contains no faces. Engagement level cannot be understood...'
        with open('../flask_app/model_results/model_results/result.txt', 'w') as f:
            f.write(res)
        return res
