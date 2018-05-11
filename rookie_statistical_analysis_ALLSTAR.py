import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import RFE, RFECV
import seaborn as sns
sns.set()
from sklearn.ensemble import RandomForestRegressor
from itertools import compress
from sklearn.cross_validation import train_test_split



def prepare_data(data):
    #dropping rank features and not useble columns
    features_drop_ranks = ['GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK', 'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK', 'NBA_FANTASY_PTS_RANK', 'DD2_RANK', 'TD3_RANK','CFPARAMS',"PLAYER_ID", "PLAYER_NAME",'TEAM_ABBREVIATION','CFPARAMS','CFID']
    data = data.drop(features_drop_ranks, axis=1)
    return data

def check_corr(data_train):
    plt.figure(figsize=(15, 6))
    sns.heatmap(data_train.corr())
    plt.show()


def accuracy(y_pred_binary, y_test):
    conf_matrix = confusion_matrix(y_pred_binary,y_test)
    print conf_matrix
    true_neg,false_pos,false_neg,true_pos = conf_matrix[1][1],conf_matrix[1][0],conf_matrix[0][1],conf_matrix[0][0]
    print "prediction of not-great players "+str(round(float(true_neg)/(true_neg+false_neg),2))
    print "prediction of great players "+str(round(float(true_pos)/(true_pos+false_pos),2))


def preform_regression(data_train,data_test):
    cols = ['TEAM_ID', 'AGE', 'GP', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'AST', 'TOV', 'STL', 'BLK',
            'BLKA', 'PF', 'PFD', 'PTS', 'DD2', 'TD3']
    #cols = ['TEAM_ID','AGE','GP','AST','OREB','PTS']
    X_train = data_train.drop(["ALL_STAR"],axis=1)
    y_train = data_train["ALL_STAR"]
    X_test = data_test
    y_test = data_test["ALL_STAR"]
    model  = LogisticRegression()
    rfe = RFE(model,5)
    fit = rfe.fit(X_train,y_train)
    col_select = list(compress(cols,fit.support_))
    logreg = LogisticRegression()
    logreg.fit(X_train[cols], y_train)
    y_pred = logreg.predict(X_test[cols])
    accuracy(y_pred,y_test)


def preform_rand_forest(data_train,data_test):
    X_train = data_train.drop(["ALL_STAR"],axis=1)
    y_train = data_train["ALL_STAR"]
    X_test = data_test.drop(["ALL_STAR"],axis=1)
    y_test = data_test["ALL_STAR"]
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(X_train, y_train)
    print "--------feature importances--------"
    for feature in zip(list(X_train.columns.values), rf.feature_importances_):
        print(feature)
    rfecv = RFECV(estimator=rf, step=1, cv=2, scoring='roc_auc', verbose=2)
    selector = rfecv.fit(X_train,y_train)
    predictions = selector.predict(X_test)
    y_pred_binary =predictions
    y_pred_binary[y_pred_binary >= 0.5] = 1
    y_pred_binary[y_pred_binary < 0.5] = 0
    accuracy(y_pred_binary,y_test)


rookie_data = pd.DataFrame.from_csv('rookies_allstar_labeled.csv')
rookie_data_prepared = prepare_data(rookie_data)
train, test = train_test_split(rookie_data_prepared, test_size=0.2)
check_corr(train)
#preform_regression(train,test)
#preform_rand_forest(train,test)










