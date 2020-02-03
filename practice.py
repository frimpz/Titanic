from sklearn.datasets import load_digits, fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
import pandas as pd
import numpy as np
import sys

from sklearn.svm import SVC

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize)


# import data
data = pd.read_csv('dataset/train.csv')
data = data.set_index('PassengerId')


# Splitting name
data[['lname', 'Othernames']] = data.Name.str.split(',', 1, expand=True)
data[['title', 'othernames']] = data.Othernames.str.split('.', 1, expand=True)
data[['Othername', 'Alias']] = data.othernames.str.split('(', 1, expand=True)
data[['TicketPrefix', 'TicketNumber']] = data.Ticket.str.rsplit(' ', 1, expand=True)


# Replacing null values
# Alias
data['Alias'].replace(np.nan, 777, inplace=True)
data['Alias'] = np.where(data['Alias'] == 777, 0, 1)
# Age
data['Age'] = data.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
# Embarked
data['Embarked'] = data['Embarked'].fillna('Unkwnown')


# cleaning
#data['lname'] = 'bnn'


# Ticket
data['TicketNumber'] = data['TicketNumber'].fillna(data['TicketPrefix'])
data['TicketPrefix'] = np.where(data['TicketPrefix'] == data['TicketNumber'], '', data['TicketPrefix'])


#print(type(data['Cabin'][2]))
#print(data.isnull().sum())

# Removing Unused Columns
data = data.drop(columns=['Name', 'Othernames', 'othernames', 'Cabin', 'lname', 'Othername', 'Ticket', 'TicketPrefix'])

features = data.drop(columns=['Survived'])
target = data.Survived.values


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.30, random_state=0)

numeric_features = ['Fare', 'SibSp', 'Parch', 'Age']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['Sex' ,'TicketNumber', 'title', 'Pclass', 'Alias',  'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


str_transformer = Pipeline(steps=[
    ('TFIDF', TfidfVectorizer(stop_words=None, lowercase=True, strip_accents=None,
                                decode_error='replace',
                                encoding='utf-8', analyzer='word',
                                #token_pattern='(?u)\b\w\w+\b',
                                token_pattern='[a-zA-Z]',
                                ngram_range=(1, 2), max_features=100))])


preprocess = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        #('Othername', str_transformer, 'Othername'),
        #('lname', str_transformer, 'lname')
         ], remainder='passthrough')


model = make_pipeline(
    preprocess,
    SVC())

cvb = cross_val_score(model, X_train, y_train, scoring='accuracy', cv = 10)

print(cvb.mean(),cvb.std()*2)
#print("model score: %.3f" % model.score(X_train, y_train))

# SVC
# LogisticRegression()
# RandomForestClassifier
# ExtraTreesClassifier

#model.fit(X_train, y_train)


#y_train_pred = model.predict(X_train)
#y_pred = model.predict(X_test)

x = features

for column in x.columns:
    if x[column].dtype == type(object):
        le = LabelEncoder()
        x[column] = le.fit_transform(x[column])

#print(x.head(3))

#rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=12, max_depth=3, n_jobs=-1)
#rf.fit(features, target)
#print(1-rf.oob_score_)

#importance = rf.feature_importances_
#print(importance)



#train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
#test_rmse = np.sqrt(mean_squared_error(y_pred, y_test))
#print('Train RMSE: %.4f' % train_rmse)
#print('Test RMSE: %.4f' % test_rmse)
#print("model score: %.3f" % model.score(X_test, y_test))
#print("model score: %.3f" % model.score(X_train, y_train))
#print (model[1].feature_importances_)

