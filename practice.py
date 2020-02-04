from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# submission data
sub = pd.read_csv('dataset/test.csv')
sub = sub.set_index('PassengerId')
sub.insert(0, 'Survived', 2)



# Adding train and test for preprocessing
data = data.append(sub)


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


# Ticket
data['TicketNumber'] = data['TicketNumber'].fillna(data['TicketPrefix'])
data['TicketPrefix'] = np.where(data['TicketPrefix'] == data['TicketNumber'], '', data['TicketPrefix'])


# Removing Unused Columns
data = data.drop(columns=['Name', 'Othernames', 'othernames', 'Cabin', 'lname', 'Othername', 'Ticket', 'TicketPrefix'])


submission = data.iloc[891:, :]
data = data.iloc[:891, :]

features = data.drop(columns=['Survived'])
submission = submission.drop(columns=['Survived'])
target = data.Survived.values


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=0)

numeric_features = ['Fare', 'SibSp', 'Parch', 'Age']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])


categorical_features = ['Sex', 'TicketNumber', 'title', 'Pclass', 'Alias',  'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# didn't use
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
         ], remainder='passthrough')


# build composite model
estimators = [
    ('SVM', SVC()),
    ('lr', LogisticRegression())
]
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
model = make_pipeline(preprocess, stacking_classifier)
model.fit(X_train, y_train)


cvb = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)
print(cvb.mean(), cvb.std()*2)

print("model score: %.3f" % model.score(X_train, y_train))
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))


test_rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("model score: %.3f" % model.score(X_test, y_test))


y_sub= model.predict(submission)


submission = pd.DataFrame({'PassengerId':submission.index,'Survived':y_sub})

#print(submission)

#filename = 'Predictions.csv'
#submission.to_csv(filename,index=False)

#print('Saved file: ' + filename)