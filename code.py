# --------------
# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Code starts here
df = pd.read_json(path,lines=True)
#print(df.head())
print(type(df.columns))
df.columns = df.columns.str.replace(' ','_')
missing_data = df.isnull()
df = df.drop(columns = ['waist', 'bust', 'user_name','review_text','review_summary','shoe_size','shoe_width'])
X = df.drop(columns=['fit'])
y = df['fit']
#print(X.head())
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state = 6)

# Code ends here


# --------------
def plot_barh(df,col, cmap = None, stacked=False, norm = None):
    df.plot(kind='barh', colormap=cmap, stacked=stacked)
    fig = plt.gcf()
    fig.set_size_inches(24,12)
    plt.title("Category vs {}-feedback -  cloth {}".format(col, '(Normalized)' if norm else ''), fontsize= 20)
    plt.ylabel('Category', fontsize = 18)
    plot = plt.xlabel('Frequency', fontsize=18)

# Code starts here
g_by_category = df.groupby(['category'])
cat_fit = g_by_category['fit'].value_counts()
#print(type(cat_fit))
cat_fit = cat_fit.unstack()
plot_barh(cat_fit,'fit')
# Code ends here


# --------------
# Code starts here
cat_len = g_by_category['length'].value_counts()
#print(cat_len)
cat_len = cat_len.unstack()
plot_barh(cat_len,'length')
# Code ends here


# --------------
# Code starts here
def get_cms(x):
    if type(x) == type(1.0):
        return
    try:
        return (int(x[0])*30.48) + (int(x[4:-2])*2.54)
    except:
        return (int(x[0])*30.48)
    return x
#print(X_train['height'])
#print(X_train.isnull().sum())
X_train.height = X_train['height'].apply(lambda x: get_cms(x))
X_test.height = X_test['height'].apply(lambda x:get_cms(x))
# if(pd.notnull(x)) else x )
#if len(x[4:-2])! =0 else)
#print(X_train['height'])
# Code ends here


# --------------
# Code starts here

X_train[['height','length','quality']] = X_train[['height','length','quality']].dropna(axis = 1,inplace=True)
X_test[['height','length','quality']] = X_test[['height','length','quality']].dropna(axis = 1,inplace=True)
y_train = y_train.drop(index = y_train[y_train.index != X_train.index],axis=0)
y_test = y_test.drop(index = y_test[y_test.index != X_test.index],axis=0)
print(X_train.isnull().sum())
X_train[['bra_size','hips']] = X_train[['bra_size','hips']].fillna(X_train[['bra_size','hips']].mean())
X_test[['bra_size','hips']] = X_test[['bra_size','hips']].fillna(X_test[['bra_size','hips']].mean())
mode_1 = X_train['cup_size'].mode()[0]
mode_2 = X_test['cup_size'].mode()[0]
X_train[['cup_size']] = X_train[['cup_size']].fillna(mode_1)
X_test[['cup_size']] = X_test[['cup_size']].fillna(mode_2)
print(X_train.isnull().sum())
# Code ends here


# --------------
# Code starts here
X_train = pd.get_dummies(data=X_train,columns=["category","cup_size","length"],prefix = ["category","cup_size","length"])
X_test = pd.get_dummies(data=X_test,columns=["category","cup_size","length"],prefix = ["category","cup_size","length"])
#X_train.drop(['height','quality'],axis=1, inplace=True)
#X_test.drop(['height','quality'],axis=1, inplace=True)
print(X_train.info())
print(X_test.info())
# Code ends here


# --------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# Code starts here
model = DecisionTreeClassifier(random_state = 6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test ,y_pred)
precision = precision_score(y_test,y_pred,average=None)
print(score,precision)
# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# parameters for grid search
parameters = {'max_depth':[5,10],'criterion':['gini','entropy'],'min_samples_leaf':[0.5,1]}

# Code starts here
model = DecisionTreeClassifier(random_state=6)
grid = GridSearchCV(estimator=model,param_grid=parameters)
grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
# Code ends here


