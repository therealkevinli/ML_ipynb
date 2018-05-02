#example from this: https://www.youtube.com/watch?v=1i0zu9jHN6U
import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True) #this is to treat these as outliers when there is missing data
df.drop(['id'],1,inplace = True) #drop the id column

#df.dropna(inplace=True) #this will just drop all the missing stuff

#X = np.array(df.drop(['class'],1)) #define X as the features
    #we include everything in the features except the class column
y = np.array(df['class']) #define y as labels (class)

#df.drop(['class'],1,inplace = True)
print(type(df))
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
print(df.columns)
X = np.array(df.drop(['class'],1)) #define X as the features
print(len(X))

#the test size is 20 percent
#quickly shuffle the data and seperate it into training and testing chunks
X_train, X_test, y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train) #fit the xtrain and ytrain

accuracy = clf.score(X_test,y_test)

print(accuracy)

example_measures = np.array([[4,2,1,2,2,2,3,2,1],[4,2,1,5,1,2,3,2,1]]) #don't forget 2d list,
#even if you have only one list
#example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)
print(prediction)

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))
