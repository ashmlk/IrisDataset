#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import pandas as pd
from pandas.plotting import scatter_matrix
iris = pd.read_csv('D:\My Projects\Python Projects\Iris\iris.csv')
fig, ax = plt.subplots()
#Scatter ploth againt the sepal length and sepal width
ax.scatter(iris['SepalLengthCm'], iris['SepalWidthCm'])
ax.set_title('Iris')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
#Different Classification models
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
X = iris.iloc[:, :-1].values
Y = iris.iloc[:, -1].values
#Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Using LR(Logistics regression) 
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
prediction = classifier.predict(X_test)

#summary of predictions mades by classifier
print(classification_report(Y_test,prediction))
print(confusion_matrix(Y_test, prediction))

#Accuracy score of classifier
from sklearn.metrics import accuracy_score
print('accuracy is: ', accuracy_score(prediction,Y_test))

#plotting the predictions
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.scatter(Y_test, prediction)
plt.show()

#Using kNN classifier
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(X_train, Y_train)
prediction = classifier.predict(X_test)

#summary of predictions mades by classifier
print(classification_report(Y_test,prediction))
print(confusion_matrix(Y_test, prediction))

#Accuracy score of classifier
from sklearn.metrics import accuracy_score
print('accuracy is: ', accuracy_score(prediction,Y_test))

#plotting the predictions
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.scatter(Y_test, prediction)
plt.show()

#Using DT(Decision Tree) classifier
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)
prediction = classifier.predict(X_test)

#summary of predictions mades by classifier
print(classification_report(Y_test,prediction))
print(confusion_matrix(Y_test, prediction))

#Accuracy score of classifier
from sklearn.metrics import accuracy_score
print('accuracy is: ', accuracy_score(prediction,Y_test))

#plotting the predictions
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.scatter(Y_test, prediction)
plt.show()


iris_data = iris.drop(['Id'], axis = 1)
graph = sns.pairplot(iris_data, hue='Species', markers='x')
graph = graph.map_upper(plt.scatter)
graph = graph.map_lower(sns.kdeplot)

#%%
