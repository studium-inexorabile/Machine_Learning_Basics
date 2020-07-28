''' I use the logistic regression algorithm to predict the species of an
    iris given the sepal length, sepal width, petal length, and 
    petal width.'''
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

''' I load and store the iris data set into a variable,
    store the input into variable X,
    and the output in variable y  '''
iris = load_iris()
X=iris.data
y=iris.target

''' I instantiate the LogisticRegression class, 
    fit my output and input to the model, 
    and run a prediction with two observations.
    I then print the results.'''
logisticreg = LogisticRegression()
logisticreg.fit(X,y)
prediction_lr = logisticreg.predict([[5.1,3.5,1.4,0.2],[6.3,3.3,4.7,1.6]])
print(prediction_lr)