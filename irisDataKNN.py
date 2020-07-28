''' I use the k-nearest neighbors (KNN) algorithm to predict the species
    of an iris given the sepal length, sepal width, petal length, and 
    petal width.'''
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

''' I load and store the iris data set into a variable,
    store the input into variable X,
    and the output in variable y  '''
iris = load_iris()
X=iris.data
y=iris.target

''' I instantiate the KNeighborsClassifier class, 
    and supply it values 1-30 for the K value.
    I fit my output and input to the knn model, 
    and run a prediction with two observations.
    I then print the results in a string that 
    displays the K value and the prediction.'''
for i in range(1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X,y)
    prediction = knn.predict([[5.1,3.5,1.4,0.2],[6.3,3.3,4.7,1.6]])
    print("n_neighbors=%i predicts: %s" % (i, prediction)) 