''' I systematically determine the best model to use based on splitting
    the data into train and test. I apply this first to the logistic regression algorthm.
    I then test different values (between 1-30)to use for K in the the 
    k-nearest neighbors (KNN) algorithm.'''
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

''' I load and store the iris data set into a variable,
    store the input into variable X,
    and the output in variable y  '''
iris = load_iris()
X= iris.data
y= iris.target
''' I split the data into train and test for both the input
    and output and store the returned values in variables'''
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=4)

''' I instantiate the LogisticRegression class, 
    then fit my output and input to the model.'''
logisticreg = LogisticRegression()
logisticreg.fit(X,y)
''' I use the test input to create a prediction, then compare 
    that prediction against the test set and print the accuracy
    value'''
y_pred = logisticreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

''' I use a for loop to test values 1-30 for the K value in the
    k-nearest neighbors algorithm.'''
y_lst= []
for i in range(1,31):
    ''' I instantiate the KNeighborsClassifier class,
    then fit my training output and training input to the knn model. 
    I use the test input to create a prediction.
    I then test that prediction against the test output,
    and append the accuracy value to a list.'''
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    y_lst.append(metrics.accuracy_score(y_test, y_pred))

''' I use the list containing the accuracy values to create a
    graph using matplotlib. The accuracy is displayed as y, and the K
    value is displayed as x.'''
plt.plot(range(1,31), y_lst)
plt.ylabel('Accuracy')
plt.xlabel('(K)')
plt.show()