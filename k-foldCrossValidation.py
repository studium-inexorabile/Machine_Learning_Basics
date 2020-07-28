''' I use K-fold cross validation to shuffle my training and testing sets
    and calculate accuracy scores for the K value in the K-nearest 
    neighbors algorithm. '''
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

''' I load and store the iris data set into a variable,
    store the input into variable X,
    and the output in variable y  '''
iris = load_iris()
X= iris.data
y= iris.target

k_scores = []
for i in range(1,46):
    ''' I instantiate the KNeighborsClassifier class, and use values 
    1-46 for the K value. I supply the knn model, the output and the 
    input to cross validation, and derive an accuracy score. I then 
    append the mean of the scores for each K value to a list.'''
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

''' I use the mean scores of all the shuffled training / test sets
    to create a graph. The mean is displayed on the y-axis, and the K 
    value used for each test set is displayed on the x-axis'''
plt.plot(range(1,46), k_scores)
plt.ylabel('Mean for accuracy Scores')
plt.xlabel('K value for KNN Algorithm')
plt.show()