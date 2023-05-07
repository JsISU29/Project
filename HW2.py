import numpy as np 
import matplotlib.pyplot as plt

import numpy # import again 
import matplotlib.pyplot # import again 

import numpy.linalg 
import numpy.random

import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def generate_data(Para1, Para2, seed=0):
    """Generate binary random data
    Para1, Para2: dict, {str:float} for each class, 
      keys are mx (center on x axis), my (center on y axis), 
               ux (sigma on x axis), ux (sigma on y axis), 
               y (label for this class)
    seed: int, seed for NUMPy's random number generator. Not Python's random.
    """
    numpy.random.seed(seed)
    X1 = numpy.vstack((numpy.random.normal(Para1['mx'], Para1['ux'], Para1['N']), 
                       numpy.random.normal(Para1['my'], Para1['uy'], Para1['N'])))
    X2 = numpy.vstack((numpy.random.normal(Para2['mx'], Para2['ux'], Para2['N']), 
                       numpy.random.normal(Para2['my'], Para2['uy'], Para2['N'])))
    Y = numpy.hstack(( Para1['y']*numpy.ones(Para1['N']), 
                       Para2['y']*numpy.ones(Para2['N'])))            
    X = numpy.hstack((X1, X2)) 
    X = numpy.transpose(X)
    return X, Y 

def plot_data_hyperplane(X, y, w, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array, the labels 
    w: 1-by-3 numpy array, the last element of which is the bias term
    Examples
    --------------
    >>> X = numpy.array([[1,2], \
                         [4,5], \
                         [7,8]]) 
    >>> y = numpy.array([1,-1,1])
    >>> w = [1, 2, -10]
    >>> filename = "test.png"
    >>> plot_data_hyperplane(X, y, w, filename)
    """
    
    plt.figure()
    plt.scatter(X[y==-1,0], X[y==-1,1], color='r', marker='X', label='Class 0')
    plt.scatter(X[y==1,0], X[y==1,1], color='b', marker='o', label='Class 1')
    
    # Plot the decision boundary
    x1 = numpy.linspace(numpy.min(X[:,0]), numpy.max(X[:,0]), 100)
    x2 = (-w[0]*x1 - w[2])/w[1]
    plt.plot(x1, x2, 'g-', label='decision boundary')
    matplotlib.pyplot.xlim(numpy.min(X[:,0]), numpy.max(X[:,0]))
    matplotlib.pyplot.ylim(numpy.min(X[:,1]), numpy.max(X[:,1]))
    plt.legend()
    plt.savefig(filename)
    plt.show()
    plt.close('all')

def Learning_and_Visual_mse(X,y,filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array
    Examples
    -----------------
    >>> X,y = generate_data(\
        {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=10)
    >>> learn_and_visual_mse(X, y, 'test1.png')
    array([-1.8650779 , -0.03934209,  2.91707992])
    >>> X,y = generate_data(\
    {'mx':1,'my':-2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
    {'mx':-1,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
    seed=10)
    >>> # print (X, y)
    >>> learn_and_visual_mse(X, y, 'test2.png')
    array([ 0.93061084, -0.01833983,  0.01127093])
    """
    w = np.array([0,0,0]) # just a placeholder

    X = np.hstack((X, np.ones((X.shape[0],1))))
    w = np.linalg.inv(X.T @ X)@ X.T @ y

    plot_data_hyperplane(X, y, w, filename)

    return w

def learn_and_visual_fisher(X,y,filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array
    Examples
    -----------------
    >>> X,y = generate_data(\
        {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=10)
    >>> learn_and_visual_fisher(X, y, 'test3.png')
    array([-1.61707972, -0.0341108 ,  2.54419773])
    >>> X,y = generate_data(\
        {'mx':-1.5,'my':2, 'ux':0.1, 'uy':2, 'y':1, 'N':200}, \
        {'mx':2,'my':-4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=1)
    >>> learn_and_visual_fisher(X, y, 'test4.png')
    array([-1.54593468,  0.00366625,  0.40890079])
    """

    w = np.array([0,0,0]) # just a placeholder

    X1 = X[y==1]
    X2 = X[y==-1]
    mu1 = np.mean(X1,axis=0)
    mu2 = np.mean(X2,axis=0)

    # your code below
    S1 = np.dot((X1-mu1).T,(X1-mu1))
    S2 = np.dot((X2-mu2).T,(X2-mu2))
    Sw = S1 + S2
    w1 = np.dot(np.linalg.inv(Sw),(mu1-mu2))
    b = -0.5*np.dot(np.dot(mu1.T,np.linalg.inv(Sw)),mu1) + 0.5*np.dot(np.dot(mu2.T,np.linalg.inv(Sw)),mu2)
    w = np.append(w1,b)
    # your code above


    # Plot after you have w.
    plot_data_hyperplane(X, y, w, filename)
    return w

def cal_fisher_score(X, y):
    X1 = X[y == 1]
    X2 = X[y == -1]
    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)

    # your code below
    mu = np.mean(X, axis=0)
    Sb = (mu1 - mu) ** 2 * X1.shape[0] / X.shape[0] + (mu2 - mu) ** 2 * X2.shape[0] / X.shape[0]
    Sw = (np.sum((X1 - mu1) ** 2, axis=0) + np.sum((X2 - mu2) ** 2, axis=0)) / X.shape[0]
    fisher_score = Sb / Sw
    return fisher_score

if __name__ == '__main__':
    data = pd.read_csv("breast-cancer-wisconsin.data", header=None)
    data.columns = [
        "ID", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
        "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses",
        "Class"
    ]
    
    data = data[data["BareNuclei"] != "?"]
    data["BareNuclei"] = data["BareNuclei"].astype(int)

    X, y = data.iloc[:, 1:-1], data.iloc[:, -1].map({2:-1, 4:1})
    fisher_score = cal_fisher_score(X, y)

    plt.figure(figsize=(6, 2.5), dpi=300)
    plt.barh(fisher_score.index, fisher_score.values)
    plt.title("Fisher Score for Each Feature", fontsize=9)
    plt.tick_params(labelsize=3)
    plt.show()
    
    select_features = fisher_score.sort_values(ascending=False).index.tolist()[:2]
    select_X = X[select_features]

    knn = KNeighborsClassifier()

    knn_params = {
        'n_neighbors': list(range(2, 21)),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    gs = GridSearchCV(knn, knn_params, cv=10, scoring="accuracy")
    gs.fit(select_X, y)
    print("best accuracy:", gs.best_score_)
    print("best params:", gs.best_params_)

    final_knn = KNeighborsClassifier(
        n_neighbors=gs.best_params_["n_neighbors"],
        weights=gs.best_params_["weights"],
        algorithm=gs.best_params_["algorithm"]
    )

    final_knn.fit(select_X, y)
    preds = final_knn.predict(select_X)
    conf_mat = confusion_matrix(y, preds)
    print(conf_mat)

    plt.figure(figsize=(4, 2.5), dpi=300)
    sns.heatmap(conf_mat, annot=True, xticklabels=['2', '4'], yticklabels=['2', '4'], fmt='.20g')
    plt.title('confusion matrix')
    plt.xlabel('prediction')
    plt.ylabel('truth')
    plt.show()
    
    learn_and_visual_fisher(select_X.values, y.values, 'test4.png')