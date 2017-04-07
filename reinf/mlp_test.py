'''
Created on 7 Mar 2017

@author: Russell
'''
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

def main():
    X = [[0.,1.,0.,1.,0.,0.], [1.,0.,0., 1., 1., 0.]]
    y = [0,15]
    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.partial_fit(X, y)
    res = clf.predict([[1., 1., 0., 1., 1., 1.], [0, 0, 1 ,1, 1.,0]])
    print("res",res)

if __name__=="__main__":
    main()