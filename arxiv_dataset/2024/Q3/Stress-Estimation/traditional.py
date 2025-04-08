# Import necessary metrics and classifiers from scikit-learn
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

# Function to train and evaluate Support Vector Machine (SVM)
def SVM(X_train, y_train, X_test, y_test):
    """
    Train an SVM classifier with a linear kernel and evaluate its performance.
    """
    clf = svm.SVC(kernel='linear')  # Using linear kernel

    # Train the model on the training data
    clf.fit(X_train, y_train)

    # Predict the labels for both training and test sets
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    # Return evaluation metrics: F1-score, test accuracy, and training accuracy
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), accuracy_score(y_train, y_pred_train)

# Function to train and evaluate Random Forest Classifier
def RandomForest(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest classifier and evaluate its performance.
    """
    clf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=12, n_jobs=-1, random_state=42)

    # Train the model on the training data
    clf.fit(X_train, y_train)

    # Predict the labels for both training and test sets
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    # Return evaluation metrics: F1-score, test accuracy, and training accuracy
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), accuracy_score(y_train, y_pred_train)

# Function to train and evaluate AdaBoost with Decision Tree as the base classifier
def AdaBoost_DecisionTree(X_train, y_train, X_test, y_test):
    """
    Train an AdaBoost classifier with Decision Tree as the base estimator.
    """
    abc = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)  # Configure AdaBoost parameters

    # Train the AdaBoost classifier
    model = abc.fit(X_train, y_train)

    # Predict the labels for both training and test sets
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Return evaluation metrics: F1-score, test accuracy, and training accuracy
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), accuracy_score(y_train, y_pred_train)

# Function to train and evaluate AdaBoost with SVM as the base classifier
def Adaboost_SVC(X_train, y_train, X_test, y_test):
    """
    Train an AdaBoost classifier using SVM as the base estimator.
    """
    svc = svm.SVC(probability=True, kernel='linear')  # SVM with probability estimates
    abc = AdaBoostClassifier(n_estimators=100, base_estimator=svc, learning_rate=0.1)

    # Train the AdaBoost classifier
    model = abc.fit(X_train, y_train)

    # Predict the labels for both training and test sets
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Return evaluation metrics: F1-score, test accuracy, and training accuracy
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), accuracy_score(y_train, y_pred_train)

# Function to train and evaluate Neural Network (Multilayer Perceptron)
def NeuralNetwork(X_train, y_train, X_test, y_test):
    """
    Train a Multilayer Perceptron (MLP) neural network and evaluate its performance.
    """
    # Create a neural network with 3 hidden layers of 50 neurons each
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 50), activation='relu', solver='adam', max_iter=1000)

    # Train the neural network
    # Use to_numpy() instead of ravel() to avoid the warning
    model = mlp.fit(X_train, y_train.to_numpy())

    # Predict the labels for both training and test sets
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Return evaluation metrics: F1-score, test accuracy, and training accuracy
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), accuracy_score(y_train, y_pred_train)


# Function to train and evaluate Linear Discriminant Analysis (LDA)
def LinearDiscriminant(X_train, y_train, X_test, y_test):
    """
    Train a Linear Discriminant Analysis classifier and evaluate its performance.
    """
    clf = LinearDiscriminantAnalysis()

    # Train the model on the training data
    clf.fit(X_train, y_train)

    # Predict the labels for both training and test sets
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    # Return evaluation metrics: F1-score, test accuracy, and training accuracy
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), accuracy_score(y_train, y_pred_train)

# Function to train and evaluate Stochastic Gradient Descent (SGD) classifier
def SGD(X_train, y_train, X_test, y_test):
    """
    Train a Stochastic Gradient Descent (SGD) classifier and evaluate its performance.
    """
    clf = SGDClassifier(loss="log_loss", penalty="l2")  # Using log loss and l2 penalty

    # Train the model on the training data
    clf.fit(X_train, y_train)

    # Predict the labels for both training and test sets
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    # Return evaluation metrics: F1-score, test accuracy, and training accuracy
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), accuracy_score(y_train, y_pred_train)

# Function to train and evaluate K-Nearest Neighbors (KNN) classifier
def KNN(X_train, y_train, X_test, y_test):
    """
    Train a K-Nearest Neighbors (KNN) classifier and evaluate its performance.
    """
    knn = KNeighborsClassifier(n_neighbors=24)  # Setting number of neighbors to 24

    # Train the model on the training data
    knn.fit(X_train, y_train)

    # Predict the labels for both training and test sets
    y_pred = knn.predict(X_test)
    y_pred_train = knn.predict(X_train)

    # Return evaluation metrics: F1-score, test accuracy, and training accuracy
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred), accuracy_score(y_train, y_pred_train)