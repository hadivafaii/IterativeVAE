from sklearnex import patch_sklearn

patch = [
    'SVC',
    'LogisticRegression',
    'KNeighborsClassifier',
]
patch_sklearn(patch, verbose=False)
