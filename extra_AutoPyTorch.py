from autoPyTorch import AutoNetClassification

import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection


import json

X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=193)

autoPyTorch = AutoNetClassification(config_preset="tiny_cs", 
                                    budget_type='epochs', 
                                    min_budget=50, 
                                    max_budget=200, 
                                    num_iterations=5, 
                                    log_level='debug')
results_fit = autoPyTorch.fit(optimize_metric='accuracy',
                          cross_validator='k_fold',
                          early_stopping_patience=3,
                          loss_modules=['cross_entropy', 'cross_entropy_weighted'],
                          log_level="debug",
                          X_train=X_train,
                          Y_train=Y_train,
                          validation_split=0.3,
                          cuda=True)


y_pred = autoPyTorch.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))

with open("logs/results_fit.json", "w") as file:
    json.dump(results_fit, file)
