from autoPyTorch import AutoNetClassification

import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import json



# Substitute this with your actual path. This is the root folder of ESC-50, where
# you can find the subfolders 'audio' and 'meta'.
datapath = Path('data/ESC-50')

# Using Path is fundamental to have reproducible code across different operating systems.
csv = pd.read_csv(datapath / Path('meta/esc50.csv'))


# We need only filename, fold, and target
csv.head()





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
                          Y_train=y_train,
                          validation_split=0.3,
                          cuda=True)


y_pred = autoPyTorch.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))

with open("logs/results_fit.json", "w") as file:
    json.dump(results_fit, file)
