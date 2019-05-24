# forgeNet

forgeNet: **for**est **g**raph-**e**mbedded deep feedforward **net**works - Tensorflow implementation

The method is introduced in .

## Prerequisites

The following packages are required for executing the main code file:

* NumPy http://www.numpy.org/
* Scikit-learn http://scikit-learn.org/stable/install.html
* XGBoost https://xgboost.readthedocs.io/en/latest/index.html
* Tensorflow (1.x) https://www.tensorflow.org/install/

## Usage

### Data formats

* data matrix (example_expression.csv): a csv file with n rows and p+1 columns. n is the number of samples and p is the number of features. The additional column at last is the 0/1 binary outcome variable vector. n=100 and p=500 for this example dataset.

NOTE: no headers are allowed in both files.

### Run forgeNet

In the terminal, change the directory to the folder under which forgeNet.py is located, then type the command

```
 python forgeNet.py "example_expression.csv" "RF"
```

where "RF" is the choice of the forest type for forgeNet. Currently, only the random forest classifier ("RF") from sklearn and the gradient boosting classifier ("XGB") from xgboost are available. An output file for variable importance will be created by the program automatically and saved in the current working directory. 

The program will run while printing logs

```
Case proportion in training data: 0.538
Case proportion in testing data: 0.35
Epoch: 0 cost = 0.633589327 Training accuracy: 0.462  Training auc: 0.444
Epoch: 5 cost = 0.582940525 Training accuracy: 0.738  Training auc: 0.926
Epoch: 10 cost = 0.552909470 Training accuracy: 0.9  Training auc: 0.964
Epoch: 15 cost = 0.490178621 Training accuracy: 0.9  Training auc: 0.973
Epoch: 20 cost = 0.407882863 Training accuracy: 0.912  Training auc: 0.979
Epoch: 25 cost = 0.315436481 Training accuracy: 0.912  Training auc: 0.983
Epoch: 30 cost = 0.262423600 Training accuracy: 0.938  Training auc: 0.988
Epoch: 35 cost = 0.223586632 Training accuracy: 0.938  Training auc: 0.993
Epoch: 40 cost = 0.205377628 Training accuracy: 0.975  Training auc: 0.996
Epoch: 45 cost = 0.157847774 Training accuracy: 0.988  Training auc: 0.998
*****===== Testing accuracy:  0.7  Testing auc:  0.879 =====*****
```

and the variable importance file can be found in this repo.

### Hyperparameters and training options

Modified directly in forgeNet.py.
