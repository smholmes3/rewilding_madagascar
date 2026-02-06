import pandas as pd

import os
print("CWD:", os.getcwd())

train = pd.read_csv("/home/sholmes3/Python_downloads_linux/rewilding_madagascar/data/train_labels.csv")
val   = pd.read_csv("/home/sholmes3/Python_downloads_linux/rewilding_madagascar/data/val_labels.csv")
test  = pd.read_csv("/home/sholmes3/Python_downloads_linux/rewilding_madagascar/data/test_labels.csv")

print("Train clips:", len(train))
print("Val clips:", len(val))
print("Test clips:", len(test))