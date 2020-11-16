import numpy as np
import tensorflow as tf
from LSTM_5_folds import LSTM_Model


lstm_model_1 = LSTM_Model(n_steps = 4, n_features = 1, n_folds = 5)
lstm_model_1.set_data("databases/6.csv", "new_case_count")
lstm_model_1.make_data(6)
# lstm_model_1.n_fold_train()
output = lstm_model_1.predict()

print(output)

