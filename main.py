import numpy as np
import tensorflow as tf
from LSTM_TimeSeries import LSTM_Model


lstm_model_1 = LSTM_Model(n_steps = 7, n_features = 1, n_folds = 5)
lstm_model_1.set_data("databases/39.csv", "case_count")
lstm_model_1.make_data(stateID=39)
# lstm_model_1.n_fold_train()
# lstm_model_1.train_model()
output = lstm_model_1.predict_3_days()
# 
# lstm_model_1.update_csv()

print(output)

