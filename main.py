import numpy as np
import tensorflow as tf
from LSTM_5_folds import LSTM_Model

<<<<<<< HEAD

=======
>>>>>>> 7c4ecc6e3876183b7c4c1e9ec50f96d119e64e15
lstm_model_1 = LSTM_Model(n_steps = 4, n_features = 1, n_folds = 5)
lstm_model_1.set_data("databases/6.csv", "new_case_count")
lstm_model_1.make_data(6)
lstm_model_1.n_fold_train()
output = lstm_model_1.predict()
<<<<<<< HEAD
print(output)
=======
print(output)
>>>>>>> 7c4ecc6e3876183b7c4c1e9ec50f96d119e64e15
