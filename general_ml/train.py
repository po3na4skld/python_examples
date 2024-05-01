import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

import constants

train_data = pd.read_csv(constants.DATA_PATH)

features = [c for c in train_data if c != constants.LABEL_NAME]
x_data, y_data = train_data[features], train_data[constants.LABEL_NAME]

train_x, val_x, train_y, val_y = train_test_split(
    x_data,
    y_data,
    test_size=constants.VAL_SIZE,
    random_state=constants.RANDOM_STATE
)

d_train = xgb.DMatrix(train_x, label=train_y)
d_val = xgb.DMatrix(val_x, label=val_y)

# Using general parameters as no model optimization was stated in the task
params = {
    "max_depth": 4,
    "min_child_weight": 1,
    "eta": 0.2,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "objective": "reg:squarederror",
}

# Specify validations set to watch performance
watchlist = [(d_train, "train"), (d_val, "eval")]

# Train the model
bst = xgb.train(params, d_train, constants.N_ROUNDS, watchlist, early_stopping_rounds=constants.N_EARLY_STOPPING)

y_pred_train = bst.predict(d_train)
y_pred_val = bst.predict(d_val)

train_rmse = root_mean_squared_error(train_y, y_pred_train)
val_rmse = root_mean_squared_error(val_y, y_pred_val)

print(f"Training finished.")
print(f"RMSE train: {train_rmse}")
print(f"RMSE val: {val_rmse}")

# Optionally, save the model
bst.save_model(constants.MODEL_PATH)
