import pandas as pd
import xgboost as xgb

import constants

inference_data = pd.read_csv(constants.INFERENCE_DATA_PATH)

model = xgb.XGBRegressor()
model.load_model(constants.MODEL_PATH)

predictions = model.predict(inference_data)

pd.Series(predictions, name='inference_results').to_csv(constants.INFERENCE_RESULTS)
