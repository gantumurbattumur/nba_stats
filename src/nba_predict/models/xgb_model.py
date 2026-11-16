"""XGBoost win and total forecasts, with explicit history variants."""
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from ._nonlinear import NonlinearModel


class XGBModel(NonlinearModel):
    model_name = "xgb"

    def estimator(self):
        return XGBClassifier(n_estimators=120, max_depth=2, learning_rate=.04,
                             subsample=.8, colsample_bytree=.8, reg_lambda=5,
                             eval_metric="logloss", n_jobs=1, random_state=0)
