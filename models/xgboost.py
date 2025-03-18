from xgboost import XGBClassifier
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def build_model(self):
        """
        Initialize an XGBoost model with the specified parameters.
        """
        self.model = XGBClassifier(**self.params)
