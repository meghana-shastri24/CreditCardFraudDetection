from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def build_model(self):
        """
        Initialize a Random Forest model with the specified parameters.
        """
        self.model = RandomForestClassifier(**self.params)
