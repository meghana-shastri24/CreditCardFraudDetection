class BaseModel:
    def __init__(self, **kwargs):
        """
        BaseModel initializer to accept model-specific parameters.
        """
        self.params = kwargs
        self.model = None

    def build_model(self):
        """
        Abstract method to build a model.
        Must be implemented by child classes.
        """
        raise NotImplementedError("Child classes must implement `build_model`.")

    def train(self, X_train, y_train):
        """
        Train the model using provided data.
        """
        if self.model is None:
            raise ValueError("Model not instantiated. Call `build_model` first.")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict using the trained model.
        """
        if self.model is None:
            raise ValueError("Model not instantiated. Call `build_model` first.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities using the trained model (if supported).
        """
        if self.model is None:
            raise ValueError("Model not instantiated. Call `build_model` first.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("The model does not support predict_proba.")
