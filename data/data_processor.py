import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


class DataProcessor:
    def __init__(self, file_path):
        """
        Initialize DataProcessor with the file path.
        """
        self.data = pd.read_csv(file_path)

    def preprocess_data(self):
        """
        Preprocess data by standardizing `Time` and `Amount` columns.
        """
        print("Standardizing `Time` and `Amount`...")
        scaler = StandardScaler()
        self.data["Time"] = scaler.fit_transform(self.data["Time"].values.reshape(-1, 1))
        self.data["Amount"] = scaler.fit_transform(self.data["Amount"].values.reshape(-1, 1))

        X = self.data.drop("Class", axis=1)
        y = self.data["Class"]
        return X, y

    def get_train_test_split(self, X, y, test_size=0.3, random_state=42):
        """
        Split data into train and test sets.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def apply_smote(self, X_train, y_train):
        """
        Apply SMOTE to handle class imbalance.
        """
        print("Applying SMOTE to oversample minority class...")
        smote = SMOTE(sampling_strategy=0.2, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
