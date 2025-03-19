# CreditCardFraudDetection
Fraud detection using Kaggle dataset
## About the Data
The dataset used for this project is sourced from Kaggle and contains anonymized credit card transactions labeled as fraudulent or legitimate. It includes features derived from PCA for privacy.

## Models Used
The project employs machine learning models such as Random Forest, and XGBoosting to detect fraudulent transactions. Model performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

## How to Use the API
1. Clone the repository:
    ```bash
    https://github.com/meghana-shastri24/CreditCardFraudDetection.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the API server: localhost
    ```bash
    uvicorn app:app --reload
    ```
4. Use a tool like Postman or `curl` or use http://127.0.0.1:8000/docs with SwaggerUI to send a POST request to the API endpoint `/predict` with transaction data in JSON format:
    ```json
    {
    "features": [
        0.0, -1.359807, -0.072781, 2.536346, 1.378155, -0.338321, 0.462388,
        0.239599, 0.098698, 0.363787, 0.090794, -0.551600, -0.617801, -0.991390,
        -0.311169, 1.468177, -0.470401, 0.207971, 0.025790, 0.403993, 0.251412,
        -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558,
        -0.021053, 149.62
    ]
    }

    ```
    The API will return a prediction indicating whether the transaction is fraudulent or legitimate with confidence.
   Swagger example pdf: [FastAPI - Swagger UI.pdf](https://github.com/user-attachments/files/19333504/FastAPI.-.Swagger.UI.pdf)
