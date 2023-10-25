import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, hamming_loss


class XGBoostClassifierEvaluator:
    """
    A class for training an XGBoost classifier and evaluating it on multi-label classification tasks.

    Args:
        features_file (str): File path to the CSV containing feature data (lab results).
        labels_file (str): File path to the CSV containing target data (diagnosis labels).
        random_state (int, optional): Random seed for reproducibility. Default is 46.

    Attributes:
        features_file (str): File path to the CSV containing feature data (lab results).
        labels_file (str): File path to the CSV containing target data (diagnosis labels).
        random_state (int): Random seed for reproducibility.
        model (xgb.XGBClassifier): The trained XGBoost classifier.
    """

    def __init__(self, features_file, labels_file, random_state=46): 
        self.features_file = features_file # features_file is a string
        self.labels_file = labels_file # labels_file is a string
        self.random_state = random_state # random_state is an integer
        self.model = None # model is None
        self.X_train = None # X_train is None
        self.X_test = None # X_test is None
        self.y_train = None # y_train is None
        self.y_test = None # y_test is None
        self.load_data()
        self.train_xgboost_model()
        self.evaluate_model()

    def load_data(self):
        X = pd.read_csv(self.features_file) # X is a dataframe, loading features_file
        y_binary = pd.read_csv(self.labels_file) # y_binary is a dataframe, loading labels_file

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_binary, 
            test_size=0.1, 
            random_state=self.random_state
        ) # X_train, X_test, y_train, y_test are dataframes

    def train_xgboost_model(self, 
                            objective= "binary:logistic",
                            max_depth = 4,
                            learning_rate = 0.31, 
                            n_estimators = 325, 
                            subsample = 0.89, 
                            colsample_bytree = 0.9, 
                            seed = 46) -> None:
                            
        model_params = {} # model_params is a dictionary
        model_params["objective"] = objective
        model_params["max_depth"] = max_depth
        model_params["learning_rate"] = learning_rate
        model_params["n_estimators"] = n_estimators
        model_params["subsample"] = subsample
        model_params["colsample_bytree"] = colsample_bytree
        model_params["seed"] = seed

        # Create and train the XGBoost model
        self.model = xgb.XGBClassifier(**model_params) # model is an XGBClassifier object
        self.model.fit(self.X_train, self.y_train) # fitting the model

    def evaluate_model(self):
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test) 

        y_pred_training = self.model.predict(self.X_train) 

        # Evaluate the model using appropriate multi-label classification metrics
        precision = precision_score(self.y_test, y_pred, average="micro")
        f1 = f1_score(self.y_test, y_pred, average="micro")
        f1_training = f1_score(self.y_train, y_pred_training, average="micro")
        hamming = hamming_loss(self.y_test, y_pred)

        print(f"Precision: {precision}")
        print(f"F1 Score: {f1}")
        print(f"Hamming Loss: {hamming}")
        print(f"F1 Shap: {f1_training}")





boost = XGBoostClassifierEvaluator("features.csv", "labels.csv")
