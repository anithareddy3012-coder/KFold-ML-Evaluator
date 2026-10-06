import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================
# DATASET INFORMATION
# =========================================

def get_dataset_info():

    iris = load_iris()

    return {
        "name": "Iris Dataset",
        "samples": iris.data.shape[0],
        "features": iris.data.shape[1],
        "classes": len(np.unique(iris.target))
    }


# =========================================
# AVAILABLE MODELS
# =========================================

def get_models():

    return {

        "Logistic Regression":
            LogisticRegression(max_iter=200),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "K-Nearest Neighbors":
            KNeighborsClassifier(),

        "Random Forest":
            RandomForestClassifier(
                random_state=42
            ),

        "SVM":
            SVC()
    }


# =========================================
# RUN K-FOLD EVALUATION
# =========================================

def run_kfold(
    k=5,
    selected_models=None,
    X=None,
    y=None
):

    # Use Iris dataset if
    # no custom dataset is supplied

    if X is None or y is None:

        iris = load_iris()

        X = iris.data
        y = iris.target


    # Convert to numpy arrays

    X = np.asarray(X)
    y = np.asarray(y)


    # K-Fold configuration

    kfold = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=42
    )


    all_models = get_models()


    if not selected_models:

        selected_models = list(
            all_models.keys()
        )


    results = {}


    # =====================================
    # MODEL EVALUATION
    # =====================================

    for model_name in selected_models:

        if model_name not in all_models:

            continue


        model = all_models[model_name]


        # Accuracy

        scores = cross_val_score(
            model,
            X,
            y,
            cv=kfold,
            scoring="accuracy"
        )


        average_accuracy = np.mean(
            scores
        )


        standard_deviation = np.std(
            scores
        )


        # Predictions

        predictions = cross_val_predict(
            model,
            X,
            y,
            cv=kfold
        )


        # Precision

        precision = precision_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )


        # Recall

        recall = recall_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )


        # F1 Score

        f1 = f1_score(
            y,
            predictions,
            average="weighted",
            zero_division=0
        )


        # Confusion Matrix

        matrix = confusion_matrix(
            y,
            predictions
        )


        # Store results

        results[model_name] = {

            "fold_scores":
                scores,

            "average_accuracy":
                float(average_accuracy),

            "standard_deviation":
                float(standard_deviation),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1_score":
                float(f1),

            "confusion_matrix":
                matrix.tolist()
        }


    return results


# =========================================
# FIND BEST MODEL
# =========================================

def find_best_model(results):

    if not results:

        return None


    return max(
        results,
        key=lambda model:
            results[model][
                "average_accuracy"
            ]
    )


# =========================================
# LOAD CSV DATASET
# =========================================

def load_csv_dataset(
    file_path,
    target_column
):

    # Read CSV

    df = pd.read_csv(
        file_path
    )


    # Check target column

    if target_column not in df.columns:

        raise ValueError(
            "Target column not found in dataset."
        )


    # Remove rows with missing values

    df = df.dropna()


    # Separate features and target

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]


    # Convert categorical
    # feature columns to numbers

    X = pd.get_dummies(
        X,
        drop_first=True
    )


    # Convert target column
    # into numerical labels

    if not pd.api.types.is_numeric_dtype(
        y
    ):

        y = pd.factorize(
            y
        )[0]


    # Convert to numpy arrays

    X = X.to_numpy()

    y = np.asarray(y)


    return X, y, df


# =========================================
# CSV INFORMATION
# =========================================

def get_csv_info(
    file_path,
    target_column
):

    df = pd.read_csv(
        file_path
    )


    if target_column not in df.columns:

        raise ValueError(
            "Target column not found."
        )


    return {

        "name":
            file_path.split("\\")[-1],

        "samples":
            int(df.shape[0]),

        "features":
            int(
                df.shape[1] - 1
            ),

        "classes":
            int(
                df[target_column].nunique()
            ),

        "columns":
            list(df.columns)
    }


# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    results = run_kfold(5)

    best_model = find_best_model(
        results
    )


    print(
        "\nK-FOLD RESULTS\n"
    )


    for model_name, result in results.items():

        print(
            "\nModel:",
            model_name
        )

        print(
            "Average Accuracy:",
            round(
                result[
                    "average_accuracy"
                ] * 100,
                2
            ),
            "%"
        )

        print(
            "Standard Deviation:",
            round(
                result[
                    "standard_deviation"
                ] * 100,
                2
            ),
            "%"
        )

        print(
            "Precision:",
            round(
                result[
                    "precision"
                ] * 100,
                2
            ),
            "%"
        )

        print(
            "Recall:",
            round(
                result[
                    "recall"
                ] * 100,
                2
            ),
            "%"
        )

        print(
            "F1 Score:",
            round(
                result[
                    "f1_score"
                ] * 100,
                2
            ),
            "%"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            np.array(
                result[
                    "confusion_matrix"
                ]
            )
        )


    print(
        "\nBest Model:",
        best_model
    )