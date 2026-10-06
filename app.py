from flask import (
    Flask,
    render_template,
    jsonify,
    request
)

import os
import numpy as np
import pandas as pd

from ml_engine import (
    run_kfold,
    find_best_model,
    get_dataset_info,
    get_models,
    load_csv_dataset
)

app = Flask(__name__)

UPLOAD_FOLDER = "data"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================
# HOME PAGE
# =========================================

@app.route("/")
def home():

    dataset_info = get_dataset_info()
    models = list(get_models().keys())

    return render_template(
        "index.html",
        dataset=dataset_info,
        models=models
    )


# =========================================
# RUN K-FOLD ON IRIS DATASET
# =========================================

@app.route("/run", methods=["POST"])
def run_model():

    data = request.get_json()

    k = int(data.get("k", 5))
    selected_models = data.get("models", [])

    if not selected_models:
        selected_models = list(get_models().keys())

    results = run_kfold(
        k,
        selected_models
    )

    best_model = find_best_model(results)

    response = {
        "k": k,
        "best_model": best_model,
        "best_accuracy":
            results[best_model]["average_accuracy"]
            if best_model else 0,
        "results": {}
    }

    for model_name, result in results.items():

        response["results"][model_name] = {

            "fold_scores":
                result["fold_scores"].tolist(),

            "average_accuracy":
                result["average_accuracy"],

            "standard_deviation":
                result["standard_deviation"],

            "precision":
                result["precision"],

            "recall":
                result["recall"],

            "f1_score":
                result["f1_score"],

            "confusion_matrix":
                result["confusion_matrix"]
        }

    return jsonify(response)


# =========================================
# UPLOAD CSV DATASET
# =========================================

@app.route("/upload", methods=["POST"])
def upload_dataset():

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No file was uploaded."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "Please select a CSV file."
            }), 400

        if not file.filename.lower().endswith(".csv"):

            return jsonify({
                "success": False,
                "error": "Only CSV files are supported."
            }), 400

        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(file_path)

        # Read CSV
        df = pd.read_csv(file_path)

        if df.empty:

            return jsonify({
                "success": False,
                "error": "The uploaded CSV file is empty."
            }), 400

        # =========================================
        # FIRST 5 ROWS FOR DATASET PREVIEW
        # =========================================

        preview = (
            df.head(5)
            .fillna("")
            .astype(str)
            .to_dict(orient="records")
        )

        dataset_info = {

            "name":
                file.filename,

            "samples":
                int(df.shape[0]),

            "features":
                int(df.shape[1]),

            "columns":
                list(df.columns),

            "preview":
                preview
        }

        return jsonify({

            "success": True,

            "dataset":
                dataset_info
        })

    except Exception as e:

        print("Upload error:", e)

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# =========================================
# RUN K-FOLD ON CSV DATASET
# =========================================

@app.route("/run_csv", methods=["POST"])
def run_csv_model():

    try:

        data = request.get_json()

        file_name = data.get("file")
        target_column = data.get("target")

        k = int(
            data.get(
                "k",
                5
            )
        )

        selected_models = data.get(
            "models",
            []
        )

        if not file_name:

            return jsonify({

                "success": False,

                "error":
                    "Dataset file is missing."

            }), 400

        if not target_column:

            return jsonify({

                "success": False,

                "error":
                    "Target column is required."

            }), 400

        file_path = os.path.join(

            app.config["UPLOAD_FOLDER"],

            file_name
        )

        if not os.path.exists(file_path):

            return jsonify({

                "success": False,

                "error":
                    "Dataset file not found."

            }), 404

        # Load dataset
        X, y, df = load_csv_dataset(

            file_path,

            target_column
        )

        # Run K-Fold
        results = run_kfold(

            k,

            selected_models,

            X,

            y
        )

        best_model = find_best_model(
            results
        )

        response = {

            "success":
                True,

            "k":
                k,

            "dataset": {

                "name":
                    file_name,

                "samples":
                    len(df),

                "features":
                    X.shape[1],

                "classes":
                    len(np.unique(y))
            },

            "best_model":
                best_model,

            "best_accuracy":
                results[best_model]["average_accuracy"]
                if best_model else 0,

            "results":
                {}
        }

        for model_name, result in results.items():

            response["results"][model_name] = {

                "fold_scores":
                    result["fold_scores"].tolist(),

                "average_accuracy":
                    result["average_accuracy"],

                "standard_deviation":
                    result["standard_deviation"],

                "precision":
                    result["precision"],

                "recall":
                    result["recall"],

                "f1_score":
                    result["f1_score"],

                "confusion_matrix":
                    result["confusion_matrix"]
            }

        return jsonify(response)

    except Exception as e:

        print(
            "CSV evaluation error:",
            e
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# =========================================
# START FLASK
# =========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )