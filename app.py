from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# 1) Load model & chuẩn bị dữ liệu
# -----------------------------
MODEL_PATH = "xgb_final_model.joblib"

# Load model đã train
model = joblib.load(MODEL_PATH)

# Danh sách tất cả các feature trong model
all_features = model.get_booster().feature_names

# Chọn top feature quan trọng nhất (theo Gain)
TOP_FEATURES = [
    "NumberOfTimes90DaysLate",
    "NumberOfTimes60-89DaysPastDueNotWorse",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberRealEstateLoansOrLines"
]

# -----------------------------
# 2) Flask App
# -----------------------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None

    if request.method == "POST":
        # Tạo dict chứa tất cả feature
        row = {}
        for f in all_features:
            if f in TOP_FEATURES:
                v = request.form.get(f, "").strip()
                if v == "":
                    row[f] = np.nan
                else:
                    try:
                        row[f] = float(v.replace(",", "."))
                    except:
                        row[f] = np.nan
            else:
                # Feature không nhập thì gán NaN (model sẽ xử lý được)
                row[f] = np.nan

        # Chuyển thành DataFrame
        df_in = pd.DataFrame([row], columns=all_features)

        # Dự đoán
        proba = model.predict_proba(df_in)[0, 1]
        prediction = int(proba >= 0.5)

    return render_template("index.html",
                           features=TOP_FEATURES,
                           prediction=prediction,
                           proba=proba)

# -----------------------------
# 3) Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
