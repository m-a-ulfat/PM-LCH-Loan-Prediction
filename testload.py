from catboost import CatBoostClassifier

try:
    model = CatBoostClassifier()
    model.load_model("PMLP_catboost_model.cbm")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)