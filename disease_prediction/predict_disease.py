# ================================================================
# 🧠 Disease Predictor - Binary Model
# ================================================================

import sys
import joblib
import pandas as pd
import numpy as np

# ================================================================
# 📘 تابع پیش‌بینی بیماری
# ================================================================
def predict_disease(symptom_list, model_path, label_path, scaler_path, all_symptoms):
    # بارگذاری اجزای مدل
    clf = joblib.load(model_path)
    le = joblib.load(label_path)
    scaler = joblib.load(scaler_path)

    # آماده‌سازی ورودی
    symptom_vector = np.zeros((1, len(all_symptoms)))
    for s in symptom_list:
        s = s.strip().lower()
        if s in all_symptoms:
            idx = all_symptoms.index(s)
            symptom_vector[0, idx] = 1

    # نرمال‌سازی
    symptom_vector = scaler.transform(symptom_vector)

    # پیش‌بینی
    probs = clf.predict_proba(symptom_vector)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    top_diseases = [(le.inverse_transform([i])[0], probs[i]) for i in top_indices]

    return top_diseases


if __name__ == "__main__":


    # بارگذاری لیست علائم از دیتاست اصلی
    data_path = "data\Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    df = pd.read_csv(data_path, nrows=1)
    all_symptoms = [c for c in df.columns if c != "diseases"]

    # مسیر فایل‌های مدل
    model_path = "models\Final_Augmented_dataset_Diseases_and_Symptoms_binary_model.joblib"
    label_path = "models\Final_Augmented_dataset_Diseases_and_Symptoms_binary_labels.joblib"
    scaler_path = "models\Final_Augmented_dataset_Diseases_and_Symptoms_binary_scaler.joblib"

    # گرفتن ورودی علائم از کاربر
    user_input = sys.argv[1]
    symptom_list = [s.strip().lower() for s in user_input.split(",")]

    # پیش‌بینی
    result = predict_disease(symptom_list, model_path, label_path, scaler_path, all_symptoms)

    print("\n🔍 علائم وارد شده:", ", ".join(symptom_list))
    print("🩺 بیماری‌های پیشنهادی:")
    for d, p in result:
        print(f"  • {d} — احتمال: {p*100:.2f}%")
