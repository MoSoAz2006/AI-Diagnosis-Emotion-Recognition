import sys, os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score

def train_pipeline(csv_path, max_classes=100, min_count=5):
    print("📂 در حال بارگذاری داده‌ها...")
    df = pd.read_csv(csv_path, low_memory=False)
    print("✅ شکل اولیه داده:", df.shape)

    # پیدا کردن ستون بیماری
    disease_col = 'diseases' if 'diseases' in df.columns else df.columns[0]

    # بقیه‌ی ستون‌ها علائم هستند
    symptom_cols = [c for c in df.columns if c != disease_col]
    print(f"🧩 تعداد علائم شناسایی‌شده: {len(symptom_cols)}")

    # حذف ردیف‌های بدون برچسب بیماری
    df = df[df[disease_col].notna()]
    df = df[df[disease_col].astype(str).str.strip() != '']
    print("✅ بعد از حذف موارد خالی:", df.shape)

    # فیلتر کلاس‌هایی که نمونه کافی دارند
    vc = df[disease_col].value_counts()
    valid_diseases = vc[vc >= min_count].nlargest(max_classes).index.tolist()
    df = df[df[disease_col].isin(valid_diseases)]
    print(f"🎯 کلاس‌های نهایی ({len(valid_diseases)}):", valid_diseases[:10], "...")

    # جدا کردن X و y
    X = df[symptom_cols].fillna(0).astype(float).values
    y = df[disease_col].astype(str).values

    # برچسب‌گذاری بیماری‌ها
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # تقسیم داده‌ها
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # مقیاس‌بندی (اختیاری اما مفید)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # مدل اصلی
    print("🚀 آموزش مدل RandomForestClassifier ...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    # ارزیابی
    print("📊 در حال ارزیابی مدل ...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    probs = clf.predict_proba(X_test)
    top3 = top_k_accuracy_score(y_test, probs, k=3)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"✅ دقت (Accuracy): {acc*100:.2f}%")
    print(f"✅ Top-3 Accuracy: {top3*100:.2f}%")
    print("📋 گزارش مدل:\n", report[:1000])

    # ذخیره مدل و برچسب‌ها
    
    save_dir = r"models"

    os.makedirs(save_dir, exist_ok=True)
    
    base = os.path.splitext(os.path.basename(csv_path))[0]

    model_path = os.path.join(save_dir, f"{base}_binary_model.joblib")
    le_path = os.path.join(save_dir, f"{base}_binary_labels.joblib")
    scaler_path = os.path.join(save_dir, f"{base}_binary_scaler.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(le, le_path)
    joblib.dump(scaler, scaler_path)

    print("\n💾 فایل‌ها ذخیره شدند:")
    print(" -", model_path)
    print(" -", le_path)
    print(" -", scaler_path)

    return {
        "accuracy": acc,
        "top3": top3,
        "num_classes": len(valid_diseases),
        "model_path": model_path,
        "label_encoder_path": le_path,
        "scaler_path": scaler_path
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_disease_model_binary.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    result = train_pipeline(csv_path)
    print("\n✅ نتیجه نهایی:", result)
