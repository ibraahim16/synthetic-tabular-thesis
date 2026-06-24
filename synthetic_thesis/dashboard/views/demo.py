import streamlit as st
import pandas as pd
import joblib
import random
from pathlib import Path
from collections import Counter

def show():

    ROOT = Path(__file__).resolve().parents[2]

    st.title("Gerçek / Sentetik Veri Tahmini")

    st.markdown("""
    Bu bölümde kullanıcı tarafından girilen veri kaydı, her sentetik veri üretim yöntemi için eğitilmiş dedektör modelleri tarafından analiz edilmektedir.

    Her model, girilen kaydın **gerçek** veya **sentetik** veri olma durumunu tahmin etmektedir.

    Nihai karar ise tüm dedektör modellerinin sonuçları dikkate alınarak **çoğunluk oylaması** yöntemi ile belirlenmektedir.
    """)

    # =====================================================
    # VERİ SETİ SEÇİMİ
    # =====================================================

    dataset = st.selectbox(
        "Veri Seti Seçiniz",
        options=["adult", "credit"],
        format_func=lambda x: {
            "adult": "Adult Income Veri Seti",
            "credit": "Credit Fraud Veri Seti"
        }[x]
    )

    # =====================================================
    # FORM
    # =====================================================

    st.header("Özellikleri Giriniz")
    user_input = {}

    if dataset == "adult":

        sample_df = pd.read_csv(
            ROOT / "data/split/adult_train.csv"
        )

        # income kullanılmayacak
        feature_cols = [
            c for c in sample_df.columns
            if c != "income"
        ]

        col1, col2 = st.columns([3, 1])

        with col2:

            if st.button("Örnek Veri Oluştur", key="adult_random"):
                random_row = sample_df.sample(
                    n=1,
                    random_state=random.randint(0, 1000000)
                ).iloc[0]

                # Değerleri doğrudan bileşenlerin kendi session_state key'lerine yazıyoruz
                for c in feature_cols:
                    if sample_df[c].dtype == "object":
                        st.session_state[f"adult_{c}"] = str(random_row[c])
                    else:
                        st.session_state[f"adult_{c}"] = float(random_row[c])

        for col in feature_cols:
            widget_key = f"adult_{col}"

            # Kategorik sütun
            if sample_df[col].dtype == "object":
                options = sorted(
                    sample_df[col]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                # İlk açılışta varsayılan değeri atama
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = options[0]

                user_input[col] = st.selectbox(
                    label=col,
                    options=options,
                    key=widget_key
                )

            # Sayısal sütun
            else:
                # İlk açılışta varsayılan değeri atama
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = float(sample_df[col].median())

                user_input[col] = st.number_input(
                    label=col,
                    key=widget_key
                )

    # =====================================================
    # CREDIT
    # =====================================================

    else:

        sample_df = pd.read_csv(
            ROOT / "data/split/credit_train.csv"
        )

        feature_cols = [
            c for c in sample_df.columns
            if c != "Class"
        ]

        col1, col2 = st.columns([3, 1])

        with col2:

            if st.button("Örnek Veri Oluştur", key="credit_random"):
                random_row = sample_df.sample(
                    n=1,
                    random_state=random.randint(0, 1000000)
                ).iloc[0]

                # Değerleri doğrudan bileşenlerin kendi session_state key'lerine yazıyoruz
                for c in feature_cols:
                    st.session_state[f"credit_{c}"] = float(random_row[c])

        for col in feature_cols:
            widget_key = f"credit_{col}"

            # İlk açılışta varsayılan değeri atama
            if widget_key not in st.session_state:
                st.session_state[widget_key] = float(sample_df[col].median())

            user_input[col] = st.number_input(
                label=col,
                key=widget_key
            )

    # =====================================================
    # TAHMİN
    # =====================================================

    if st.button("Tahmin Yap"):

        user_df = pd.DataFrame([user_input])

        # kategorikleri dummy değişkenlere dönüştür
        user_df = pd.get_dummies(user_df)

        # eğitim sırasında kullanılan sütunları yükle
        feature_names = joblib.load(
            ROOT /
            "models" /
            "detectors" /
            f"{dataset}_feature_names.pkl"
        )

        # eksik sütunları tamamla
        user_df = user_df.reindex(
            columns=feature_names,
            fill_value=0
        )

        synthetic_models = [
            "smote", "borderline_smote", "smote_tomek", "cgan",
            "ctgan", "copulagan", "ctabgan_plus", "tabsyn", "forest_diffusion"
        ]

        classifiers = [
            "lr", "rf", "adaboost", "xgb", "catboost"
        ]

        results = []
        votes = []

        for synth in synthetic_models:
            for clf in classifiers:
                model_path = (
                        ROOT
                        / "models"
                        / "detectors"
                        / f"{dataset}_{synth}_{clf}.pkl"
                )

                try:
                    model = joblib.load(model_path)
                    pred = model.predict(user_df)[0]

                    label = "Sentetik" if pred == 1 else "Gerçek"
                    votes.append(label)

                    pretty_synth = {
                        "smote": "SMOTE",
                        "borderline_smote": "Borderline-SMOTE",
                        "smote_tomek": "SMOTE-Tomek",
                        "cgan": "CGAN",
                        "ctgan": "CTGAN",
                        "copulagan": "CopulaGAN",
                        "ctabgan_plus": "CTAB-GAN+",
                        "tabsyn": "TabSyn",
                        "forest_diffusion": "ForestDiffusion"
                    }

                    pretty_clf = {
                        "lr": "Logistic Regression",
                        "rf": "Random Forest",
                        "adaboost": "AdaBoost",
                        "xgb": "XGBoost",
                        "catboost": "CatBoost"
                    }

                    results.append({
                        "Sentetik Model": pretty_synth[synth],
                        "Sınıflandırıcı": pretty_clf[clf],
                        "Tahmin": label
                    })

                except Exception as e:
                    results.append({
                        "Sentetik Model": synth,
                        "Classifier": clf,
                        "Tahmin": f"Hata: {e}"
                    })

        # =================================================
        # ÇOĞUNLUK OYLAMASI
        # =================================================

        counter = Counter(votes)

        real_votes = counter["Gerçek"]
        synthetic_votes = counter["Sentetik"]

        final_prediction = max(counter, key=counter.get)

        st.header("Sonuçlar")

        if final_prediction == "Gerçek":
            st.success(f"GERÇEK VERİ ({real_votes}/{len(votes)} oy)")
        else:
            st.error(f"SENTETİK VERİ ({synthetic_votes}/{len(votes)} oy)")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Gerçek Oy Sayısı", real_votes)

        with col2:
            st.metric("Sentetik Oy Sayısı", synthetic_votes)

        st.header("Model Sonuçları")
        result_df = pd.DataFrame(results)

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )