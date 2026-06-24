import streamlit as st
import pandas as pd
from pathlib import Path
import streamlit.components.v1 as components

# =====================================================
# İSİM EŞLEŞTİRMELERİ
# =====================================================

MODEL_NAME_MAP = {
    "baseline": "Baseline",
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

CLASSIFIER_MAP = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "adaboost": "AdaBoost",
    "autogluon": "AutoGluon"
}

# =====================================================
# CSS STİLLERİ (Ortak)
# =====================================================
TABLE_CSS = """
<style>
table{
    width:100%;
    border-collapse:collapse;
    text-align:center;
    font-family:Arial;
}
th,td{
    border:1px solid #CBD5E1;
    padding:6px;
    font-size:13px;
}
th{
    background:#1E3A8A;
    color:white;
    font-weight:bold;
    position:sticky;
    top:0;
    z-index:2;
}
td{
    background:white;
    color:black;
}
tr:hover td{
    background:#F8FAFC;
}
td.dataset{
    background:white !important;
    color:#1E293B !important;
    font-weight:bold !important;
    font-size:18px !important;
}
.baseline .dataset,
.best .dataset,
.worst .dataset{
    background:white !important;
}
.baseline td{
    background:#DBEAFE !important;
}
.best td{
    background:#DCFCE7 !important;
}
.worst td{
    background:#FEE2E2 !important;
}
.separator td{
    border-top:4px solid #1E3A8A;
    padding:0;
    height:0;
    background:transparent !important;
    border-left:none;
    border-right:none;
    border-bottom:none;
}
</style>
"""


# =====================================================
# SINIFLANDIRMA VERİLERİNİ YÜKLE
# =====================================================

def load_all_em_classification_results():
    rows = []
    for dataset_name in ["adult", "credit"]:
        em_dir = Path("../outputs/em") / dataset_name

        if not em_dir.exists():
            continue

        files = list(em_dir.glob("*_em_classification.csv"))

        for file in files:
            model_key = file.stem.replace("_em_classification", "")
            model_name = MODEL_NAME_MAP.get(model_key, model_key)

            df = pd.read_csv(file)
            row = {"Veri Seti": dataset_name.capitalize(), "Model": model_name}

            for _, r in df.iterrows():
                clf = CLASSIFIER_MAP.get(r["model"], r["model"])
                row[f"{clf} Precision"] = round(r["precision"], 3)
                row[f"{clf} Recall"] = round(r["recall"], 3)
                row[f"{clf} F1"] = round(r["f1"], 3)
                row[f"{clf} Accuracy"] = round(r["accuracy"], 3)

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    model_order = list(MODEL_NAME_MAP.values())

    result_df["Model"] = pd.Categorical(
        result_df["Model"],
        categories=model_order,
        ordered=True
    )
    result_df = result_df.sort_values(by=["Veri Seti", "Model"])

    return result_df

# =====================================================
# SINIFLANDIRMA HTML TABLOSU
# =====================================================

def create_classification_html_table(df):
    classifiers = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost", "AdaBoost", "AutoGluon"]

    html = TABLE_CSS + "<table>"

    # Üst başlık
    html += "<tr><th rowspan='2'>Veri Seti</th><th rowspan='2'>Model</th>"
    for clf in classifiers:
        html += f"<th colspan='2'>{clf}</th>"
    html += "</tr>"

    # Alt başlık
    html += "<tr>"
    for _ in classifiers:
        html += "<th>F1 Score</th><th>Accuracy</th>"
    html += "</tr>"

    # Satırlar
    for dataset in ["Adult", "Credit"]:
        subset = df[df["Veri Seti"] == dataset]
        first = True

        for _, row in subset.iterrows():
            row_class = ""
            if row["Model"] == "Baseline":
                row_class = "baseline"

            # Yeşil
            elif (
                    (dataset == "Adult" and row["Model"] == "TabSyn")
                    or
                    (dataset == "Credit" and row["Model"] in [
                        "SMOTE",
                        "Borderline-SMOTE",
                        "SMOTE-Tomek",
                        "TabSyn",
                        "ForestDiffusion"
                    ])
            ):
                row_class = "best"

            # Kırmızı
            elif row["Model"] == "CGAN":
                row_class = "worst"

            html += f"<tr class='{row_class}'>"

            if first:
                html += f"<td rowspan='{len(subset)}' class='dataset'>{dataset}</td>"
                first = False

            html += f"<td>{row['Model']}</td>"

            for clf in classifiers:
                for metric in ["F1", "Accuracy"]:
                    value = row.get(f"{clf} {metric}", "-")
                    if value != "-":
                        value = f"{float(value):.2f}"
                    html += f"<td>{value}</td>"

            html += "</tr>"

            if dataset == "Adult" and row.name == subset.index[-1]:
                html += "<tr class='separator'><td colspan='14'></td></tr>"

    html += "</table>"
    return html


# =====================================================
# SAYFA
# =====================================================

def show():
    st.title("Kullanılabilirlik Analizi")

    st.markdown("""
    <style>
    div[data-testid="stMarkdownContainer"] {
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Kullanılabilirlik Analizi (train synthetic test real)**,
    sentetik veriler kullanılarak eğitilen modellerin gerçek
    veriler üzerindeki performansını değerlendirmek amacıyla kullanılmaktadır.
    """)

    # ---------------------------------------------------
    # İŞLEM AKIŞI
    # ---------------------------------------------------
    st.header("İşlem Akışı")

    st.image(
        "assets/images/em_pipeline.png",
        width=800
    )

    st.markdown("""
    - Makine öğrenmesi modelleri sentetik veri kümesi ile eğitilir.

    - Eğitilen modeller gerçek veri kümesi üzerinde test edilir.

    - Sınıflandırma ve regresyon performansları değerlendirilir.

    - Gerçek veride yüksek başarı elde edilmesi, sentetik verilerin kullanılabilir olduğunu göstermektedir.
    """)

    # ---------------------------------------------------
    # CLASSIFICATION
    # ---------------------------------------------------
    st.header("Kullanılabilirlik Analizi Sonuçları")

    cls_df = load_all_em_classification_results()

    if not cls_df.empty:
        html_table = create_classification_html_table(cls_df)
        st.components.v1.html(
            html_table,
            height=600,
            scrolling=True
        )
    else:
        st.warning("Efficacy Metric (Classification) sonuç dosyaları bulunamadı.")

    # ---------------------------------------------------
    # CLASSIFICATION YORUM
    # ---------------------------------------------------
    st.subheader("Değerlendirme")

    st.markdown("""
    - **Baseline (TRTR):** Gerçek veriler ile eğitilen ve gerçek veriler üzerinde test edilen referans performansı göstermektedir.

    - **Accuracy:** Doğru sınıflandırma oranını göstermektedir.

    - **F1 Score:** Genel sınıflandırma performansını ölçmektedir.

    - **Baseline'a yakın skorlar, sentetik verilerin gerçek veriler üzerinde başarılı şekilde kullanılabildiğini göstermektedir.**
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Adult Veri Seti")

        st.markdown("""
        - **SMOTE tabanlı yöntemler** ve **TabSyn**, Baseline sonuçlarına en yakın performansı göstermiştir.

        - **CTGAN**, **CopulaGAN** ve **CTAB-GAN+** modelleri genel olarak başarılı sonuçlar üretmiştir.

        - **ForestDiffusion** modeli Baseline'a yakın olsa da diğer başarılı modellere göre daha düşük performans göstermiştir.

        - **CGAN** modeli özellikle Logistic Regression sınıflandırıcısında belirgin performans kaybı yaşamış ve en düşük sonuçları üretmiştir.
        """)

    with col2:
        st.subheader("Credit Veri Seti")

        st.markdown("""
        - **TabSyn**, **SMOTE tabanlı yöntemler** ve **ForestDiffusion** modelleri Baseline ile aynı performansı elde etmiştir.

        - **CTGAN**, **CopulaGAN** ve **CTAB-GAN+** modelleri de Baseline'a oldukça yakın sonuçlar üretmiştir.

        - Credit veri setinde modellerin büyük çoğunluğu yüksek kullanılabilirlik göstermiştir.

        - **CGAN** modeli diğer modellere kıyasla daha düşük performans sergilemiştir.
        """)