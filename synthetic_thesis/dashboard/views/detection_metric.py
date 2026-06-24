
import streamlit as st
import pandas as pd
from pathlib import Path
import streamlit.components.v1 as components
# =====================================================
# İSİM EŞLEŞTİRMELERİ
# =====================================================

MODEL_NAME_MAP = {
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
# TÜM DM SONUÇLARINI YÜKLE
# =====================================================

def load_all_dm_results():

    rows = []

    for dataset_name in ["adult", "credit"]:

        dm_dir = Path("../outputs/dm") / dataset_name

        if not dm_dir.exists():
            continue

        files = list(dm_dir.glob("*_dm_results.csv"))

        for file in files:

            model_key = file.stem.replace(
                "_dm_results",
                ""
            )

            model_name = MODEL_NAME_MAP.get(
                model_key,
                model_key
            )

            df = pd.read_csv(file)

            row = {
                "Veri Seti": dataset_name.capitalize(),
                "Model": model_name
            }

            for _, r in df.iterrows():

                clf = CLASSIFIER_MAP.get(
                    r["model"],
                    r["model"]
                )

                row[f"{clf} Precision"] = round(
                    r["precision"], 3
                )

                row[f"{clf} Recall"] = round(
                    r["recall"], 3
                )

                row[f"{clf} F1"] = round(
                    r["f1"], 3
                )

                row[f"{clf} Accuracy"] = round(
                    r["accuracy"], 3
                )

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

    result_df = result_df.sort_values(
        by=["Veri Seti", "Model"]
    )

    return result_df

def create_html_table(df):

    classifiers = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "CatBoost",
        "AdaBoost",
        "AutoGluon"
    ]

    html = """
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

.dataset{
    background:#F8FAFC !important;
    color:#1E293B;
    font-weight:bold;
    font-size:18px;
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

    <table>
    """

    # Üst başlık
    html += "<tr>"
    html += "<th rowspan='2'>Veri Seti</th>"
    html += "<th rowspan='2'>Model</th>"

    for clf in classifiers:
        html += f"<th colspan='2'>{clf}</th>"

    html += "</tr>"

    # Alt başlık
    html += "<tr>"

    for _ in classifiers:
        html += """
        <th>F1 Score</th>
        <th>Accuracy</th>
        """

    html += "</tr>"

    # Satırlar
    for dataset in ["Adult", "Credit"]:

        subset = df[df["Veri Seti"] == dataset]

        first = True

        for _, row in subset.iterrows():

            row_class = ""

            if row["Model"] == "TabSyn":
                row_class = "best"

            elif row["Model"] == "CGAN":
                row_class = "worst"

            html += f"<tr class='{row_class}'>"

            if first:
                html += f"""
                <td rowspan='{len(subset)}'
                    class='dataset'>
                    {dataset}
                </td>
                """
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
                html += """
                <tr class='separator'>
                    <td colspan='14'></td>
                </tr>
                """

    html += "</table>"

    return html

# =====================================================
# SAYFA
# =====================================================

def show():

    st.title("Ayırt Edilebilirlik Analizi")

    st.markdown("""
    <style>
    div[data-testid="stMarkdownContainer"] {
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Ayırt Edilebilirlik Analizi**, üretilen sentetik verilerin
    gerçek verilerden ne ölçüde ayırt edilebildiğini
    değerlendirmek amacıyla kullanılmaktadır.
    """)

    st.header("İşlem Akışı")

    st.image(
        "assets/images/dm_pipeline.png",
        width=900
    )

    st.markdown("""
    - Gerçek ve sentetik veriler birleştirilir.

    - Her kayıt gerçek (**0**) veya sentetik (**1**) olarak etiketlenir.

    - Veri kümesi eğitim ve test verisi olarak ikiye ayrılır.

    - Çeşitli makine öğrenmesi algoritmaları ile sınıflandırma yapılır.
    """)

    st.header("Ayırt Edilebilirlik Analizi Sonuçları")

    dm_df = load_all_dm_results()

    if not dm_df.empty:

        html_table = create_html_table(dm_df)

        st.components.v1.html(
            html_table,
            height=600,
            scrolling=True
        )

    else:

        st.warning(
            "Detection Metric sonuç dosyaları bulunamadı."
        )

    st.header("Değerlendirme")

    st.markdown("""
    - **Accuracy:** Doğru sınıflandırma oranını göstermektedir.

    - **F1 Score:** Genel sınıflandırma performansını ölçmektedir.

    - **Düşük F1 ve Accuracy değerleri, daha yüksek sentetik veri kalitesine işaret etmektedir.**
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Adult Veri Seti")

        st.markdown("""
        - **TabSyn**, en düşük ayırt edilebilirlik skorlarını üreterek en başarılı model olmuştur.

        - **SMOTE tabanlı yöntemler** düşük skorlar elde ederek gerçek veriye benzer örnekler üretmiştir.

        - **CGAN**, tüm sınıflandırıcılarda yüksek skorlar elde etmiş ve en düşük performansı göstermiştir.
        """)

    with col2:

        st.subheader("Credit Veri Seti")

        st.markdown("""
        - **TabSyn**, en düşük ayırt edilebilirlik skorlarını üreterek en başarılı model olmuştur.

        - **SMOTE tabanlı yöntemler** başarılı sonuçlar göstermiştir.

        - **CGAN**, yaklaşık **1.00** skorlarına ulaşarak en düşük performansı sergilemiştir.
        """)
