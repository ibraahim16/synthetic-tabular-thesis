import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

def plot_corr_diff(real_df, synth_df, title):

    real_num = real_df.select_dtypes(include=np.number)
    synth_num = synth_df.select_dtypes(include=np.number)

    common_cols = sorted(
        list(
            set(real_num.columns)
            .intersection(synth_num.columns)
        )
    )

    real_corr = real_num[common_cols].corr()
    synth_corr = synth_num[common_cols].corr()

    diff = (real_corr - synth_corr).abs()

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        diff,
        cmap="coolwarm",
        vmin=0,
        vmax=0.10,
        annot=True,
        fmt=".2f",
        square=True,
        cbar=False,
        ax=ax
    )

    ax.set_title(title, fontsize=12)

    return fig

def show():
    ROOT = Path(__file__).resolve().parents[2]

    real_adult = pd.read_csv(
        ROOT / "data/split/adult_train.csv"
    )

    real_credit = pd.read_csv(
        ROOT / "data/split/credit_train.csv"
    )

    adult_models = {
        "SMOTE": pd.read_csv(
            ROOT / "data/synthetic/adult/smote.csv"
        ),

        "Borderline-SMOTE": pd.read_csv(
            ROOT / "data/synthetic/adult/borderline_smote.csv"
        ),

        "SMOTE-Tomek": pd.read_csv(
            ROOT / "data/synthetic/adult/smote_tomek.csv"
        ),

        "CGAN": pd.read_csv(
            ROOT / "data/synthetic/adult/cgan.csv"
        ),

        "CTGAN": pd.read_csv(
            ROOT / "data/synthetic/adult/ctgan.csv"
        ),

        "CopulaGAN": pd.read_csv(
            ROOT / "data/synthetic/adult/copulagan.csv"
        ),

        "CTAB-GAN+": pd.read_csv(
            ROOT / "data/synthetic/adult/ctabgan_plus.csv"
        ),

        "TabSyn": pd.read_csv(
            ROOT / "data/synthetic/adult/tabsyn.csv"
        ),

        "ForestDiffusion": pd.read_csv(
            ROOT / "data/synthetic/adult/forest_diffusion.csv"
        )
    }

    credit_models = {
        "SMOTE": pd.read_csv(
            ROOT / "data/synthetic/credit/smote.csv"
        ),

        "Borderline-SMOTE": pd.read_csv(
            ROOT / "data/synthetic/credit/borderline_smote.csv"
        ),

        "SMOTE-Tomek": pd.read_csv(
            ROOT / "data/synthetic/credit/smote_tomek.csv"
        ),

        "CGAN": pd.read_csv(
            ROOT / "data/synthetic/credit/cgan.csv"
        ),

        "CTGAN": pd.read_csv(
            ROOT / "data/synthetic/credit/ctgan.csv"
        ),

        "CopulaGAN": pd.read_csv(
            ROOT / "data/synthetic/credit/copulagan.csv"
        ),

        "CTAB-GAN+": pd.read_csv(
            ROOT / "data/synthetic/credit/ctabgan_plus.csv"
        ),

        "TabSyn": pd.read_csv(
            ROOT / "data/synthetic/credit/tabsyn.csv"
        ),

        "ForestDiffusion": pd.read_csv(
            ROOT / "data/synthetic/credit/forest_diffusion.csv"
        )
    }

    # =====================================================
    # BAŞLIK
    # =====================================================

    st.title("İstatistiksel Analiz")

    st.markdown("""
    **İstatistiksel Analiz**, sentetik verilerin gerçek veri
    dağılımını ne ölçüde koruduğunu değerlendirmek amacıyla
    kullanılmaktadır.
    """)


    # =====================================================
    # METRİKLER
    # =====================================================

    st.markdown("""
    - **KS (Kolmogorov-Smirnov):** Sayısal değişkenlerin dağılım benzerliğini ölçmektedir.

    - **Wasserstein (Earth Mover's Distance):** Sayısal değişkenler arasındaki dağılım uzaklığını ölçmektedir.

    - **JS (Jensen-Shannon Distance):** Kategorik değişkenlerde sınıf oranlarının benzerliğini ölçmektedir.

    - **Chi2 (Chi-Square):** Kategorik değişkenlerin dağılımları arasındaki farklılığı değerlendirmektedir.

    - **Düşük metrik değerleri, sentetik verilerin gerçek verilere daha fazla benzediğini göstermektedir.**
    """)

    # =====================================================
    # TABLOLAR
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Adult Veri Seti")

        st.image(
            "assets/images/adult_statistical_summary.png",
            use_container_width=True
        )

    with col2:

        st.subheader("Credit Veri Seti")

        st.image(
            "assets/images/credit_statistical_summary.png",
            use_container_width=True
        )

    # =====================================================
    # DEĞERLENDİRME
    # =====================================================

    st.header("Değerlendirme")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Adult Veri Seti")

        st.markdown("""
        - **SMOTE tabanlı yöntemler** ve **TabSyn**, en düşük hata değerlerini üreterek gerçek veriye en yakın sonuçları sağlamıştır.

        - **CTAB-GAN+** ve **CopulaGAN** modelleri genel olarak başarılı sonuçlar üretmiştir.

        - **CTGAN** ve **ForestDiffusion** modelleri orta düzey performans göstermiştir.

        - **CGAN** modeli tüm metriklerde yüksek hata değerleri üreterek en düşük performansı sergilemiştir.
        """)

    with col2:
        st.subheader("Credit Veri Seti")

        st.markdown("""
        - **TabSyn** modeli tüm metriklerde en başarılı sonuçları üretmiştir.

        - **SMOTE tabanlı yöntemler** ve **ForestDiffusion** modelleri de düşük hata değerleri ile başarılı performans göstermiştir.

        - **CTAB-GAN+** modeli kategorik değişkenlerde başarılı sonuçlar üretmiştir.

        - **CTGAN** ve **CopulaGAN** modelleri özellikle Chi2 metriğinde yüksek hata değerleri üretmiştir.

        - **CGAN** modeli sayısal değişkenlerde yüksek hata değerleri üreterek düşük performans sergilemiştir.
        """)

    st.header("Korelasyon Analizi")

    st.markdown("""
    - **Korelasyon**, değişkenler arasındaki ilişkinin yönünü ve gücünü göstermektedir.

    - Korelasyon fark matrisi, gerçek ve sentetik veriler arasındaki ilişki yapısının ne ölçüde korunduğunu değerlendirmektedir.

    - **Mavi renkler** düşük korelasyon farkını, **kırmızı renkler** ise yüksek korelasyon farkını göstermektedir.
    """)

    col1, col2 = st.columns(2)

    # =====================================================
    # ADULT
    # =====================================================

    with col1:

        st.subheader("Adult Veri Seti")

        fig, axes = plt.subplots(3, 3, figsize=(7, 7))

        for ax, (name, synth_df) in zip(axes.flat, adult_models.items()):
            real_num = real_adult.select_dtypes(include=np.number)
            synth_num = synth_df.select_dtypes(include=np.number)

            cols = sorted(
                list(set(real_num.columns)
                     .intersection(synth_num.columns))
            )

            diff = (
                    real_num[cols].corr()
                    - synth_num[cols].corr()
            ).abs()

            sns.heatmap(
                diff,
                cmap="coolwarm",
                vmin=0,
                vmax=0.5,
                annot=True,
                fmt=".2f",
                annot_kws={"size": 4},
                square=True,
                cbar=False,
                ax=ax
            )

            ax.set_title(name, fontsize=8)
            ax.tick_params(axis='x', labelsize=5)
            ax.tick_params(axis='y', labelsize=5)

        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)

    # =====================================================
    # CREDIT
    # =====================================================

    with col2:

        st.subheader("Credit Veri Seti")

        fig, axes = plt.subplots(3, 3, figsize=(7, 7))

        for ax, (name, synth_df) in zip(axes.flat, credit_models.items()):
            real_num = real_credit.select_dtypes(include=np.number)
            synth_num = synth_df.select_dtypes(include=np.number)

            cols = real_num.columns[:10]

            diff = (
                    real_num[cols].corr()
                    - synth_num[cols][cols].corr()
            ).abs()

            sns.heatmap(
                diff,
                cmap="coolwarm",
                vmin=0,
                vmax=0.5,
                annot=True,
                fmt=".2f",
                annot_kws={"size": 4},
                square=True,
                cbar=False,
                ax=ax
            )

            ax.set_title(name, fontsize=8)
            ax.tick_params(axis='x', labelsize=5)
            ax.tick_params(axis='y', labelsize=5)

        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)

    st.header("Değerlendirme")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Adult Veri Seti")

        st.markdown("""
        - **TabSyn** ve **Smote tabanlı yöntemler** modelleri değişkenler arasındaki ilişki yapısını büyük ölçüde korumuştur.

        - **CTAB-GAN+** ve **CopulaGAN** modelleri düşük korelasyon farkları ile başarılı sonuçlar üretmiştir.

        - **ForestDiffusion** modeli orta düzey korelasyon benzerliği göstermiştir.

        - **CGAN** modeli özellikle bazı değişken çiftlerinde yüksek korelasyon farkları üreterek en düşük performansı sergilemiştir.
        """)

    with col2:

        st.subheader("Credit Veri Seti")

        st.markdown("""
        - **TabSyn** modeli en düşük korelasyon farklarını üreterek değişkenler arasındaki ilişki yapısını en başarılı şekilde korumuştur.

        - **ForestDiffusion** modeli de düşük korelasyon farkları ile başarılı performans göstermiştir.

        - **SMOTE tabanlı yöntemler** orta düzey korelasyon benzerliği göstermiştir.

        - **CTGAN**, **CopulaGAN** ve özellikle **CGAN** modelleri yüksek korelasyon farkları üreterek değişkenler arasındaki ilişki yapısını yeterince koruyamamıştır.
        """)