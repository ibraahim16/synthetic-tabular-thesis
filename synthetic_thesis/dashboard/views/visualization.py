import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =====================================================
# PCA VARYANS HESABI
# =====================================================

def get_pca_variance(df):

    num_df = df.select_dtypes(include=np.number)

    scaler = StandardScaler()
    X = scaler.fit_transform(num_df)

    pca = PCA(n_components=2)
    pca.fit(X)

    return {
        "PC1 (%)": round(
            pca.explained_variance_ratio_[0] * 100, 2
        ),

        "PC2 (%)": round(
            pca.explained_variance_ratio_[1] * 100, 2
        ),

        "Toplam (%)": round(
            pca.explained_variance_ratio_.sum() * 100, 2
        )
    }


# =====================================================
# PCA GRID ÇİZİMİ
# =====================================================

def draw_pca_grid(real_df, models, title):

    st.subheader(title)

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    for ax, (name, synth_df) in zip(
            axes.flat,
            models.items()
    ):

        real_num = real_df.select_dtypes(include=np.number)
        synth_num = synth_df.select_dtypes(include=np.number)

        cols = sorted(
            list(
                set(real_num.columns)
                .intersection(synth_num.columns)
            )
        )

        scaler = StandardScaler()

        real_scaled = scaler.fit_transform(
            real_num[cols]
        )

        synth_scaled = scaler.transform(
            synth_num[cols]
        )

        sample_size = min(
            2000,
            len(real_scaled),
            len(synth_scaled)
        )

        np.random.seed(42)

        idx_real = np.random.choice(
            len(real_scaled),
            sample_size,
            replace=False
        )

        idx_syn = np.random.choice(
            len(synth_scaled),
            sample_size,
            replace=False
        )

        real_scaled = real_scaled[idx_real]
        synth_scaled = synth_scaled[idx_syn]

        pca = PCA(n_components=2)

        real_pca = pca.fit_transform(real_scaled)
        synth_pca = pca.transform(synth_scaled)

        ax.scatter(
            real_pca[:, 0],
            real_pca[:, 1],
            s=3,
            alpha=0.30,
            label="Real"
        )

        ax.scatter(
            synth_pca[:, 0],
            synth_pca[:, 1],
            s=3,
            alpha=0.30,
            label="Synthetic"
        )

        ax.set_title(
            name,
            fontsize=7
        )

        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=8
    )

    plt.tight_layout(pad=0.8)

    st.pyplot(
        fig,
        use_container_width=True
    )


# =====================================================
# SAYFA
# =====================================================

def show():

    ROOT = Path(__file__).resolve().parents[2]

    # =====================================================
    # GERÇEK VERİLER
    # =====================================================

    real_adult = pd.read_csv(
        ROOT / "data/split/adult_train.csv"
    )

    real_credit = pd.read_csv(
        ROOT / "data/split/credit_train.csv"
    )

    # =====================================================
    # ADULT MODELLERİ
    # =====================================================

    adult_models = {

        "SMOTE":
            pd.read_csv(
                ROOT / "data/synthetic/adult/smote.csv"
            ),

        "Borderline-SMOTE":
            pd.read_csv(
                ROOT / "data/synthetic/adult/borderline_smote.csv"
            ),

        "SMOTE-Tomek":
            pd.read_csv(
                ROOT / "data/synthetic/adult/smote_tomek.csv"
            ),

        "CGAN":
            pd.read_csv(
                ROOT / "data/synthetic/adult/cgan.csv"
            ),

        "CTGAN":
            pd.read_csv(
                ROOT / "data/synthetic/adult/ctgan.csv"
            ),

        "CopulaGAN":
            pd.read_csv(
                ROOT / "data/synthetic/adult/copulagan.csv"
            ),

        "CTAB-GAN+":
            pd.read_csv(
                ROOT / "data/synthetic/adult/ctabgan_plus.csv"
            ),

        "TabSyn":
            pd.read_csv(
                ROOT / "data/synthetic/adult/tabsyn.csv"
            ),

        "ForestDiffusion":
            pd.read_csv(
                ROOT / "data/synthetic/adult/forest_diffusion.csv"
            )
    }

    # =====================================================
    # CREDIT MODELLERİ
    # =====================================================

    credit_models = {

        "SMOTE":
            pd.read_csv(
                ROOT / "data/synthetic/credit/smote.csv"
            ),

        "Borderline-SMOTE":
            pd.read_csv(
                ROOT / "data/synthetic/credit/borderline_smote.csv"
            ),

        "SMOTE-Tomek":
            pd.read_csv(
                ROOT / "data/synthetic/credit/smote_tomek.csv"
            ),

        "CGAN":
            pd.read_csv(
                ROOT / "data/synthetic/credit/cgan.csv"
            ),

        "CTGAN":
            pd.read_csv(
                ROOT / "data/synthetic/credit/ctgan.csv"
            ),

        "CopulaGAN":
            pd.read_csv(
                ROOT / "data/synthetic/credit/copulagan.csv"
            ),

        "CTAB-GAN+":
            pd.read_csv(
                ROOT / "data/synthetic/credit/ctabgan_plus.csv"
            ),

        "TabSyn":
            pd.read_csv(
                ROOT / "data/synthetic/credit/tabsyn.csv"
            ),

        "ForestDiffusion":
            pd.read_csv(
                ROOT / "data/synthetic/credit/forest_diffusion.csv"
            )
    }

    # =====================================================
    # BAŞLIK
    # =====================================================

    st.title("Görsel Benzerlik Analizi")

    st.markdown("""
    **Principal Component Analysis (PCA)**, yüksek boyutlu verileri
    daha düşük boyutlara indirerek gerçek ve sentetik veri
    dağılımlarını görselleştirmek amacıyla kullanılmaktadır.

    - **PCA**, verideki bilginin büyük bölümünü ilk iki temel bileşende (PC1 ve PC2) toplamaktadır.
    
    - **Açıklanan varyans oranı**, temel bileşenlerin verideki toplam değişkenliğin ne kadarını açıkladığını göstermektedir.
    
    - Gerçek ve sentetik verilerin benzer bölgelerde yoğunlaşması,
    sentetik verilerin gerçek veri dağılımını başarılı şekilde öğrendiğini göstermektedir.
    """)

    # =====================================================
    # VARYANS TABLOLARI
    # =====================================================

    st.header("Açıklanan Varyans Oranları")

    adult_rows = []

    adult_rows.append(
        {
            "Model": "Real",
            **get_pca_variance(real_adult)
        }
    )

    for name, df in adult_models.items():

        adult_rows.append(
            {
                "Model": name,
                **get_pca_variance(df)
            }
        )

    credit_rows = []

    credit_rows.append(
        {
            "Model": "Real",
            **get_pca_variance(real_credit)
        }
    )

    for name, df in credit_models.items():

        credit_rows.append(
            {
                "Model": name,
                **get_pca_variance(df)
            }
        )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Adult Veri Seti")

        st.dataframe(
            pd.DataFrame(adult_rows),
            use_container_width=True,
            hide_index=True
        )

    with col2:

        st.subheader("Credit Veri Seti")

        st.dataframe(
            pd.DataFrame(credit_rows),
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # PCA GRAFİKLERİ
    # =====================================================

    st.header("PCA Dağılım Grafikleri")

    col1, col2 = st.columns(2)

    with col1:

        draw_pca_grid(
            real_adult,
            adult_models,
            "Adult Veri Seti"
        )

    with col2:

        draw_pca_grid(
            real_credit,
            credit_models,
            "Credit Veri Seti"
        )

    st.header("Değerlendirme")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Adult Veri Seti")

        st.markdown("""
        - **SMOTE tabanlı yöntemler** ve **TabSyn** modellerinin PCA dağılımları gerçek veri ile büyük ölçüde örtüşmektedir. Bu durum, modellerin gerçek veri dağılımını başarılı şekilde öğrendiğini göstermektedir.

        - **CTAB-GAN+** modeli genel dağılım yapısını korumakla birlikte bazı bölgelerde gerçek veriden kısmen ayrışmaktadır.

        - **CGAN** modeli ise gerçek veri dağılımından belirgin şekilde sapmakta ve veri yoğunluğunu farklı bölgelerde oluşturmaktadır. Bu nedenle görsel benzerlik açısından en düşük performansı göstermektedir.
        - **Adult veri setinde**, SMOTE tabanlı yöntemler, TabSyn, CTAB-GAN+ ve CopulaGAN modelleri gerçek veriye oldukça yakın açıklanan varyans oranları üretmiştir. Buna karşın **CGAN** modeli, gerçek veriden belirgin şekilde saparak en yüksek varyans oranlarını göstermiştir.
        """)

    with col2:

        st.subheader("Credit Veri Seti")

        st.markdown("""
        - **SMOTE tabanlı yöntemler** gerçek veri dağılımını yüksek doğrulukla korumuş ve PCA uzayında gerçek veriler ile büyük ölçüde çakışmıştır.

        - **TabSyn** modeli gerçek veri yapısını başarılı şekilde öğrenmiş ve veri yoğunluklarını büyük ölçüde korumuştur.

        - **CTGAN**, **CopulaGAN** ve **CTAB-GAN+** modelleri genel dağılım eğilimini yakalamakla birlikte veri yoğunluklarında belirli sapmalar göstermektedir.

        - **CGAN** modeli gerçek veri dağılımından önemli ölçüde sapmakta ve PCA uzayında farklı bölgelerde yoğunlaşmaktadır. Bu nedenle görsel benzerlik açısından en düşük performansı sergilemektedir.
        - **Credit veri setinde**, TabSyn modeli gerçek veri ile benzer açıklanan varyans oranları üretmiştir. SMOTE tabanlı yöntemler, CTGAN ve CopulaGAN modelleri ise gerçek veriye kıyasla daha yüksek varyans oranları oluşturmuştur.

        """)