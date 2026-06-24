import streamlit as st
import matplotlib.pyplot as plt

def show():

    st.title("Üretken Ağlar ile Tabular Sentetik Veri Üretimi")

    st.header("Proje Amacı")

    st.markdown("""
    - Sentetik veri üretim modellerinin gerçek veri dağılımını koruma başarısını değerlendirmek

    - Üretilen verilerin istatistiksel ve görsel benzerliklerini analiz etmek

    - Sentetik verilerin makine öğrenmesi performanslarını karşılaştırmak

    - Veri setine bağlı olarak en başarılı sentetik veri üretim yöntemlerini belirlemek
    """)

    st.header("Sentetik Veri Üretimi Neden Gereklidir?")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="
            background-color:#EFF6FF;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #2563EB;
            height:240px;">
            <h3>🔒 Gizlilik</h3>
            <p>Gerçek verilerin paylaşımı, gizlilik ihlali ve güvenlik riskleri oluşturabilmektedir.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background-color:#ECFDF5;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #10B981;
            height:240px;">
            <h3>📁 Veri Erişimi</h3>
            <p>Sağlık, finans ve kamu alanlarında gerçek verilere erişim çoğu zaman kısıtlıdır.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="
            background-color:#F5F3FF;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #8B5CF6;
            height:240px;">
            <h3>📈 Veri Artırma</h3>
            <p>Sentetik veriler mevcut veri miktarını artırarak model eğitimini desteklemektedir.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="
            background-color:#FEF2F2;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #EF4444;
            height:240px;">
            <h3>🎯 Temsil Yeteneği</h3>
            <p>Sentetik veriler gerçek verilerin istatistiksel özelliklerini korumayı hedefler.</p>
        </div>
        """, unsafe_allow_html=True)

    st.header("Kullanılan Veri Setleri")

    col1, col2 = st.columns(2)

    # =====================================================
    # ADULT
    # =====================================================

    with col1:
        st.info("""
        ## Adult Income Veri Seti

        **Amaç:** Bireylerin yıllık gelir düzeyini
        tahmin etmeye yönelik bir sınıflandırma veri setidir.

        **Problem Türü:** Gelir Tahmini

        **Hedef Değişken:** income

        **Sınıflar:**
        - <=50K
        - \>50K

        **Veri Özellikleri**
        - Toplam Örnek Sayısı: 32.561
        - Toplam Özellik Sayısı: 14
        - Sayısal Özellik: 6
        - Kategorik Özellik: 8
        
        **Bu Çalışmada Kullanılma Nedeni**
        - Sayısal ve kategorik değişkenleri birlikte içermesi
        - Karma veri tipleri üzerinde sentetik veri üretim performansını değerlendirebilmesi
        - Literatürde yaygın olarak kullanılan bir veri seti olması
        """)
    # =====================================================
    # CREDIT
    # =====================================================

    with col2:
        st.warning("""
        ## Credit Card Fraud Veri Seti

        **Amaç:** Kredi kartı işlemlerinin dolandırıcılık
        içerip içermediğinin belirlenmesidir.

        **Problem Türü:** Dolandırıcılık Tespiti

        **Hedef Değişken:** Class

        **Sınıflar:**
        - 0 → Normal İşlem
        - 1 → Dolandırıcılık

        **Veri Özellikleri**
        - Toplam Örnek Sayısı: 284.807
        - Toplam Özellik Sayısı: 30

        **Bu Çalışmada Kullanılma Nedeni**
        - Yüksek sınıf dengesizliği içermesi
        - Gizlilik nedeniyle PCA ile anonimleştirilmiş finansal verilerden oluşması
        """)

    graph_col1, graph_col2 = st.columns(2)
    with graph_col1:
        st.header("Gelir Sınıf Dağılımı")

        fig, ax = plt.subplots(figsize=(2, 2))

        ax.pie(
            [76, 24],
            labels=["<=50K", ">50K"],
            autopct="%1.1f%%",
            textprops={"fontsize": 5},
            pctdistance=0.7,
            labeldistance=1.05
        )

        ax.set_title(
            "Adult Income",
            fontsize=8
        )

        fig.tight_layout()

        # Streamlit'in resmi büyütmesini engelle
        st.pyplot(fig, use_container_width=False)

    with graph_col2:
        st.header("Dolandırıcılık Sınıf Dağılımı")

        fig, ax = plt.subplots(figsize=(2, 2))

        ax.pie(
            [99.83, 0.17],
            labels=["Normal", "Fraud"],
            autopct="%1.2f%%",
            textprops={"fontsize": 5},
            pctdistance=0.7,
            labeldistance=1.05
        )

        ax.set_title(
            "Credit Fraud",
            fontsize=8
        )

        fig.tight_layout()

        # Streamlit'in resmi büyütmesini engelle
        st.pyplot(fig, use_container_width=False)

    st.header("Kullanılan Modeller")

    col1, col2, col3, col4 = st.columns(4)

    # =====================================================
    # SMOTE
    # =====================================================

    with col1:
        st.markdown("""
        <div style="
            background-color:#EFF6FF;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #2563EB;
            height:420px;">
            <h3>🔵 SMOTE Tabanlı</h3>
            <p>
    <b>SMOTE:</b>
    Azınlık sınıfı örnekleri arasından yeni sentetik örnekler üretir.
</p>

<p>
    <b>Borderline-SMOTE:</b>
    Karar sınırına yakın örnekleri çoğaltarak sınıflandırma başarısını artırmayı hedefler.
</p>

<p>
    <b>SMOTE-Tomek:</b>
    Sentetik örnek üretimi sonrası gürültülü verileri temizler.
</p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # GAN
    # =====================================================

    with col2:
        st.markdown("""
        <div style="
            background-color:#F5F3FF;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #8B5CF6;
            height:420px;">
            <h3>🟣 GAN Tabanlı</h3>
            <p><b>CGAN:</b> Sınıf bilgisi kullanarak koşullu sentetik veri üretir.</p>
            <p><b>CTGAN:</b> Tabular veriler için geliştirilmiş GAN tabanlı bir yöntemdir.</p>
            <p><b>CopulaGAN:</b> Değişkenler arasındaki istatistiksel bağımlılıkları korumayı amaçlar.</p>
            <p><b>CTAB-GAN+:</b> Karma ve dengesiz tabular veriler için optimize edilmiştir.</p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # DIFFUSION
    # =====================================================

    with col3:
        st.markdown("""
        <div style="
            background-color:#ECFDF5;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #10B981;
            height:420px;">
            <h3>🟢 Difüzyon Tabanlı</h3>
            <p><b>ForestDiffusion:</b> Difüzyon süreci ve karar ağaçlarını birleştirerek veri üretir.</p>
            <p><b>TabSyn:</b> Verileri özellik türlerine göre farklı gruplara ayırarak işlemektedir. Bu yaklaşım değişkenler arasındaki karmaşık ilişkilerin daha başarılı öğrenilmesini sağlamaktadır.</p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # AUTOGLUON
    # =====================================================

    with col4:
        st.markdown("""
        <div style="
            background-color:#FFFBEB;
            padding:20px;
            border-radius:15px;
            border-left:8px solid #F59E0B;
            height:420px;">
            <h3>🔶 AutoGluon</h3>
            <br>
            • Amazon tarafından geliştirilen açık kaynaklı bir AutoML platformudur.
            <br><br>
            • Farklı makine öğrenmesi algoritmalarını otomatik olarak eğitmekte ve optimize etmektedir.
            <br><br>
            • Ensemble yaklaşımı kullanarak yüksek tahmin performansı elde etmeyi amaçlamaktadır.
        </div>
        """, unsafe_allow_html=True)