import streamlit as st
import pandas as pd


def show():

    st.title("Genel Sonuçlar ve Değerlendirme")
    # =====================================================
    # VERİ SETİ BAZLI SIRALAMA
    # =====================================================
    col1, col2, col3 = st.columns(3)

    # =====================================================
    # ADULT
    # =====================================================

    with col1:
        st.markdown("""
        <div style="
            background-color:#EFF6FF;
            padding:25px;
            border-radius:15px;
            border-left:8px solid #2563EB;
            height:220px;">
            <h3>Adult Income Veri Seti</h3>
            <p style="font-size:20px;">🥇 TabSyn</p>
            <p style="font-size:20px;">🥈 SMOTE tabanlı yöntemler</p>
            <p style="font-size:20px;">🥉 CTAB-GAN+</p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # CREDIT
    # =====================================================

    with col2:
        st.markdown("""
        <div style="
            background-color:#ECFDF5;
            padding:25px;
            border-radius:15px;
            border-left:8px solid #10B981;
            height:220px;">
            <h3>Credit Fraud Veri Seti</h3>
            <p style="font-size:20px;">🥇 TabSyn</p>
            <p style="font-size:20px;">🥈 SMOTE tabanlı yöntemler</p>
            <p style="font-size:20px;">🥉 ForestDiffusion</p>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # GENEL
    # =====================================================

    with col3:
        st.markdown("""
        <div style="
            background-color:#FEF3C7;
            padding:25px;
            border-radius:15px;
            border-left:8px solid #F59E0B;
            height:220px;">
            <h3>Genel Sıralama</h3>
            <p style="font-size:20px;">🥇 TabSyn</p>
            <p style="font-size:20px;">🥈 SMOTE tabanlı yöntemler</p>
            <p style="font-size:20px;">🥉 ForestDiffusion</p>

        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # METRİK TABLOSU
    # =====================================================

    st.header("Metrik Özetleri")

    summary_df = pd.DataFrame({

        "Kriter": [
            "İstatistiksel Benzerlik",
            "Görsel Benzerlik",
            "Ayırt Edilebilirlik",
            "Kullanılabilirlik"
        ],

        "Adult En İyi": [
            "TabSyn",
            "SMOTE",
            "TabSyn",
            "TabSyn"
        ],

        "Credit En İyi": [
            "TabSyn",
            "ForestDiffusion",
            "TabSyn",
            "SMOTE-Tomek"
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # GÖZLEMLER
    # =====================================================

    st.header("Genel Sonuçlar")

    col1, col2 = st.columns(2)

    with col1:

        st.success("""
        ### 🟢 Güçlü Yönler
        - **TabSyn** her iki veri setinde de en başarılı yöntemlerden biri olarak öne çıkmıştır.

        - ForestDiffusion özellikle Credit Fraud veri setinde başarılı sonuçlar üretmiştir.

        - CTAB-GAN+ karma veri tiplerinde güçlü performans sergilemiştir.
        - CTAB-GAN+ Adult veri setinde rekabetçi performans göstermiştir.

        - SMOTE tabanlı yöntemler birçok metrikte etkili sonuçlar vermiştir.
        """)

    with col2:

        st.error("""
        ### 🔴 Zayıf Yönler
        - **CGAN** birçok değerlendirme metriğinde düşük performans göstermiştir.

        - Bazı modeller değişkenler arasındaki bağımlılıkları korumakta yetersiz kalmıştır.
        
         - Tek bir model tüm veri setlerinde ve tüm metriklerde en iyi sonucu verememiştir.
        
        - Veri setine bağlı olarak model performanslarında önemli değişimler gözlenmiştir.
        """)

    st.markdown("""
    <div style="
        background-color:#DBEAFE;
        padding:30px;
        border-radius:15px;
        border-left:8px solid #2563EB;">
        <h2>Genel Değerlendirme</h2>
        <p>
        Bu çalışma kapsamında değerlendirilen yöntemler arasında
        <b>TabSyn</b> modeli genel olarak en başarılı sentetik veri
        üretim yöntemi olarak belirlenmiştir.
        </p>
        <p>
        Bununla birlikte model performanslarının veri setinin
        yapısına bağlı olarak değiştiği gözlemlenmiştir.
        </p>
        <p>
        Dolayısıyla sentetik veri üretim yöntemi seçiminin,
        veri setinin özellikleri ve uygulama alanı dikkate alınarak
        gerçekleştirilmesi önerilmektedir.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # GELECEK ÇALIŞMALAR
    # =====================================================
    st.header("")
    st.markdown("""
    <div style="
        background-color:#F5F3FF;
        padding:30px;
        border-radius:15px;
        border-left:8px solid #8B5CF6;">
        <h2>Gelecek Çalışmalar</h2>
        <ul style="font-size:18px;">
            <li>Daha fazla veri seti üzerinde kapsamlı deneylerin gerçekleştirilmesi</li>
            <br>
            <li>Hibrit sentetik veri üretim yaklaşımlarının geliştirilmesi</li>
            <br>
            <li>Gerçek zamanlı sentetik veri üretim sistemlerinin tasarlanması</li>
            <br>
            <li>Açıklanabilir yapay zeka yöntemleri ile model analizlerinin desteklenmesi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    # =====================================================
    # TEŞEKKÜR
    # =====================================================
    st.header("")
    st.markdown("""
    <div style="
        background-color:#F8FAFC;
        padding:30px;
        border-radius:15px;
        border-left:8px solid #16A34A;">
        <h2>Teşekkürler</h2>
        <p>
        Başta proje danışmanımız <b>Doç. Dr. Murtaza Cicioğlu</b>'na,
        çalışma süresince sağladığı değerli katkılar, yönlendirmeleri
        ve destekleri için teşekkür ederiz.
        </p>
        <p>
        Ayrıca bu çalışmanın gerçekleştirilmesi sürecinde bilgi ve
        deneyimlerinden faydalandığımız tüm hocalarımıza ve bizi
        dinleyerek değerli zamanlarını ayıran tüm katılımcılara
        teşekkürlerimizi sunarız.
        </p>
        <p style="text-align:center; font-size:20px;">
        <b>İlginiz ve katılımınız için teşekkür ederiz.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)