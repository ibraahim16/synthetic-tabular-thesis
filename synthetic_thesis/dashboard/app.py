import streamlit as st

from config import APP_TITLE
from components.sidebar import show_sidebar

from views import (
    home,
    detection_metric,
    efficacy_metric,
    statistical_analysis,
    visualization,
    demo,
    conclusions,
)

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)

selected_page = show_sidebar()

if selected_page == "Ana Sayfa":
    home.show()

elif selected_page == "Ayırt Edilebilirlik Analizi":
    detection_metric.show()

elif selected_page == "Kullanılabilirlik Analizi":
    efficacy_metric.show()

elif selected_page == "İstatistiksel Benzerlik Analizi":
    statistical_analysis.show()

elif selected_page == "Görsel Benzerlik Analizi":
    visualization.show()

elif selected_page == "Uygulama ve Tahmin Arayüzü":
    demo.show()

elif selected_page == "Genel Değerlendirme ve Sonuçlar":
    conclusions.show()
