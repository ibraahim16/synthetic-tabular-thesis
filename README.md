# PythonProject — Tabular Sentetik Veri Üretimi

Üretken ağlar, GAN ve difüzyon modelleriyle tabular sentetik veri üretimini karşılaştıran bitirme projesi deposu.

## Bileşenler

| Dizin | Açıklama |
|-------|----------|
| [`synthetic_thesis/`](synthetic_thesis/) | Ana proje: veri hazırlama, model eğitimi, değerlendirme metrikleri ve Streamlit dashboard |
| [`CTAB-GAN-Plus/`](CTAB-GAN-Plus/) | CTAB-GAN+ vendored implementasyonu |
| [`tabsyn/`](tabsyn/) | TabSyn vendored implementasyonu |

## Hızlı Başlangıç

Kurulum, pipeline adımları, dashboard kullanımı ve çıktı yapısı için ana proje dokümantasyonuna bakın:

**[synthetic_thesis/README.md](synthetic_thesis/README.md)**

```bash
# Örnek: dashboard'u başlatma (pipeline çıktıları mevcut olduğunda)
cd synthetic_thesis/dashboard
streamlit run app.py
```
