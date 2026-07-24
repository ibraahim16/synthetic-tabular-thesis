# Üretken Ağlar ile Tabular Sentetik Veri Üretimi

Klasik yeniden örnekleme yöntemleri, GAN temelli üretken modeller ve difüzyon temelli modellerin tabular sentetik veri üretimindeki başarımını; dengeli (**Adult**) ve yüksek derecede dengesiz (**Credit Card Fraud**) iki veri kümesi üzerinde, çok boyutlu bir değerlendirme çerçevesiyle karşılaştıran bitirme projesi.

Sonuçların incelenmesi için **Streamlit tabanlı interaktif bir dashboard** ve gerçek/sentetik ayrımı yapan **dedektör modelleri** içerir.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Karşılaştırılan Modeller](#karşılaştırılan-modeller)
- [Veri Setleri](#veri-setleri)
- [Değerlendirme Metrikleri](#değerlendirme-metrikleri)
- [Öne Çıkan Bulgular](#öne-çıkan-bulgular)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Çalıştırma (Pipeline)](#çalıştırma-pipeline)
- [Dashboard](#dashboard)
- [Yapılandırma](#yapılandırma)
- [Çıktılar](#çıktılar)

## Genel Bakış

Tabular verilerin sürekli, kategorik ve ayrık değişkenleri bir arada barındıran karma yapısı, sentetik veri üretimini zorlaştırır. Bu çalışma üç yaklaşım ailesini aynı deneysel koşullar altında karşılaştırır:

1. **Klasik yeniden örnekleme** — SMOTE türevleri
2. **GAN temelli modeller** — CGAN, CTGAN, CopulaGAN, CTAB-GAN+
3. **Difüzyon temelli modeller** — TabSyn, ForestDiffusion

Üretilen veriler hem istatistiksel sadakat (fidelity) hem de aşağı akış makine öğrenmesi faydalılığı (utility) açısından değerlendirilir. Metrik sonuçları birleştirilerek **genel sıralama skoru** hesaplanır; tüm bulgular dashboard üzerinden görselleştirilir.

## Karşılaştırılan Modeller

`src/config.py` içindeki model listesi:

| Aile | Model anahtarı | Uygulama |
|------|----------------|----------|
| Yeniden örnekleme | `smote`, `borderline_smote`, `smote_tomek` | `imbalanced-learn` |
| GAN | `cgan` | PyTorch (özel) |
| GAN | `ctgan`, `copulagan` | `sdv` |
| GAN | `ctabgan_plus` | Vendored `CTAB-GAN-Plus/` |
| Difüzyon | `tabsyn` | Vendored `tabsyn/` (VAE + latent difüzyon) |
| Difüzyon | `forest_diffusion` | `ForestDiffusion` |

## Veri Setleri

| Veri seti | Sınıflandırma hedefi | Regresyon hedefi | Özellik |
|-----------|----------------------|------------------|---------|
| **Adult** (UCI) | `income` | `age` | Karma tipli, dengeliye yakın |
| **Credit Card Fraud** (OpenML) | `Class` | `Amount` | PCA dönüşümlü, ~%0,17 azınlık sınıf |

Veri setleri `prepare_data.py` tarafından otomatik indirilir; her birinden sabit tohumla **20.000** satırlık benchmark alt kümesi oluşturulur ve %80/%20 eğitim/test ayrımı yapılır.

## Değerlendirme Metrikleri

**İstatistiksel benzerlik** (`evaluate/statistical_tests.py`): KS testi, Wasserstein uzaklığı, Jensen-Shannon ıraksaması, Ki-Kare testi, korelasyon analizi, PCA ve t-SNE temelli manifold analizi.

**Ayırt edilebilirlik** (`evaluate/detection_metric.py`): Detection Metric (DM) — gerçek/sentetik ayrımı. Düşük skor, sentetik verinin gerçeğe daha yakın olduğunu gösterir.

**Faydalılık** (`evaluate/efficacy_metric.py`): Efficacy Metric (EM), TSTR yaklaşımı. Sınıflandırmada Accuracy/Precision/Recall/F1, regresyonda RMSE/R². Kullanılan modeller: Logistic/Linear Regression, Random Forest, XGBoost, CatBoost, AdaBoost ve AutoGluon (AutoML).

**Genel skor** (`evaluate/general_score.py`): İstatistiksel, DM ve EM metriklerini normalize ederek birleştirir. Ağırlıklar: istatistik %40, ayırt edilebilirlik %30, faydalılık %30.

## Öne Çıkan Bulgular

- **TabSyn**, üretken modeller arasında DM, EM, istatistiksel benzerlik ve korelasyon korunumunda en tutarlı ve en yüksek başarıyı gösterdi.
- **SMOTE temelli yöntemler**, KS/Wasserstein ve faydalılık görevlerinde öngörülenden güçlü, çoğu durumda TabSyn ile yarışan sonuçlar verdi.
- **ForestDiffusion** veri setine bağlı değişkenlik gösterdi (Credit'te güçlü, Adult DM'de zayıf).
- **CTGAN, CopulaGAN, CTAB-GAN+** orta düzeyde rekabetçi kaldı.
- **CGAN** tüm boyutlarda en düşük başarımı gösterdi (mode collapse).

> Hiçbir model tüm ölçütlerde mutlak üstünlük sağlamadı; yöntem seçimi veri yapısına ve uygulama önceliğine (sadakat mi, faydalılık mı) göre yapılmalıdır.

## Proje Yapısı

Depo kökü (`PythonProject/`) üç ana bileşenden oluşur: ana tez projesi, vendored CTAB-GAN+ ve vendored TabSyn.

```
PythonProject/
├── CTAB-GAN-Plus/                # Vendored CTAB-GAN+ implementasyonu
├── tabsyn/                       # Vendored TabSyn implementasyonu
└── synthetic_thesis/             # Ana proje
    ├── README.md
    ├── requirements.txt
    ├── train_detectors.py        # Dashboard demo arayüzü için dedektör eğitimi
    ├── data/
    │   ├── raw/                  # İndirilen ham veri
    │   ├── benchmark/            # adult_20k.csv, credit_20k.csv
    │   ├── split/                # *_train.csv, *_test.csv
    │   └── synthetic/            # adult/, credit/ → her model için .csv
    ├── models/
    │   ├── adult/                # Eğitilmiş üretken model ağırlıkları
    │   ├── credit/
    │   └── detectors/            # Gerçek/sentetik sınıflandırıcılar (.pkl)
    ├── outputs/
    │   ├── stats/                # İstatistiksel benzerlik çıktıları
    │   ├── dm/                   # Detection Metric sonuçları
    │   ├── em/                   # Efficacy Metric sonuçları
    │   ├── plots/                # stats/, dm/, em/ alt grafikleri
    │   ├── logs/
    │   ├── adult_general_ranking.csv
    │   └── credit_general_ranking.csv
    ├── dashboard/                # Streamlit değerlendirme arayüzü
    │   ├── app.py
    │   ├── config.py
    │   ├── components/
    │   ├── views/
    │   └── assets/
    └── src/
        ├── config.py             # Yollar, hedef sütunlar, hiperparametreler
        ├── prepare_data.py       # İndirme + benchmark + train/test ayrımı
        ├── data_quality_check.py
        ├── train/
        │   ├── train_statistical_models.py
        │   ├── train_cgan.py
        │   ├── train_ctgan.py
        │   ├── train_copulagan.py
        │   ├── train_ctabgan_plus.py
        │   ├── train_tabsyn.py
        │   └── train_forest_diffusion.py
        └── evaluate/
            ├── statistical_tests.py
            ├── detection_metric.py
            ├── efficacy_metric.py
            └── general_score.py
```

## Kurulum

Depo kökünden sanal ortam oluşturun:

```bash
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Ana proje bağımlılıkları
pip install -r synthetic_thesis/requirements.txt
```

Modellere ve dashboard'a göre ek bağımlılıklar gerekir:

```bash
pip install imbalanced-learn      # SMOTE, Borderline-SMOTE, SMOTE-Tomek
pip install sdv                   # CTGAN, CopulaGAN
pip install torch                 # CGAN, TabSyn
pip install catboost autogluon    # Detection / Efficacy metrikleri
pip install ForestDiffusion       # ForestDiffusion
pip install streamlit             # Dashboard

# Vendored implementasyonlar
pip install -r CTAB-GAN-Plus/requirements_ctabgan.txt
pip install -r tabsyn/requirements.txt
```

## Çalıştırma (Pipeline)

Betikler `synthetic_thesis/src` dizininden çalıştırılır ve her biri tüm veri setleri ile tüm modeller üzerinde otomatik döner (komut satırı argümanı gerektirmez).

```bash
cd synthetic_thesis/src

# 1) Veri indirme, benchmark örnekleme, train/test ayrımı
python prepare_data.py

# 2) Sentetik veri üretimi
python train/train_statistical_models.py
python train/train_cgan.py
python train/train_ctgan.py
python train/train_copulagan.py
python train/train_ctabgan_plus.py
python train/train_tabsyn.py
python train/train_forest_diffusion.py

# 3) (İsteğe bağlı) üretilen verilerin sağlık kontrolü
python data_quality_check.py

# 4) Değerlendirme
python evaluate/statistical_tests.py
python evaluate/detection_metric.py
python evaluate/efficacy_metric.py
python evaluate/general_score.py
```

Dashboard demo arayüzü için dedektör modelleri ayrıca eğitilir (`synthetic_thesis/` kökünden):

```bash
cd synthetic_thesis
python train_detectors.py
```

> Pipeline tamamlanmadan dashboard açılabilir; ancak metrik sayfaları ve demo arayüzü ilgili CSV/PKL çıktılarının mevcut olmasını bekler.

## Dashboard

Streamlit tabanlı arayüz, pipeline çıktılarını interaktif olarak sunar:

```bash
cd synthetic_thesis/dashboard
streamlit run app.py
```

| Sayfa | İçerik |
|-------|--------|
| Ana Sayfa | Proje amacı, yöntemler ve metrik özeti |
| Ayırt Edilebilirlik Analizi | DM sonuçları ve grafikler |
| Kullanılabilirlik Analizi | EM (sınıflandırma + regresyon) sonuçları |
| İstatistiksel Benzerlik Analizi | KS, Wasserstein, JS, Ki-Kare, korelasyon |
| Görsel Benzerlik Analizi | PCA / t-SNE karşılaştırmaları |
| Uygulama ve Tahmin Arayüzü | Kullanıcı girdisiyle gerçek/sentetik tahmin (dedektör modelleri) |
| Genel Değerlendirme ve Sonuçlar | Veri seti bazlı sıralama ve bulgular |

## Yapılandırma

Tüm temel ayarlar `src/config.py` içindedir:

| Ayar | Değer |
|------|-------|
| `RANDOM_STATE` | 42 |
| `BENCHMARK_ROWS` | 20.000 |
| `EPOCHS` | 500 |
| `BATCH_SIZE` | 512 |
| `TRAIN_RATIO` / `TEST_RATIO` | 0.80 / 0.20 |

Hedef sütunlar: Adult → `income` (sınıflandırma), `age` (regresyon); Credit → `Class` (sınıflandırma), `Amount` (regresyon).

Dashboard başlık ve sayfa listesi `dashboard/config.py` dosyasındadır.

## Çıktılar

| Konum | Açıklama |
|-------|----------|
| `data/synthetic/<dataset>/<model>.csv` | Üretilen sentetik veriler |
| `models/<dataset>/` | Eğitilmiş üretken model dosyaları |
| `models/detectors/` | Demo arayüzü için gerçek/sentetik sınıflandırıcılar |
| `outputs/stats/<dataset>/` | İstatistiksel benzerlik özet ve korelasyon/PCA çıktıları |
| `outputs/dm/<dataset>/` | Detection Metric sonuçları ve AutoGluon leaderboard'ları |
| `outputs/em/<dataset>/` | Efficacy Metric (sınıflandırma + regresyon) sonuçları |
| `outputs/plots/<dataset>/` | Değerlendirme grafikleri (`stats/`, `dm/`, `em/` alt dizinleri) |
| `outputs/<dataset>_general_ranking.csv` | Birleşik genel sıralama skoru |
