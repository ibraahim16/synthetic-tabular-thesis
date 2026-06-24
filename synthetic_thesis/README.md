# Üretken Ağlar ile Tabular Sentetik Veri Üretimi

Klasik yeniden örnekleme yöntemleri, GAN temelli üretken modeller ve difüzyon temelli modellerin tabular sentetik veri üretimindeki başarımını; dengeli (**Adult**) ve yüksek derecede dengesiz (**Credit Card Fraud**) iki veri kümesi üzerinde, çok boyutlu bir değerlendirme çerçevesiyle karşılaştıran bitirme projesi.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Karşılaştırılan Modeller](#karşılaştırılan-modeller)
- [Veri Setleri](#veri-setleri)
- [Değerlendirme Metrikleri](#değerlendirme-metrikleri)
- [Öne Çıkan Bulgular](#öne-çıkan-bulgular)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Çalıştırma (Pipeline)](#çalıştırma-pipeline)
- [Yapılandırma](#yapılandırma)
- [Çıktılar](#çıktılar)

## Genel Bakış

Tabular verilerin sürekli, kategorik ve ayrık değişkenleri bir arada barındıran karma yapısı, sentetik veri üretimini zorlaştırır. Bu çalışma üç yaklaşım ailesini aynı deneysel koşullar altında karşılaştırır:

1. **Klasik yeniden örnekleme** — SMOTE türevleri
2. **GAN temelli modeller** — CGAN, CTGAN, CopulaGAN, CTAB-GAN+
3. **Difüzyon temelli modeller** — TabSyn, ForestDiffusion

Üretilen veriler hem istatistiksel sadakat (fidelity) hem de aşağı akış makine öğrenmesi faydalılığı (utility) açısından değerlendirilir.

## Karşılaştırılan Modeller

`config.py` içindeki model listesi:

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

**İstatistiksel benzerlik** (`statistical_tests.py`): KS testi, Wasserstein uzaklığı, Jensen-Shannon ıraksaması, Ki-Kare testi, korelasyon analizi, PCA temelli manifold analizi.

**Ayırt edilebilirlik** (`detection_metric.py`): Detection Metric (DM) — gerçek/sentetik ayrımı.

**Faydalılık** (`efficacy_metric.py`): Efficacy Metric (EM), TSTR yaklaşımı. Sınıflandırmada Accuracy/Precision/Recall/F1, regresyonda RMSE/R². Kullanılan modeller: Logistic/Linear Regression, Random Forest, XGBoost, CatBoost, AdaBoost ve AutoGluon (AutoML).

## Öne Çıkan Bulgular

- **TabSyn**, üretken modeller arasında DM, EM, istatistiksel benzerlik ve korelasyon korunumunda en tutarlı ve en yüksek başarıyı gösterdi.
- **SMOTE temelli yöntemler**, KS/Wasserstein ve faydalılık görevlerinde öngörülenden güçlü, çoğu durumda TabSyn ile yarışan sonuçlar verdi.
- **ForestDiffusion** veri setine bağlı değişkenlik gösterdi (Credit'te güçlü, Adult DM'de zayıf).
- **CTGAN, CopulaGAN, CTAB-GAN+** orta düzeyde rekabetçi kaldı.
- **CGAN** tüm boyutlarda en düşük başarımı gösterdi (mode collapse).

> Hiçbir model tüm ölçütlerde mutlak üstünlük sağlamadı; yöntem seçimi veri yapısına ve uygulama önceliğine (sadakat mi, faydalılık mı) göre yapılmalıdır.

## Proje Yapısı

```
synthetic-tabular-thesis-main/
├── CTAB-GAN-Plus/                # Vendored CTAB-GAN+ implementasyonu
├── tabsyn/                       # Vendored TabSyn implementasyonu
└── synthetic_thesis/            # Ana proje
    ├── requirements.txt
    ├── data/
    │   ├── raw/                  # İndirilen ham veri
    │   ├── benchmark/            # adult_20k.csv, credit_20k.csv
    │   ├── split/               # *_train.csv, *_test.csv
    │   └── synthetic/           # adult/, credit/ -> her model için .csv
    ├── outputs/
    │   ├── dm/                   # Detection Metric sonuçları
    │   ├── em/                   # Efficacy Metric sonuçları
    │   ├── stats/               # İstatistiksel benzerlik çıktıları
    │   ├── plots/  └─ logs/
    └── src/
        ├── config.py            # Yollar, hedef sütunlar, hiperparametreler
        ├── prepare_data.py      # İndirme + benchmark + train/test ayrımı
        ├── data_quality_check.py
        ├── train/
        │   ├── train_statistical_models.py   # SMOTE ailesi
        │   ├── train_cgan.py
        │   ├── train_ctgan.py
        │   ├── train_copulagan.py
        │   ├── train_ctabgan_plus.py
        │   ├── train_tabsyn.py
        │   └── train_forest_diffusion.py
        └── evaluate/
            ├── detection_metric.py
            ├── efficacy_metric.py
            └── statistical_tests.py
```

## Kurulum

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Ana proje bağımlılıkları
pip install -r synthetic_thesis/requirements.txt
```

Modellere göre ek bağımlılıklar gerekir:

```bash
pip install imbalanced-learn      # SMOTE, Borderline-SMOTE, SMOTE-Tomek
pip install sdv                   # CTGAN, CopulaGAN
pip install torch                 # CGAN
pip install catboost autogluon    # Detection / Efficacy metrikleri
pip install ForestDiffusion       # ForestDiffusion

# CTAB-GAN+ ve TabSyn vendored implementasyonları kendi gereksinim dosyalarını kullanır
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
python train/train_statistical_models.py     # SMOTE ailesi
python train/train_cgan.py
python train/train_ctgan.py
python train/train_copulagan.py
python train/train_ctabgan_plus.py
python train/train_tabsyn.py
python train/train_forest_diffusion.py

# 3) (İsteğe bağlı) üretilen verilerin sağlık kontrolü
python data_quality_check.py

# 4) Değerlendirme
python evaluate/statistical_tests.py          # KS, Wasserstein, JS, Chi², korelasyon, PCA
python evaluate/detection_metric.py           # DM
python evaluate/efficacy_metric.py            # EM (TSTR)
```

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

## Çıktılar

- `data/synthetic/<dataset>/<model>.csv` — üretilen sentetik veriler
- `outputs/stats/<dataset>/` — istatistiksel benzerlik özet ve korelasyon/PCA çıktıları
- `outputs/dm/<dataset>/` — Detection Metric sonuçları ve AutoGluon leaderboard'ları
- `outputs/em/<dataset>/` — Efficacy Metric (sınıflandırma + regresyon) sonuçları
