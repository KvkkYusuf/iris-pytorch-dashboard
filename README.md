# 🌸 Iris Lab — PyTorch & FastAPI İnteraktif Model Dashboard'u

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

**Iris Lab**, ünlü Kaggle Iris Species veri seti üzerinde eğitilmiş derin öğrenme (PyTorch MLP) modelini, modern bir **FastAPI** REST API servisi ve karanlık mod odaklı, harici kütüphane bağımlılığı olmayan reaktif bir **Web Dashboard'u** ile buluşturan uçtan uca makine öğrenimi projesidir.

---

## 📌 İçindekiler

- [Öne Çıkan Özellikler](#-öne-çıkan-özellikler)
- [Mimari ve Teknoloji Yığını](#-mimari-ve-teknoloji-yığını)
- [Proje Dizin Yapısı](#-proje-dizin-yapısı)
- [Sistem Gereksinimleri](#-sistem-gereksinimleri)
- [İndirme ve Kurulum Rehberi](#-i̇ndirme-ve-kurulum-rehberi)
  - [1. Yöntem: Tek Tıkla Otomatik Başlatma (Windows)](#1-yöntem-tek-tıkla-otomatik-başlatma-windows)
  - [2. Yöntem: Manuel Kurulum (Windows, macOS, Linux)](#2-yöntem-manuel-kurulum-windows-macos-linux)
- [Modeli Yeniden Eğitme (Retraining)](#-modeli-yeniden-eğitme-retraining)
- [API Dokümantasyonu ve Uç Noktalar](#-api-dokümantasyonu-ve-uç-noktalar)
  - [Tahmin İsteği Örneği (cURL)](#tahmin-isteği-örneği-curl)
  - [Swagger Arayüzü](#swagger-arayüzü)
- [Web Dashboard Kullanımı](#-web-dashboard-kullanımı)
- [Veri Seti Bilgisi](#-veri-seti-bilgisi)
- [Sorun Giderme (FAQ)](#-sorun-giderme-faq)
- [Lisans](#-lisans)

---

## 🚀 Öne Çıkan Özellikler

- **Uçtan Uca Derin Öğrenme Hattı:** Veri yükleme, ölçekleme (`StandardScaler`), PyTorch tensörlerine dönüştürme, katmanlı modelleme ve ağırlık serileştirme.
- **Yüksek Doğruluk Oranı:** Stratified %80-%20 ayrımıyla test kümesinde **%93.3+ doğruluk**.
- **Yüksek Performanslı REST API:** Asenkron `FastAPI` ve `uvicorn` altyapısı, Pydantic ile güçlü veri doğrulama ve CORS desteği.
- **Modern & Hafif Dashboard:** React/Vue gibi ağır framework'ler gerektirmeyen, saf HTML5/CSS/JavaScript ile kodlanmış, koyu temalı (dark mode) şık arayüz.
- **Gerçek Zamanlı İstatistikler & SVG Grafikleri:**
  - Canlı eğitim seyri grafiği (Epoch bazında kayıp ve doğruluk eğrileri).
  - Test kümesi karışıklık matrisi (Confusion Matrix).
  - Veri seti istatistik özeti ve örnek veri tablosu.
- **Canlı Tahmin & Güven Oranı:** Çiçek boyutları (sepal & petal uzunluk/genişlik) girildiğinde anında sınıf (Setosa, Versicolor, Virginica) ve olasılık dağılımı çıktısı.
- **Tek Tıkla Başlatma:** Windows için ortamı kuran, bağımlılıkları yükleyen, API'yi ayağa kaldıran ve tarayıcıyı açan `Baslat.bat` betiği.

---

## 🛠 Mimari ve Teknoloji Yığını

| Katman | Teknoloji / Araç | Açıklama |
| :--- | :--- | :--- |
| **Model / ML** | `PyTorch 2.x` | `4 → 16 → 12 → 3` yapısında çok katmanlı yapay sinir ağı (MLP), ReLU aktivasyonu, CrossEntropyLoss, Adam optimizasyonu. |
| **Veri & Ön İşleme** | `Pandas`, `NumPy`, `Scikit-Learn` | Veri temizleme, özellik ölçekleme (`StandardScaler`), Stratified Train/Test split. |
| **Backend / API** | `FastAPI`, `Uvicorn`, `Pydantic` | JSON tabanlı RESTful uç noktalar, girdi sınır doğrulamaları, OpenAPI/Swagger desteği. |
| **Frontend / Arayüz** | `HTML5`, `CSS3 (Custom Variables)`, `Vanilla JS` | Sıfır bağımlılık, responsive tasarım, dinamik SVG grafikleri. |

---

## 📂 Proje Dizin Yapısı

```text
iris-pytorch-dashboard/
│
├── artifacts/                  # Eğitilmiş model ağırlıkları ve metrikler
│   ├── iris_model.pt           # PyTorch model state_dict ve ölçekleme parametreleri
│   └── metrics.json            # Eğitim geçmişi, doğruluk, karmaşıklık matrisi ve veri özeti
│
├── data/                       # Veri seti dosyaları
│   ├── iris.zip                # Arşivlenmiş orijinal veri
│   └── raw/
│       ├── database.sqlite     # SQLite veri tabanı formatı
│       └── Iris.csv            # Ham CSV veri seti
│
├── api.py                      # FastAPI servis sunucusu ve tahmin uç noktaları
├── train.py                    # PyTorch modelini eğiten ve metrikleri kaydeden betik
├── index.html                  # İnteraktif karanlık mod web dashboard'u
├── requirements.txt            # Python kütüphane bağımlılıkları listesi
├── Baslat.bat                  # Windows için tek tıkla kurulum ve başlatma betiği
├── .gitignore                  # Git takip dışı dosyalar (.venv, __pycache__, vb.)
└── README.md                   # Proje ayrıntılı tanıtım ve dokümantasyon dosyası
```

---

## 💻 Sistem Gereksinimleri

- **İşletim Sistemi:** Windows 10/11, macOS veya Linux
- **Python:** Python 3.10 veya üzeri sürümler
- **Git:** Projeyi klonlamak için (veya doğrudan ZIP indirme)

---

## 📥 İndirme ve Kurulum Rehberi

### Projeyi Bilgisayarınıza İndirin

Terminal, PowerShell veya komut istemcisini açıp projeyi klonlayın:

```bash
git clone https://github.com/KvkkYusuf/iris-pytorch-dashboard.git
cd iris-pytorch-dashboard
```

*(Git kurulu değilse, GitHub sayfasındaki **Code -> Download ZIP** butonuna basarak indirip bir klasöre çıkartabilirsiniz).*

---

### 1. Yöntem: Tek Tıkla Otomatik Başlatma (Windows)

Windows kullanıyorsanız herhangi bir komut yazmanıza gerek yoktur:

1. Proje klasöründeki **`Baslat.bat`** dosyasına çift tıklayın.
2. Betik sırasıyla şunları yapacaktır:
   - Yerel bir `.venv` sanal Python ortamı oluşturur.
   - `requirements.txt` içerisindeki PyTorch, FastAPI ve diğer bağımlılıkları kurar.
   - FastAPI sunucusunu (`127.0.0.1:8000`) arka planda başlatır ve sağlık kontrolü yapar.
   - Hazır olduğunda varsayılan web tarayıcınızda `index.html` dashboard'unu otomatik olarak açar.
3. Dashboard üzerinde dilediğiniz değerleri girerek hemen tahmin alabilirsiniz!

> **Not:** API penceresini kapatmadığınız sürece sistem canlı olarak çalışmaya devam eder.

---

### 2. Yöntem: Manuel Kurulum (Windows, macOS, Linux)

Adımları kendiniz yönetmek isterseniz terminal üzerinden aşağıdaki komutları izleyin:

#### 1. Sanal Ortam Oluşturun ve Aktif Edin:

- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### 2. Bağımlılıkları Yükleyin:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. (İsteğe Bağlı) Modeli Eğitin:

Projede önceden eğitilmiş model ağırlıkları hazırdır. İsterseniz modeli sıfırdan tekrar eğitebilirsiniz:

```bash
python train.py
```

#### 4. API Sunucusunu Başlatın:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Sunucu başladığında konsolda şu çıktıyı göreceksiniz:
```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

#### 5. Dashboard'u Açın:

Proje kök dizinindeki `index.html` dosyasını favori tarayıcınızda açın (çift tıklayarak veya tarayıcıya sürükleyip bırakarak).

---

## 🧠 Modeli Yeniden Eğitme (Retraining)

Model mimarisinde veya eğitim parametrelerinde değişiklik yapmak isterseniz `train.py` dosyasını düzenleyebilirsiniz.

```python
# Model Mimarisi (train.py):
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 3)
        )
```

Eğitimi başlatmak için:
```bash
python train.py
```
Eğitim tamamlandığında:
- Güncel ağırlıklar ve ölçekleme parametreleri `artifacts/iris_model.pt` dosyasına yazılır.
- Epoch geçmişi, doğruluk ve karışıklık matrisi `artifacts/metrics.json` dosyasına işlenir.
- API veya dashboard yeniden başlatıldığında bu yeni modeli kullanmaya başlar.

---

## 📡 API Dokümantasyonu ve Uç Noktalar

FastAPI arka planda çalışırken aşağıdaki REST uç noktalarını sunar:

| Metot | Uç Nokta | Açıklama |
| :--- | :--- | :--- |
| `GET` | `/api/health` | API servisinin çalışma durumunu ve model bilgisini döner. |
| `GET` | `/api/metrics` | Eğitim geçmişi, doğruluk oranları, karışıklık matrisi ve veri özetini döner. |
| `GET` | `/api/sample` | Veri kümesinden rastgele seçilmiş 12 referans örneği listeler. |
| `POST` | `/api/predict` | Verilen 4 ölçüme göre çiçek türünü ve olasılıklarını hesaplar. |
| `GET` | `/docs` | Etkileşimli Swagger UI dokümantasyonu. |
| `GET` | `/redoc` | ReDoc alternatif API dokümantasyonu. |

### Tahmin İsteği Örneği (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

#### Örnek JSON Yanıtı:

```json
{
  "prediction": "Iris-setosa",
  "confidence": 0.99841,
  "probabilities": {
    "Iris-setosa": 0.99841,
    "Iris-versicolor": 0.00156,
    "Iris-virginica": 0.00003
  },
  "input": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

### Swagger Arayüzü

Tarayıcınızdan `http://127.0.0.1:8000/docs` adresine giderek API uç noktalarını tarayıcı üzerinden canlı olarak test edebilirsiniz.

---

## 🖥 Web Dashboard Kullanımı

`index.html` sayfası açıldığında:
1. **Durum İndikatörü:** Sağ üstte API'nin aktif olup olmadığını gösteren yeşil sinyal ışığı yer alır.
2. **Genel İstatistik Kartları:** Test kümesi başarımı (%93.3), toplam satır sayısı (150) ve eğitim tur adedi (180 epoch).
3. **Canlı Tahmin Formu:**
   - Sepal Length (Çanak yaprak uzunluğu) [3.0 - 9.0 cm]
   - Sepal Width (Çanak yaprak genişliği) [1.0 - 6.0 cm]
   - Petal Length (Taç yaprak uzunluğu) [0.0 - 8.0 cm]
   - Petal Width (Taç yaprak genişliği) [0.0 - 4.0 cm]
   Değerleri girip **"Türü tahmin et"** butonuna basarak olasılık çubuklarını görüntüleyin.
4. **Eğitim Seyri Grafiği:** Kayıp (loss) düşüşü ve doğrulama başarımı artışını gösteren dinamik SVG grafiği.
5. **Karışıklık Matrisi (Confusion Matrix):** Modelin hangi sınıfta kaç adet doğru ve hatalı tahmin yaptığını gösteren matris tablosu.
6. **Örnek Veri Seti Tablosu:** Test ve karşılaştırma yapabilmeniz için gerçek veri kümesinden örnek satırlar.

---

## 📊 Veri Seti Bilgisi

- **Kaynak:** [Kaggle Iris Species Dataset](https://www.kaggle.com/uciml/iris)
- **Orijinal Referans:** R.A. Fisher (1936)
- **Örnek Sayısı:** 150 (Her sınıftan 50 adet)
- **Sınıflar:**
  - `Iris-setosa` (Setosa)
  - `Iris-versicolor` (Versicolor)
  - `Iris-virginica` (Virginica)
- **Lisans:** CC0: Public Domain

---

## ❓ Sorun Giderme (FAQ)

<details>
<summary><b>1. "API çalışmıyor · Baslat.bat dosyasını çalıştır" uyarısı alıyorum</b></summary>
FastAPI sunucusu henüz başlatılmamış olabilir. <code>Baslat.bat</code> dosyasını çalıştırın veya terminalden <code>uvicorn api:app --port 8000</code> komutunu verin. Sayfayı yenilediğinizde sağ üstteki gösterge yeşile dönecektir.
</details>

<details>
<summary><b>2. Port 8000 başka bir uygulama tarafından kullanılıyor hatası</b></summary>
Farklı bir port kullanmak için terminalden:
<code>uvicorn api:app --port 8080</code>
komutunu verin. Ardından <code>index.html</code> sayfasının sağ üstündeki API adres kutucuğuna <code>http://127.0.0.1:8080</code> yazmanız yeterlidir.
</details>

<details>
<summary><b>3. PowerShell "Script Execution Policy" kısıtlaması veriyor</b></summary>
PowerShell'de venv aktif ederken hata alırsanız yönetici olarak PowerShell'i açıp:
<code>Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser</code>
komutunu çalıştırabilirsiniz veya doğrudan CMD / <code>Baslat.bat</code> kullanabilirsiniz.
</details>

---

## 📄 Lisans

Bu proje açık kaynaklı olup, eğitim ve geliştirme amaçlı özgürce kullanılabilir. Veri seti **CC0 1.0 Public Domain** kapsamındadır.
