# Predictive Maintenance — RUL Estimation & Fault Classification

A machine learning pipeline for **bearing health monitoring** using the **NASA IMS Bearing dataset**. Extracts time-frequency features via FFT and Wavelet transforms, then trains models for **Remaining Useful Life (RUL) regression** and **3-class health classification**.

---

## Dataset

**NASA IMS Bearing Dataset** — University of Cincinnati
- Test 1: 2156 files, 8 channels (4 bearings × 2 sensors)
- Test 2: 984 files, 4 channels (4 bearings × 1 sensor)
- Sampling frequency: 20,000 Hz
- Each file: 20,480 samples (~1 second of vibration data)

> Dataset not included due to size (1 GB+). Download from [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

---

## Features Extracted (33 per file)

**FFT Features (15)**
| Feature | Description |
|---|---|
| rms | Root Mean Square amplitude |
| kurtosis | Impulsiveness indicator |
| skewness | Signal asymmetry |
| crest_factor | Peak to RMS ratio |
| peak2peak | Signal range |
| fft_mean / std / max | Spectral statistics |
| fft_peak_freq | Dominant frequency |
| fft_energy | Total spectral energy |
| fft_entropy | Spectral complexity |
| fft_band1–5_energy | Energy in 5 frequency bands |

**Wavelet Features (18) — DWT db4, Level 5**
| Band | Frequency Range | Significance |
|---|---|---|
| cA5 | 0–312 Hz | Normal rotation |
| cD5 | 312–625 Hz | Low-mid fault |
| cD4 | 625–1250 Hz | Mid fault |
| cD3 | 1250–2500 Hz | Mid-high fault |
| cD2 | 2500–5000 Hz | High fault |
| cD1 | 5000–10000 Hz | Early fault signatures |

---

## Labels

**RUL (Regression)**
- 1.0 = first file (healthy)
- 0.0 = last file (failure)

**Health Stage (Classification)**
| Label | Stage | Files |
|---|---|---|
| 0 | Normal | First 60% |
| 1 | Suspect | Next 25% |
| 2 | Failure | Last 15% |

---

## Pipeline

```
Raw Vibration Files (.txt)
        |
        v
FFT + Wavelet Feature Extraction
        |
        v
Feature CSV (33 features × N files)
        |
        v
RUL Regression  +  Health Classification
        |
        v
Model Comparison (RF / GB / XGBoost)
Hyperparameter Tuning (RandomizedSearchCV / BayesSearchCV)
```

---

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn pywavelets tqdm
```

Open `PredictiveML.ipynb` in **Google Colab** or **Jupyter Notebook**.

Update the dataset path in the config section:
```python
DATA_DIR = "/path/to/your/Test_2"
```

---

## Authors

### Harie Goutaym D A  
B.Tech ELC (IoT) — Amrita Vishwa Vidyapeetham  
[GitHub](https://github.com/HarieGoutaym) | [LinkedIn](https://linkedin.com/in/harie-goutaym-d-a-67722a36a)

### Sarvesh V  
B.Tech Electrical and Computer Engineering — Amrita Vishwa Vidyapeetham  
BS in Data Science and Applications — IIT Madras  

[GitHub](https://github.com/Sarvesh4700) | [LinkedIn](www.linkedin.com/in/sarvesh43142929b)
