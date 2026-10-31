# Prediksi Curah Hujan Mingguan - LSTM + Flask

## Struktur
```
curah-hujan-lstm/
├── app.py                      # Flask app (inferensi NumPy, tanpa TensorFlow)
├── templates/index.html        # Dashboard (Chart.js)
├── notebooks/01_modeling.ipynb # EDA, preprocessing, training, evaluasi, simpan model
├── data/
│   ├── weather.csv             # dataset asli
│   └── weather_clean.csv       # hasil pembersihan (dibaca aplikasi)
├── model/
│   ├── lstm_model.h5           # model Keras
│   ├── scaler.joblib           # MinMaxScaler (joblib)
│   ├── lstm_weights.npz        # bobot untuk inferensi NumPy (dipakai app)
│   └── meta.json               # fitur, timesteps, parameter scaler, metrik
├── requirements.txt            # dependensi runtime (Vercel)
├── requirements-notebook.txt   # dependensi lokal untuk notebook
├── vercel.json / .vercelignore
```

## Jalankan lokal
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-notebook.txt
jupyter notebook notebooks/01_modeling.ipynb            # Run All -> regenerasi folder model/ dan data/weather_clean.csv
pip install -r requirements.txt
python app.py                                           # http://127.0.0.1:5000
```
