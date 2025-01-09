
# import streamlit as st
# import joblib
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import pandas as pd


# try : 
#     from custom_rf import RandomForestRegressor
# except ImportError:
#     class RandomForestRegressor:
#         pass 

# try : 
#     from custom_rf import DecisionTree
# except ImportError:   
#     class DecisionTree:
#         pass

# # Load model
# @st.cache_resource
# def load_model():
#     try:
#         # Pastikan path file model sudah benar
#         model = joblib.load(open('model.joblib', 'rb'))  # File Random Forest Regressor
#         return model
#     except FileNotFoundError:
#         st.error("Model tidak ditemukan. Pastikan path file benar.")
#         return None

# model = load_model()

# # Fungsi untuk preprocessing data
# def preprocess_data(bedroom, bathroom, lokasi, area, tipe, unit):
#     # Encoding lokasi dan tipe properti
#     lokasi= {
#         'Alor Gajah, Melaka': 0, 'Barat Daya, Pulau Pinang': 1, 'Gombak, Selangor': 2, 'Hilir Perak, Perak': 3, 'Johor': 4,
#         'Johor Bahru, Johor': 5, 'Keningau, Sabah': 6, 'Kinta, Perak': 7, 'Klang, Selangor': 8, 'Kota Kinabalu, Sabah': 9,
#         'Kuala Kangsar, Perak': 10, 'Kuala Langat, Selangor': 11, 'Kuala Lumpur, Kuala Lumpur': 12, 'Kuala Muda, Kedah': 13,
#         'Kuala Selangor, Selangor': 14, 'Labuan, Labuan': 15, 'Muar, Johor': 16, 'Papar, Sabah': 17, 'Penampang, Sabah': 18,
#         'Petaling, Selangor': 19, 'Pontian, Johor': 20, 'Putrajaya, Putrajaya': 21, 'Seberang Perai Utara, Pulau Pinang': 22,
#         'Sepang, Selangor': 23, 'Seremban, Negeri Sembilan': 24, 'Ulu Langat, Selangor': 25, 'Ulu Selangor, Selangor': 26
#     }
#     tipe= {
#         'Apartment': 0, 'Commercial': 1, 'Condo': 2, 'House': 3, 'Office': 4,
#         'Serviced Apartment': 5, 'Townhouse': 6, 'Villa': 7, 'Hotel / Resort': 8
#     }

#     lokasi_encoded = lokasi.get(lokasi, -1)  # Default nilai jika lokasi tidak ditemukan
#     tipe_encoded = tipe.get(tipe, -1)

#     if lokasi_encoded == -1 or tipe_encoded == -1:
#         st.error("Terjadi kesalahan pada encoding data. Periksa input lokasi atau tipe properti.")
#         return None

#     # Gabungkan data menjadi array
#     data = np.array([[bedroom, bathroom, lokasi_encoded, area, tipe_encoded, unit]])
#     return data

# # Fungsi untuk melakukan prediksi
# def predict_price(data):
#     if data is None or model is None:
#         return None
#     prediction = model.predict(data)
#     return prediction[0]

# # Fungsi untuk menghitung metrik evaluasi
# def evaluate_model(test_data, test_labels):
#     if model is None:
#         st.error("Model tidak tersedia.")
#         return None, None, None

#     # Prediksi untuk data uji
#     predictions = model.predict(test_data)

#     # Hitung metrik evaluasi
#     mae = mean_absolute_error(test_labels, predictions)
#     mse = mean_squared_error(test_labels, predictions)
#     r2 = r2_score(test_labels, predictions)

#     return mae, mse, r2

# # Aplikasi Streamlit
# st.title('Prediksi Harga Properti di Malaysia - Random Forest Regressor')

# # Input pengguna
# bedroom = st.number_input('Masukkan jumlah kamar tidur:', min_value=0, max_value=40, step=1)
# bathroom = st.number_input('Masukkan jumlah kamar mandi:', min_value=0, max_value=40, step=1)
# lokasi = st.selectbox(
#     'Lokasi',
#     options=['Johor Bahru, Johor', 'Kinta, Perak', 'Kuala Kangsar, Perak', 'Johor',
#              'Seberang Perai Utara, Pulau Pinang', 'Kuala Lumpur, Kuala Lumpur',
#              'Sepang, Selangor', 'Putrajaya, Putrajaya', 'Petaling, Selangor',
#              'Kuala Langat, Selangor', 'Pontian, Johor', 'Muar, Johor',
#              'Ulu Langat, Selangor', 'Seremban, Negeri Sembilan', 'Klang, Selangor',
#              'Penampang, Sabah', 'Papar, Sabah', 'Keningau, Sabah', 'Kuala Muda, Kedah',
#              'Gombak, Selangor', 'Ulu Selangor, Selangor', 'Alor Gajah, Melaka',
#              'Kota Kinabalu, Sabah', 'Barat Daya, Pulau Pinang',
#              'Kuala Selangor, Selangor', 'Hilir Perak, Perak', 'Labuan, Labuan'])
# area = st.number_input('Masukkan luas area (m2):', min_value=0, max_value=90000, step=1)
# tipe = st.selectbox(
#     'Tipe Properti',
#     options=['House', 'Townhouse', 'Villa', 'Condo', 'Commercial', 'Apartment',
#              'Hotel / Resort', 'Serviced Apartment', 'Office'])
# unit = st.number_input('Masukkan harga per unit (Rp/m2):', min_value=0, max_value=200000000, step=1000)

# # Tombol hitung
# if st.button('Hitung'):
#     # Preprocess data
#     data = preprocess_data(bedroom, bathroom, lokasi, area, tipe, unit)

#     # Lakukan prediksi
#     predicted_price = predict_price(data)

#     # Tampilkan hasil
#     if predicted_price is not None:
#         st.success(f'Harga properti yang diprediksi: Rp {predicted_price:,.2f}')
#     else:
#         st.error("Prediksi gagal. Pastikan model tersedia atau input benar.")

# # Evaluasi model dengan data dari penyimpanan lokal
# st.subheader("Evaluasi Model")

# # Mengatur jalur file lokal
# load_data = 'dataset_cleaning (1).csv'  # Ganti dengan jalur file lokal Anda

# try:
#     # Membaca file CSV dengan pandas, melewatkan baris pertama (header)
#     test_data = pd.read_csv(load_data, skiprows=1, header=None)  # Menggunakan header=None untuk membaca tanpa header
    
#     # Mengasumsikan kolom terakhir adalah label harga dan kolom lainnya adalah fitur
#     # Sesuaikan indeks kolom sesuai dengan struktur dataset Anda
#     X_test = test_data.iloc[:, :-1].values  # Mengambil semua kolom kecuali kolom terakhir sebagai fitur
#     y_test = test_data.iloc[:, -1].values  # Mengambil kolom terakhir sebagai label

#     # Evaluasi model
#     mae, mse, r2 = evaluate_model(X_test, y_test)
#     st.write("### Metrik Evaluasi")
#     st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
#     st.write(f"Mean Squared Error (MSE): {mse:.2f}")
#     st.write(f"R-squared (R²): {r2:.2f}")

# except FileNotFoundError:
#     st.error("File tidak ditemukan. Pastikan jalur file sudah benar.")
# except Exception as e:
#     st.error(f"Terjadi kesalahan: {str(e)}")


import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cek dan impor model custom jika ada
try:
    from custom_rf import RandomForestRegressor
except ImportError:
    class RandomForestRegressor:
        pass

try:
    from custom_rf import DecisionTree
except ImportError:
    class DecisionTree:
        pass

# Load model
@st.cache_resource
def load_model():
    try:
        # Pastikan path file model sudah benar
        model = joblib.load('model.joblib')  # File Random Forest Regressor
        return model
    except FileNotFoundError:
        st.error("Model tidak ditemukan. Pastikan path file benar.")
        return None

model = load_model()

# Fungsi untuk preprocessing data
def preprocess_data(bedroom, bathroom, lokasi, area, tipe, unit):
    # Encoding lokasi dan tipe properti
    lokasi_mapping = {
        'Alor Gajah, Melaka': 0, 'Barat Daya, Pulau Pinang': 1, 'Gombak, Selangor': 2, 'Hilir Perak, Perak': 3, 'Johor': 4,
        'Johor Bahru, Johor': 5, 'Keningau, Sabah': 6, 'Kinta, Perak': 7, 'Klang, Selangor': 8, 'Kota Kinabalu, Sabah': 9,
        'Kuala Kangsar, Perak': 10, 'Kuala Langat, Selangor': 11, 'Kuala Lumpur, Kuala Lumpur': 12, 'Kuala Muda, Kedah': 13,
        'Kuala Selangor, Selangor': 14, 'Labuan, Labuan': 15, 'Muar, Johor': 16, 'Papar, Sabah': 17, 'Penampang, Sabah': 18,
        'Petaling, Selangor': 19, 'Pontian, Johor': 20, 'Putrajaya, Putrajaya': 21, 'Seberang Perai Utara, Pulau Pinang': 22,
        'Sepang, Selangor': 23, 'Seremban, Negeri Sembilan': 24, 'Ulu Langat, Selangor': 25, 'Ulu Selangor, Selangor': 26
    }
    tipe_mapping = {
        'Apartment': 0, 'Commercial': 1, 'Condo': 2, 'House': 3, 'Office': 4,
        'Serviced Apartment': 5, 'Townhouse': 6, 'Villa': 7, 'Hotel / Resort': 8
    }

    lokasi_encoded = lokasi_mapping.get(lokasi, -1)  # Default nilai jika lokasi tidak ditemukan
    tipe_encoded = tipe_mapping.get(tipe, -1)

    if lokasi_encoded == -1 or tipe_encoded == -1:
        st.error("Terjadi kesalahan pada encoding data. Periksa input lokasi atau tipe properti.")
        return None

    # Gabungkan data menjadi array
    data = np.array([[bedroom, bathroom, lokasi_encoded, area, tipe_encoded, unit]])
    return data

# Fungsi untuk melakukan prediksi
def predict_price(data):
    if data is None or model is None:
        return None
    prediction = model.predict(data)
    return prediction[0]

# Aplikasi Streamlit
st.title('Prediksi Harga Properti di Malaysia - Random Forest Regressor')

# Input pengguna
bedroom = st.number_input('Masukkan jumlah kamar tidur:', min_value=0, max_value=40, step=1)
bathroom = st.number_input('Masukkan jumlah kamar mandi:', min_value=0, max_value=40, step=1)
lokasi = st.selectbox(
    'Lokasi',
    options=['Johor Bahru, Johor', 'Kinta, Perak', 'Kuala Kangsar, Perak', 'Johor',
             'Seberang Perai Utara, Pulau Pinang', 'Kuala Lumpur, Kuala Lumpur',
             'Sepang, Selangor', 'Putrajaya, Putrajaya', 'Petaling, Selangor',
             'Kuala Langat, Selangor', 'Pontian, Johor', 'Muar, Johor',
             'Ulu Langat, Selangor', 'Seremban, Negeri Sembilan', 'Klang, Selangor',
             'Penampang, Sabah', 'Papar, Sabah', 'Keningau, Sabah', 'Kuala Muda, Kedah',
             'Gombak, Selangor', 'Ulu Selangor, Selangor', 'Alor Gajah, Melaka',
             'Kota Kinabalu, Sabah', 'Barat Daya, Pulau Pinang',
             'Kuala Selangor, Selangor', 'Hilir Perak, Perak', 'Labuan, Labuan'])
area = st.number_input('Masukkan luas area (m2):', min_value=0, max_value=90000, step=1)
tipe = st.selectbox(
    'Tipe Properti',
    options=['House', 'Townhouse', 'Villa', 'Condo', 'Commercial', 'Apartment',
             'Hotel / Resort', 'Serviced Apartment', 'Office'])
unit = st.number_input('Masukkan harga per unit (Rp/m2):', min_value=0, max_value=200000000, step=1000)

# Tombol hitung
if st.button('Hitung'):
    # Preprocess data
    data = preprocess_data(bedroom, bathroom, lokasi, area, tipe, unit)

    # Lakukan prediksi
    predicted_price = predict_price(data)

    # Tampilkan hasil
    if predicted_price is not None:
        st.success(f'Harga properti yang diprediksi: Rp {predicted_price:,.2f}')
    else:
        st.error("Prediksi gagal. Pastikan model tersedia atau input benar.")

# Mengatur jalur file lokal
load_data = 'dataset_cleaning (1).csv'  # Ganti dengan jalur file lokal Anda

try:
    # Membaca file CSV dengan pandas
    test_data = pd.read_csv(load_data, skiprows=1, header=None)

    # Konversi tipe data menjadi numerik
    test_data = test_data.apply(pd.to_numeric, errors='coerce')

    # Isi nilai yang hilang dengan rata-rata
    if test_data.isnull().values.any():
        test_data = test_data.fillna(test_data.mean())

    # Memisahkan fitur dan label
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

except FileNotFoundError:
    st.error("File tidak ditemukan. Pastikan jalur file sudah benar.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
