import pandas as pd

# Data loading
# Digunakan untuk membaca isi file
test = pd.read_csv('./data/test.csv')
print("5 baris pertama dari dataset test:")
print(test.head())

train = pd.read_csv('./data/train.csv')
print("\n5 baris pertama dari dataset train:")
print(train.head())

# Data cleaning

# Menampilkan ringkasan informasi dari dataset
print("\nInformasi dari dataset train:")
train_info = train.info()
print(train_info)

# Menampilkan statistik deskriptif dari dataset
print("\nStatistik deskriptif dari dataset train:")
train_description = train.describe(include='all')
print(train_description)

# Memeriksa jumlah nilai yang hilang di setiap kolom
missing_value = train.isnull().sum()
print("\nJumlah nilai yang hilang di setiap kolom:")
print(missing_value[missing_value > 0])

# Mengatasi Missing Value
# Pertama-tama, mari kita pisahkan kolom yang memiliki missing value lebih dari 75% dan kurang dari 75%.
less = missing_value[missing_value < 1000].index
over = missing_value[missing_value >= 1000].index

# Contoh mengisi nilai yang hilang dengan median untuk kolom numerik
numeric_features = train[less].select_dtypes(include=['number']).columns
train[numeric_features] = train[numeric_features].fillna(train[numeric_features].median())

# Contoh mengisi nilai yang hilang dengan modus untuk kolom kategori
kategorical_features = train[less].select_dtypes(include=['object']).columns

for column in kategorical_features:
    train[column] = train[column].fillna(train[column].mode()[0])

# Menghapus kolom dengan terlalu banyak nilai yang hilang
df = train.drop(columns=over)

# Pemeriksaan terhadap data yang sudah melewati tahapan verifikasi missing value
missing_value = df.isnull().sum()
print("\nJumlah nilai yang hilang di setiap kolom setelah pembersihan:")
print(missing_value[missing_value > 0])

# Outliers
import seaborn as sns
import matplotlib.pyplot as plt
# Apakah dataset yang digunakan memiliki outlier atau tidak
for feature in numeric_features: 
    # Membuat figure baru
    # Baris ini membuat sebuah objek figure baru menggunakan Matplotlib dengan ukuran 10 inci (lebar) dan 6 inci (tinggi). Ukuran ini ditentukan dengan parameter figsize, yang membantu mengatur ukuran plot agar lebih sesuai untuk ditampilkan.
    plt.figure(figsize=(10, 6))

    # Membuat box plot
    # Baris ini menggunakan Seaborn (sns) untuk membuat box plot dari data di kolom yang ditunjuk oleh feature. df[feature] mengambil data dari kolom tersebut di DataFrame df. Box plot ini digunakan untuk menggambarkan distribusi data dan mendeteksi outlier.
    sns.boxplot(x=df[feature])

    # Menambahkan judul
    # Baris ini menetapkan judul pada plot saat ini. Judul berisi nama dari fitur yang sedang dianalisis, yang disisipkan dalam string menggunakan format f-string. Misalnya, jika feature adalah age, judul yang ditampilkan adalah "Box Plot of age".
    plt.title(f'Box Plot of {feature}')

    # Menampilkan plot
    # Baris ini memanggil fungsi plt.show() untuk menampilkan figure dan plot yang telah dibuat sebelumnya. Ini diperlukan agar visualisasi muncul di layar.
    plt.show()


# print(feature)

# Contoh sederhana untuk mengidentifikasi outliers menggunakan IQR
Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1

# print(IQR)

# Filter dataframe untuk hanya menyimpan baris yang tidak mengandung outliers pada kolom numerik
condition = ~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)
df_filtered_numeric = df.loc[condition, numeric_features]

# Menggabungkan kembali dengan kolom kategorikal
categorical_features = df.select_dtypes(include=['object']).columns
df = pd.concat([df_filtered_numeric, df.loc[condition, categorical_features]], axis=1)

# Normalisasi dan Standardisasi Data
from sklearn.preprocessing import StandardScaler

# Standardisasi fitur numerik
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Histogram Sebelum Standardisasi
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train[numeric_features[3]], kde=True)
plt.title("Histogram Sebelum Standardisasi")

# Histogram Setelah Standardisasi
plt.subplot(1, 2, 2)
sns.histplot(df[numeric_features[3]], kde=True)
plt.title("Histogram Setelah Standardisasi")

# Mengidentifikasi baris duplikat
duplicated = df.duplicated()
print("Baris duplikat:")
print(df[duplicated])

# Menghapus baris duplikat
df = df.drop_duplicates()
print("DataFrame setelah menghapus duplikat:")
print(df)

category_features = df.select_dtypes(include=['object']).columns
df[category_features]

# One Hot Encoding
df_one_hot = pd.get_dummies(df, columns=category_features)
df_one_hot

# Label Encoding
from sklearn.preprocessing import LabelEncoder

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()
df_lencoder = pd.DataFrame(df)

for col in categorical_features:
    df_lencoder[col] = label_encoder.fit_transform(df[col])

# Menampilkan hasil
df_lencoder

# Exploratory dan Explanatory Data Analysis

df_lencoder.head()

# Menghitung jumlah dan persentase missing values di setiap kolom
missing_values = df_lencoder.isnull().sum()
missing_percentage = (missing_values / len(df_lencoder)) * 100

missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
}).sort_values(by='Missing Values', ascending=False)

# Menampilkan kolom dengan missing values
missing_data[missing_data['Missing Values'] > 0] 

# Menghitung jumlah variabel
num_vars = df_lencoder.shape[1]

# Menentukan jumlah baris dan kolom untuk grid subplot
n_cols = 4 # Jumlah kolom yang diinginkan
n_rows = -(-num_vars // n_cols) # Ceiling division untuk jumlah baris

# Membuat subplot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))

# Flatten axes array untuk memudahkan iterasi jika diperlukan
axes = axes.flatten()

# Plot setiap variabel 
for i, column in enumerate(df_lencoder.columns):
    df_lencoder[column].hist(ax=axes[i], bins=20, edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

# Menghapus subplot yang tidak terpakai (jika ada)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Menyesuaikan layout agar lebih rapi
plt.tight_layout()
plt.show()

# Visualisasi distribusi data untuk beberapa kolom
columns_to_plot = ['OverallQual', 'YearBuilt', 'LotArea', 'SaleType', 'SaleCondition']

plt.figure(igsize=(15, 10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_lencoder[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()

# Visualisasi korelasi antar variabel numerik
plt.figure(figsize=(12, 10))
correlation_matrix = df_lencoder.corr()

sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Menghitung korelasi antara variabel target dan semua variabel lainnya
target_corr = df_lencoder.corr()['SalePrice']

# (Opsional) Mengurutkan hasil korelasi berdasarkan korelasi
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
target_corr_sorted.plot(kind='bar')
plt.title(f'Correlation with SalePrice')
plt.xlabel('Variables')
plt.ylabel('Correlation Coefficient')
plt.show()

# Data Splitting

# Memisahkan fitur (X) dan target (y)
X = df_lencoder.drop(columns=['SalePrice'])
y = df_lencoder['SalePrice']

from sklearn.model_selection import train_test_split

# membagi dataset menjadi training dan testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# menghitung panjang/jumlah data
print("Jumlah data: ", len(X))
# menghitung panjang/jumlah data pada x_train
print("Jumlah data latih: ", len(x_train))
# menghitung panjang/jumlah data pada x_test
print("Jumlah data test: ", len(x_test))

# Melatih Model (Training)

from sklearn import linear_model
# Melatih model 1 dengan algoritma Least Angle Regression
lars = linear_model.Lars(n_nonzero_coefs=1).fit(x_train, y_train)

# Melatih model 2 dengan algoritma Linear Regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression().fit(x_train, y_train)

# Melatih model 3 dengan algoritma Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(random_state=183)
GBR.fit(x_train, y_train)

# Evaluasi Model 

# Evaluasi pada model LARS
from sklearn.linear_model import Lars
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
pred_lars = lars.predict(x_test)
mae_lars = mean_absolute_error(y_test, pred_lars)
mse_lars = mean_squared_error(y_test, pred_lars)
r2_lars = r2_score(y_test, pred_lars)

# Membuat dictionary untuk menyimpan hasil evaluasi
data = {
    'MAE': [mae_lars],
    'MSE': [mse_lars],
    'R2': [r2_lars]
}

# Konversi dictionary menjadi DataFrame
df_results = pd.DataFrame(data, index=['Lars'])
df_results

# Evaluasi pada model Linear Regression
pred_LR = LR.predict(x_test)
mae_LR = mean_absolute_error(y_test, pred_LR)
mse_LR = mean_squared_error(y_test, pred_LR)
r2_LR = r2_score(y_test, pred_LR)

# Menambahkan hasil evaluasi LR ke DataFrame
df_results.loc['Linear Regression'] = [mae_LR, mse_LR, r2_LR]
df_results

# Evaluasi pada model Linear Regression
pred_GBR = GBR.predict(x_test)
mae_GBR = mean_absolute_error(y_test, pred_GBR)
mse_GBR = mean_squared_error(y_test, pred_GBR)
r2_GBR = r2_score(y_test, pred_GBR)
 
# Menambahkan hasil evaluasi LR ke DataFrame
df_results.loc['GradientBoostingRegressor'] = [mae_GBR, mse_GBR, r2_GBR]
df_results

# Menyimpan Model

import joblib
 
# Menyimpan model ke dalam file
joblib.dump(GBR, 'gbr_model.joblib')

import pickle
 
# Menyimpan model ke dalam file
with open('gbr_model.pkl', 'wb') as file:
    pickle.dump(GBR, file)

# Deployment dan Monitoring

# Memuat model dari file joblib
joblib_model = joblib.load('gbr_model.joblib')
 
# Memuat model dari file pickle
with open('gbr_model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)

from flask import Flask, request, jsonify
import joblib
 
# Inisialisasi aplikasi Flask
app = Flask(__name__)
 
# Memuat model yang telah disimpan
joblib_model = joblib.load('gbr_model.joblib') # Pastikan path file sesuai dengan penyimpanan Anda
 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Mengambil data dari request JSON
    prediction = joblib_model.predict(data)  # Melakukan prediksi (harus dalam bentuk 2D array)
    return jsonify({'prediction': prediction.tolist()})
 
if name == '__main__':
    app.run(debug=True)