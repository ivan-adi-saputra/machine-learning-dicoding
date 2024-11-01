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