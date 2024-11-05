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



