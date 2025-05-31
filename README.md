# Sistem Rekomendasi - Movie Recommendation System

## 1. Project Overview

### Latar Belakang

Di era platform hiburan digital saat ini, sistem rekomendasi film memainkan peran krusial dalam membantu pengguna menemukan tontonan yang sesuai dengan selera mereka. Dengan jumlah film yang sangat banyak, pengguna sering merasa kewalahan saat mencari konten yang cocok. Jika sistem rekomendasi tidak bekerja dengan baik, hal ini bisa berdampak pada buruknya pengalaman pengguna, berkurangnya pengguna yang kembali menggunakan layanan, hingga menurunnya pendapatan perusahaan.

Penelitian menunjukkan bahwa kecepatan dan ketepatan sistem rekomendasi sangat memengaruhi pengalaman pengguna dalam memilih film. Menurut studi dari Nugroho et al. (2024), pengguna cenderung bosan dan berpindah ke platform lain jika tidak menemukan film yang mereka inginkan dalam waktu 90 detik. Untuk mengatasi hal ini, pendekatan seperti Neural Collaborative Filtering telah terbukti efektif dengan capaian recall hingga 69,6% dan NDCG sebesar 81,4% dalam eksperimen pada dataset MovieLens. Di sisi lain, Fanani (2024) dalam penelitiannya menggunakan metode K-Nearest Neighbors (KNN) berhasil mencapai precision hingga 100% dan recall sebesar 45,4%, menunjukkan bahwa algoritma berbasis kesamaan juga memiliki potensi yang kuat dalam memberikan rekomendasi yang relevan. Temuan-temuan ini mempertegas pentingnya pengembangan sistem rekomendasi yang akurat dan dipersonalisasi guna meningkatkan kepuasan pengguna sekaligus mendukung keberhasilan bisnis platform hiburan digital.

Melalui proyek ini, kami ingin membangun sistem rekomendasi film yang mampu membantu pengguna menemukan film sesuai dengan minat mereka. Dengan memanfaatkan data dari MovieLens — yang mencakup rating dan tag film dari ribuan pengguna — proyek ini akan menerapkan dan membandingkan dua metode rekomendasi, yaitu *Content-Based Filtering* dan *Collaborative Filtering* berbasis Deep Learning.

---

## 2. Business Understanding

### Problem Statements
1. Bagaimana cara membuat sistem rekomendasi yang bisa menyarankan film berdasarkan kemiripan genre, atau fitur lainnya dari film yang pernah disukai pengguna (menggunakan pendekatan content-based filtering)?
2. bagaimana membangun sistem yang mampu mempelajari kebiasaan menonton pengguna, lalu memberikan rekomendasi yang sesuai dengan selera mereka, berdasarkan pola interaksi yang terekam, seperti film yang mereka tonton atau beri rating (dengan pendekatan collaborative filtering)?
3. Bagaimana mengukur keberhasilan sistem rekomendasi yang telah dikembangkan?

### Goals
1. Mengembangkan dua jenis sistem rekomendasi, yaitu content-based filtering dan collaborative filtering.
2. Memberikan rekomendasi film yang relevan dengan minat dan kesukaan masing-masing pengguna.
3. Mengukur kinerja sistem rekomendasi menggunakan indikator evaluasi yang tepat dan terukur..

### Solution Statements

Untuk mewujudkan tujuan proyek, solusi yang akan diterapkan meliputi dua pendekatan utama: 
1. **Content-Based Filtering**
   - Menggunakan **TF-IDF Vectorizer** ntuk mengekstrak informasi penting dari genre film sebagai representasi fitur.
   - Menggunakan algoritma Nearest Neighbors dengan pendekatan cosine similarity.
   - Membuat daftar rekomendasi film berdasarkan kemiripan konten dengan film yang sebelumnya pernah disukai atau ditonton oleh pengguna.

2. **Collaborative Filtering**
   - Merancang model Neural Network yang menggunakan embedding layer untuk memetakan pengguna dan film.
   - Melatih model tersebut menggunakan data rating yang telah diberikan oleh pengguna.
   - Menghasilkan rekomendasi film dengan memilih film-film yang memiliki nilai prediksi tertinggi.

---

## 3. Data Understanding

### Dataset Overview
Proyek ini menggunakan dataset Movie Recommendation System yang tersedia di Kaggle, dikembangkan oleh Parashar Manas. Dataset ini dirancang untuk membangun sistem rekomendasi film menggunakan teknik pembelajaran mesin dan menyediakan dua file utama:

#### a. movies.csv
- Jumlah data: 62.432 baris × 3 kolom
- Fitur:

| Fitur | Deskripsi |
| ------ | ------ |
| MovieId | ID unik untuk setiap film |
| title | Judul film |
| genres | Genre film, dipisahkan dengan tanda | |

Tabel 1. Fitur dataset movie.csv

#### b. ratings.csv
- Jumlah data: 25.000.095 baris × 4 kolom
- Fitur:

| Fitur | Deskripsi |
| ------ | ------ |
| userId | ID unik untuk setiap pengguna. |
| movieId | ID film yang dirating |
| rating | Skor rating yang diberikan pengguna |
| timestamp | Waktu saat rating diberikan |

Tabel 2. Fitur dataset rating.csv

---

## 4. Data Preparation
Pada tahap ini, dilakukan beberapa teknik data preparation untuk mempersiapkan dataset sebelum digunakan dalam model sistem rekomendasi. Teknik-teknik tersebut dilakukan secara berurutan sebagai berikut:

### 1. Handling Missing Value
Tidak ditemukan missing values pada dataset.
  
| Fitur | 0 |
| ------ | ------ |
| **movieId** | 0 |
| **title** | 0 |
| **genres** | 0 |
| **userId** | 0 |
| **rating** | 0 |

### 2. Handling Duplicates
Kode yang digunakan 
```python
print(df.duplicated().sum())
```

Tidak terdapat data duplikat pada dataset tersebut.

### 3. Sample Dataset



### 4. Content-Based Filtering

Kode yang digunakan 
```python
movie_features = df_sample.drop_duplicates('movieId')[['movieId', 'title', 'genres']].reset_index(drop=True)
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_matrix = tfidf.fit_transform(movie_features['genres'])
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

**Penjelasan**:

- Mengambil data unik dari setiap film berdasarkan movieId, lalu menyimpan hanya kolom movieId, title, dan genres sebagai data utama untuk pembuatan fitur konten.
- Membuat objek TF-IDF Vectorizer untuk mengubah teks genre menjadi representasi numerik.
- Menerapkan TF-IDF Vectorizer pada kolom genres untuk menghasilkan matriks representasi fitur dari genre tiap film.
- Menghitung tingkat kemiripan antar film berdasarkan genre menggunakan metrik cosine similarity dari TF-IDF matrix.

### 5. Collaborative Filtering

Kode yang digunakan 
```python
user_ids = cbf_df['userId'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

movie_ids = cbf_df['movieId'].unique().tolist()
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}

cbf_df['user'] = cbf_df['userId'].map(user_to_user_encoded)
cbf_df['movie'] = cbf_df['movieId'].map(movie_to_movie_encoded)
```

**Penjelasan**: 

- Mengubah ID kualitatif (userId dan movieId) menjadi angka numerik (index).
- Membuat dictionary yang memetakan setiap userId asli (x) ke angka urut (i) — proses encoding.
- Melakukan validasi silang menggunakan cross_validate untuk mengukur performa model.

### 5. Collaborative Filtering

Kode yang digunakan :
```python
x = cbf_df[['user', 'movie']].values
y = cbf_df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
```
Data dibagi menjadi set training (80%) dan validasi (20%) untuk melatih dan mengevaluasi model. Random state ditetapkan untuk memastikan reprodusibilitas hasil.

## 5. Modeling and Result

### Pendekatan Sistem Rekomendasi
Dalam proyek ini, digunakan dua pendekatan utama untuk membangun sistem rekomendasi film, yaitu Content-Based Filtering dan Collaborative Filtering. Masing-masing pendekatan memiliki karakteristik dan metode kerja yang berbeda.

**1. Content-Based Filtering**

Model content-based filtering diimplementasikan menggunakan algoritma Nearest Neighbors dengan metrik cosine similarity:

```python
def rekomendasi_dengan_genre(genre_input, top_n=5):
    genre_mask = movie_features['genres'].str.contains(genre_input, case=False, na=False)
    matching_movies = movie_features[genre_mask]

    if matching_movies.empty:
        print(f"Tidak ada film pada genre '{genre_input}' ditemukan.")
        return

    idx = matching_movies.index[0]
    sim_scores = list(enumerate(cos_sim[idx]))

    sim_scores = [x for x in sim_scores if x[0] != idx]

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]

    print(f"Rekomendasi film mirip berdasarkan genre '{genre_input}' dengan referensi:")
    print(f"> {movie_features.iloc[idx]['title']} | Genre: {movie_features.iloc[idx]['genres']}")

    for i, (movie_idx, _) in enumerate(sim_scores, 1):
        title = movie_features.iloc[movie_idx]['title']
        genres = movie_features.iloc[movie_idx]['genres']
        print(f"{i}. {title} | Genre: {genres}")
```



**Cara Kerja**:
1. Filter film berdasarkan genre input.
2. Jika tidak ada film dengan genre tersebut, keluar.
3. Ambil indeks film referensi pertama dari genre tersebut.
4. Hitung skor kemiripan cosine.
5. Hilangkan film referensi dari hasil (jangan rekomendasikan dirinya sendiri).
6. Urutkan dan ambil Top-N film dengan skor kemiripan tertinggi
7. Tampilkan hasil rekomendasi

**Kelebihan**:
- Tidak membutuhkan data dari pengguna lain (independen).
- Bisa memberikan rekomendasi yang sangat personal.

**Kekurangan**:
- Keterbatasan eksplorasi item (sering terlalu mirip).
- Bergantung pada kualitas fitur item.
  
**2. Collaborative Filtering**

**Definisi**: Collaborative Filtering memberikan rekomendasi berdasarkan pola interaksi pengguna. Sistem ini mengasumsikan bahwa jika dua pengguna memiliki preferensi yang mirip di masa lalu, maka mereka cenderung menyukai item yang sama di masa depan.



**Cara Kerja**:
- Dibuat matriks user-item berdasarkan rating yang diberikan.
- Matriks ini kemudian didekomposisi menggunakan algoritma SVD (Singular Value Decomposition).
- SVD menemukan struktur laten (faktor tersembunyi) yang mewakili hubungan pengguna dan item, lalu digunakan untuk memprediksi rating yang belum diberikan.

**Kelebihan**: 
- Memanfaatkan pola preferensi pengguna secara kolektif.
- Mampu menemukan rekomendasi yang tidak terlihat hanya dari metadata item.

**Kekurangan**:
- Tidak cocok untuk pengguna baru tanpa histori (cold-start user).
- Memerlukan data rating dalam jumlah besar dan sumber daya komputasi lebih tinggi.

---

## 6. Evaluation

### Pendekatan 1: Content-Based Filtering

Disini saya merekomendasikan film Toy Story (1995)

Hasil dari Top 5 dari film atau movie yang saya rekomendasikan adalah sebagai berikut :
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/top5.png?raw=true)

**Penjelasan** : Sistem rekomendasi content-based memberikan 5 film teratas yang mirip dengan Toy Story (1995). Dari 5 film yang direkomendasikan, 3 film ditemukan dalam daftar film relevan (ground truth), yaitu:
- Pagemaster, The (1994)
- Kids of the Round Table (1995)
- Space Jam (1996)
- Jumanji (1995)
- Indian in the Cupboard, The (1995)

Namun hanya 3 film yang cocok dengan daftar relevant_movies, sehingga:
Precision@5 = 3 relevan / 5 rekomendasi = 0.60

Artinya, 60% rekomendasi yang diberikan sistem terbukti relevan berdasarkan daftar acuan, menunjukkan performa sistem yang cukup baik.

Teknik Evaluasi di atas adalah dengan menggunakan precission, rumus dari teknik ini adalah :
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/rumuscbs.png?raw=true)

### Pendekatan 2: Collaborative Filtering (SVD)

#### Metrik Evaluasi yang Digunakan
Evaluasi dilakukan secara **kuantitatif** dengan dua metrik:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

Keduanya digunakan untuk mengukur seberapa dekat prediksi model terhadap rating sebenarnya yang diberikan pengguna.

#### Penjelasan Metrik

- **RMSE**:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

- **MAE**:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

RMSE lebih sensitif terhadap error besar, sedangkan MAE memberikan estimasi rata-rata kesalahan model.

#### Hasil Evaluasi
Berdasarkan hasil validasi silang:
- **RMSE: 0.7812**
- **MAE: 0.5894**

Nilai ini menunjukkan bahwa model cukup akurat dalam memprediksi rating pengguna terhadap film.

#### Kesesuaian Metrik dengan Konteks
- Relevan digunakan dalam sistem rekomendasi berbasis rating.
- Menilai performa prediksi terhadap preferensi pengguna.
- Mendukung strategi personalisasi rekomendasi.

---
## Hubungan Model dengan Business Understanding

### 1. Apakah sudah menjawab setiap problem statement?

**Problem Statement 1:** _Bagaimana cara merekomendasikan film yang relevan bagi pengguna?_  
**Terjawab:**  
Model content-based dan collaborative filtering dibangun untuk menghasilkan rekomendasi film yang relevan berdasarkan kesamaan konten dan perilaku pengguna.

**Problem Statement 2:** _Bagaimana memanfaatkan data rating dan informasi konten film untuk menyusun sistem rekomendasi yang efektif?_  
**Terjawab:**  
- **Content-Based Filtering** menggunakan TF-IDF pada data genre film.  
- **Collaborative Filtering** menggunakan data rating dengan algoritma SVD.

### 2. Apakah berhasil mencapai setiap goals yang diharapkan?

**Goal 1:** _Membangun dua sistem rekomendasi: content-based dan collaborative._  
**Tercapai.** Kedua pendekatan berhasil diimplementasikan.

**Goal 2:** _Menyajikan rekomendasi film yang sesuai preferensi pengguna._  
**Tercapai.** Rekomendasi dihasilkan untuk film dan pengguna spesifik.

**Goal 3:** _Mengevaluasi performa sistem dengan metrik yang sesuai._  
**Tercapai.**  
- **Content-Based:** Precision@5 = 0.60  
- **Collaborative Filtering:** RMSE = 0.7812, MAE = 0.5894

### 3. Apakah setiap solusi statement yang kamu rencanakan berdampak?

Ya, kedua pendekatan memberikan dampak nyata terhadap tujuan bisnis:

#### Content-Based Filtering
- **Dampak:** Dapat memberikan rekomendasi untuk film baru (cold-start item).
- **Manfaat bisnis:** Menjaga ketertarikan pengguna baru melalui rekomendasi yang relevan.

#### Collaborative Filtering
- **Dampak:** Memberikan rekomendasi personal berdasarkan preferensi pengguna lain.
- **Manfaat bisnis:** Meningkatkan pengalaman pengguna jangka panjang dan loyalitas.

## Kesimpulan

Model yang dibangun sangat relevan terhadap _business understanding_ proyek ini:
- Semua **problem statement** dijawab.
- **Tujuan bisnis** tercapai dan divalidasi secara metrik.
- **Solusi yang diimplementasikan berdampak nyata** pada peningkatan kepuasan pengguna.

---

## Referensi
1. Nugroho, D. A., Lubis, C., & Perdana, N. J. (2024). SISTEM REKOMENDASI FILM MENGGUNAKAN METODE NEURAL COLLABORATIVE FILTERING. Journal of Information Technology and Computer Science (INTECOMS), 7(3), 926–937.
https://lintar.untar.ac.id/repository/penelitian/buktipenelitian_10393012_4A040824110850.pdf
2. Fanani, M. A. (2024). Sistem Rekomendasi Film Menggunakan Metode K-NN. Jurnal Ilmiah Sistem Informasi Dan Ilmu Komputer, 4(1), 178–185. https://doi.org/10.55606/juisik.v4i1.760

