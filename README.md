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

#### b. ratings.csv
- Jumlah data: 25.000.095 baris × 4 kolom
- Fitur:

| Fitur | Deskripsi |
| ------ | ------ |
| userId | ID unik untuk setiap pengguna. |
| movieId | ID film yang dirating |
| rating | Skor rating yang diberikan pengguna |
| timestamp | Waktu saat rating diberikan |

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

### 3. Content-Based Filtering

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

### 4. Collaborative Filtering

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

### 5. Splitting Data

Kode yang digunakan :
```python
x = cbf_df[['user', 'movie']].values
y = cbf_df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
```
Data dibagi menjadi set training (80%) dan validasi (20%) untuk melatih dan mengevaluasi model. Random state ditetapkan untuk memastikan reprodusibilitas hasil.

---

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

def rekomendasi_film_content(title, top_n=5):
    match = movie_features[movie_features['title'].str.lower() == title.lower()]
    if match.empty:
        print("Judul film tidak ditemukan.")
        return

    idx = match.index[0]
    sim_scores = list(enumerate(cos_sim[idx]))

    sim_scores = [x for x in sim_scores if x[0] != idx]

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]

    print(f"Rekomendasi untuk: {movie_features.iloc[idx]['title']} | Genre: {movie_features.iloc[idx]['genres']}")
    for i, (movie_idx, _) in enumerate(sim_scores, 1):
        rec_title = movie_features.iloc[movie_idx]['title']
        rec_genre = movie_features.iloc[movie_idx]['genres']
        print(f"{i}. {rec_title} | Genre: {rec_genre}")
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

```python
def rekomendasi_film_dari_user(user_id, model, cbf_df, user_to_user_encoded, user_encoded_to_user,
                              movie_to_movie_encoded, movie_encoded_to_movie, top_k=10):

    if user_id not in user_to_user_encoded:
        print(f"User {user_id} tidak ditemukan di data training.")
        return []

    user_encoded = user_to_user_encoded[user_id]

    movies_watched = cbf_df[cbf_df['userId'] == user_id]['movieId'].unique()
    movies_not_watched = [m for m in movie_to_movie_encoded.keys() if m not in movies_watched]
    movies_not_watched_encoded = [movie_to_movie_encoded[m] for m in movies_not_watched]

    user_array = np.array([user_encoded] * len(movies_not_watched_encoded))
    movie_array = np.array(movies_not_watched_encoded)
    input_array = np.vstack((user_array, movie_array)).T

    pred_ratings = model.predict(input_array, verbose=0).flatten()

    top_indices = pred_ratings.argsort()[-top_k:][::-1]
    top_movie_encoded = [movies_not_watched_encoded[i] for i in top_indices]
    top_movie_ids = [movie_encoded_to_movie[m] for m in top_movie_encoded]

    print(f"\nRekomendasi {top_k} film untuk user {user_id}:")

    for movie_id in top_movie_ids:
        title = cbf_df[cbf_df['movieId'] == movie_id]['title'].values
        if len(title) > 0:
            print(f"- {title[0]}")
        else:
            print(f"- Movie ID {movie_id} (judul tidak ditemukan)")

    return top_movie_ids
```

**Cara Kerja**:
- Validasi keberadaan user.
- Encode user.
- Ambil daftar film yang belum ditonton user.
- Siapkan input untuk model prediksi.
- Prediksi rating.
- Ambil top-N rekomendasi.
- Tampilkan hasil rekomendasi.

**Kelebihan**: 
- Tidak butuh informasi konten (fitur) dari item.
- Memberikan rekomendasi yang lebih bervariasi.
- Dapat menangkap selera kompleks pengguna

**Kekurangan**:
- Tidak cocok untuk pengguna baru tanpa histori (cold-start user).
- Memerlukan data rating dalam jumlah besar dan sumber daya komputasi lebih tinggi.
- Rentan terhadap manipulasi

**3. Proses Training**

Model collaborative filtering diimplementasikan menggunakan Neural Network dengan lapisan embedding pada kode berikut:

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.user_embedding = layers.Embedding(
            num_users, embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)

        self.movie_embedding = layers.Embedding(
            num_movies, embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])

        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)
```
Parameter yang digunakan :

| Parameter | Fungsi |
| ------ | ------ |
| **num_users** | Jumlah user unik, menentukan ukuran embedding user |
| **num_movies** | Jumlah film unik, menentukan ukuran embedding movie |
| **embedding_size** | Dimensi representasi laten user dan film |
| **he_normal** | 	Inisialisasi bobot embedding secara efisien |
| **l2(1e-6)** | Regularisasi untuk menghindari overfitting |
| **user_bias** | Menangkap kebiasaan rating setiap user |
| **movie_bias** | Menangkap kecenderungan rating untuk setiap film |

```python
model = RecommenderNet(num_users, num_movies, embedding_size=50)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

history = model.fit(
    x_train, y_train,
    batch_size=256,
    epochs=3,
    validation_data=(x_val, y_val)
)
```
cara kerja :
| Parameter | Fungsi |
| ------ | ------ |
| **Data input** | x_train: pasangan user–movie, y_train: rating (0–5) |
| **Model** | Belajar vektor embedding user dan film, serta bias |
| **Prediksi** | dot(user, movie) + bias, hasil diaktivasi sigmoid |
| **Loss** | Dihitung dengan **binary crossentropy** |
| **Optimisasi** | Bobot diperbarui dengan **Adam optimizer** |
| **Evaluasi** | Diukur dengan **RMSE** selama training dan validasi |

---

## 6. Evaluation

Evaluasi model dalam proyek ini dilakukan dengan beberapa metrik dan juga dikaitkan dengan business understanding yang telah ditetapkan sebelumnya.

### 1. Root Mean Squared Error (RMSE)

RMSE digunakan untuk mengevaluasi model collaborative filtering berbasis deep learning. RMSE mengukur akar kuadrat dari rata-rata selisih kuadrat antara rating prediksi dan rating sebenarnya.

Formula RMSE:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

Dimana:
- $y_i$ adalah rating sebenarnya
- $\hat{y}_i$ adalah rating prediksi
- $n$ adalah jumlah sampel

Hasil evaluasi menunjukkan:

- RMSE pada data latih: 0.2482
- RMSE pada data validasi: 0.2502

Nilai RMSE yang rendah (mendekati 0) menunjukkan performa model yang baik. Nilai RMSE yang mirip antara data latih dan validasi juga menunjukkan bahwa model tidak mengalami overfitting.

### 2. Content-Based Filtering

Disini saya merekomendasikan 2, yaitu genre dan film. Untuk film kita gunakan genre **drama**

Hasil dari Top 5 dari genre yang saya rekomendasikan adalah sebagai berikut :
<img width="593" alt="Image" src="https://github.com/user-attachments/assets/0f1df93a-ab47-4e32-b490-0bb7a485c068" />

**Penjelasan** : Sistem rekomendasi content-based memberikan 5 film teratas yang mirip dengan genre drama, yaitu:
- Blindness (2008)
- Conspiracy Theory (1997)
- Vertigo (1958)
- Rebecca (1940)
- Boxing Helena (1993)

Disini saya akan merekomendasikan film yang berjudul **Waiting to Exhale (1995)**

Hasil dari Top 5 dari film atau movie yang saya rekomendasikan adalah sebagai berikut :
<img width="428" alt="Image" src="https://github.com/user-attachments/assets/7b491728-97a9-423c-93e9-f570fefb476f" />

**Penjelasan** : Sistem rekomendasi content-based memberikan 5 film teratas yang mirip dengan **Waiting to Exhale (1995)**, yaitu:
- Terminal, The (2004)
- Graduate, The (1967)
- About Last Night... (1986)
- Singles (1992)
- Sleepless in Seattle (1993)

### 3. Collaborative Filtering

Disini saya akan merekomendasikan berdasarkan user id **33048**.

Hasil dari Top 10 dari film atau movie yang saya rekomendasikan adalah sebagai berikut :

<img width="451" alt="Image" src="https://github.com/user-attachments/assets/3d1a09f4-01a0-465d-894b-303c82aaa8de" />

**Penjelasan** : Sistem rekomendasi content-based memberikan 5 film teratas yang mirip dengan user id **33048**, yaitu:

- Schindler's List (1993)
- Silence of the Lambs, The (1991)
- Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)
- Princess Mononoke (Mononoke-hime) (1997)
- Godfather, The (1972)
- Pulp Fiction (1994)
- Wallace & Gromit: The Wrong Trousers (1993)
- 12 Angry Men (1957)
- Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
- Star Wars: Episode IV - A New Hope (1977)

### 4. Precision@K

Content-Based Filtering (CBF) Untuk mengevaluasi pendekatan Content-Based Filtering, digunakan metrik Precision@K, yaitu rasio item yang relevan terhadap jumlah total item yang direkomendasikan sebanyak K.

Definisi:

$$
Precision@K = \frac{(Jumlah item yang relevan)}{K}
$$

Relevansi ditentukan berdasarkan kemiripan genre atau tag antara film yang direkomendasikan dan film referensi yang disukai pengguna.

Film referensi:
🎬 Waiting to Exhale (1995)
Genre: Comedy | Drama | Romance
Top 5 Rekomendasi Film:

| No | Judul Film                                  | Genre                             | Relevan |
|----|---------------------------------------------|-----------------------------------|---------|
| 1  | Terminal, The (2004)                        | Comedy \| Drama \| Romance   | ✅      |
| 2  | Graduate, The (1967)                        | Comedy \| Drama \| Romance   | ✅      |
| 3  | About Last Night... (1986)                  | Comedy \| Drama \| Romance   | ✅      |
| 4  | Singles (1992)                              | Comedy \| Drama \| Romance   | ✅      |
| 5  | Sleepless in Seattle (1993)                 | Comedy \| Drama \| Romance   | ✅      |

Hasil: Precision@5 = 5 / 5 = 100%

- Precision@K memberikan gambaran seberapa relevan rekomendasi sistem terhadap preferensi pengguna.
- Evaluasi dilakukan secara kualitatif berdasarkan konten, sesuai dengan prinsip content-based filtering.
- Pendekatan ini berguna terutama ketika data eksplisit seperti rating pengguna belum tersedia atau terbatas (cold-start).

---
## Hubungan Model dengan Business Understanding

### 1. Apakah sudah menjawab setiap problem statement?

**Problem Statement 1:** _Bagaimana cara membuat sistem rekomendasi yang bisa menyarankan film berdasarkan kemiripan genre, atau fitur lainnya dari film yang pernah disukai pengguna (menggunakan pendekatan content-based filtering)?_  
**Terjawab Melalui:**

- Implementasi TF-IDF Vectorizer pada kolom genres.
- Penerapan cosine similarity.
- Fungsi rekomendasi_dengan_genre() dan rekomendasi_film_content().
- Evaluasi relevansi dengan Precision@5 = 100% menunjukkan keberhasilan metode ini.

**Problem Statement 2:** _bagaimana membangun sistem yang mampu mempelajari kebiasaan menonton pengguna, lalu memberikan rekomendasi yang sesuai dengan selera mereka, berdasarkan pola interaksi yang terekam, seperti film yang mereka tonton atau beri rating (dengan pendekatan collaborative filtering)?_  
**Terjawab Melalui:**  

- Implementasi Neural Collaborative Filtering dengan TensorFlow (class RecommenderNet).
- Penggunaan embedding layer untuk memetakan user dan movie.
- Pelatihan model dengan model.fit(...) menggunakan data rating pengguna.
- Fungsi rekomendasi_film_dari_user() yang menghasilkan rekomendasi berdasarkan prediksi dari interaksi sebelumnya.

**Problem Statement 3:** _Bagaimana mengukur keberhasilan sistem rekomendasi yang dikembangkan?_  
**Terjawab Melalui:**  

- Evaluasi Collaborative Filtering menggunakan RMSE (dengan hasil 0.2482 dan 0.2502 yang cukup rendah).
- Evaluasi Content-Based Filtering menggunakan Precision@K (dengan hasil Precision@5 = 100%).
- Penjelasan metrik RMSE dan Precision@K disertai formula dan interpretasi bisnisnya.

### 2. Apakah berhasil mencapai setiap goals yang diharapkan?

**Goal 1:** _Mengembangkan dua jenis sistem rekomendasi, yaitu content-based filtering dan collaborative filtering._  
**Tercapai.** mengimplementasikan dan membandingkan dua pendekatan.

**Goal 2:** _Memberikan rekomendasi film yang relevan dengan minat dan kesukaan masing-masing pengguna._  
**Tercapai.**
- Content-Based menghasilkan rekomendasi berdasarkan genre dan kemiripan konten.
- Collaborative Filtering mempelajari kebiasaan user dan merekomendasikan berdasarkan preferensi historis.
- Hasil rekomendasi spesifik diberikan untuk user_id=33048 dan film Waiting to Exhale (1995).

**Goal 3:** _Mengukur kinerja sistem rekomendasi menggunakan indikator evaluasi yang tepat dan terukur._  
**Tercapai.**  
- **Content-Based:** Precision@5 = 1  
- **Collaborative Filtering:** RMSE Data Latih : 0.2482, RMSE Data Validasi : 0.2502.

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

