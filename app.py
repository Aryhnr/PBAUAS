import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Unduh stopwords bahasa Indonesia
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi preprocessing teks
def preprocess_text(text):
    if not text:
        return ""
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('indonesian'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = StemmerFactory().create_stemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Fungsi pembobotan TF-IDF
def pembobotan(df, soal):
    # Ekstraksi baris yang sesuai untuk soal yang dipilih
    row = df[df['Soal'] == soal].iloc[0]

    # Menggabungkan kunci jawaban dan jawaban mahasiswa menjadi satu list
    texts = [row['Q']] + [row[f'M{j}'] for j in range(1, num_students + 1)]

    # Menghitung TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Mendapatkan nama fitur (kata)
    term = vectorizer.get_feature_names_out()

    # Membuat DataFrame dari matriks TF-IDF
    tfidf_df = pd.DataFrame(tfidf_matrix.T.toarray(), index=term, columns=['Q'] + [f'M{j}' for j in range(1, num_students + 1)])
    return tfidf_df

# Fungsi menghitung cosine similarity
def cosineSimilarity(result_tfidf, siswa):
    q_vector = result_tfidf['Q'].values.reshape(1, -1)
    similarities = []
    for j in range(1, siswa + 1):
        m_vector = result_tfidf[f'M{j}'].values.reshape(1, -1)
        similarity = cosine_similarity(q_vector, m_vector)[0][0]
        similarities.append(similarity)
    return similarities

# Fungsi mengonversi similarity menjadi persentase
def convertToPercentage(similarities):
    similarities_percent = [similarity * 100 for similarity in similarities]
    return similarities_percent
st.title("Penilaiian Esai Otomatis")
# Input jumlah soal dan jumlah mahasiswa
num_questions = st.number_input('Jumlah Soal', min_value=1, step=1)
num_students = st.number_input('Jumlah Mahasiswa', min_value=1, max_value=6, step=1)

# Inisialisasi dictionary untuk menyimpan kunci jawaban dan jawaban mahasiswa
data = {'Q': []}
question_numbers = []
for i in range(1, num_students + 1):
    data[f'M{i}'] = []

# Input kunci jawaban untuk setiap soal
for i in range(1, num_questions + 1):
    data['Q'].append(st.text_area(f'Masukkan Kunci Jawaban untuk Soal {i}', key=f'q_{i}'))
    question_numbers.append(i)

# Input jawaban mahasiswa untuk setiap soal
for i in range(1, num_questions + 1):
    st.subheader(f'Soal {i}')
    for j in range(1, num_students + 1):
        data[f'M{j}'].append(st.text_area(f'Jawaban Mahasiswa {j} untuk Soal {i}', key=f'm_{j}_{i}'))

# Input question weights dynamically
bobot_soal = {}
for i in range(1, num_questions + 1):
    bobot_soal[f'Soal{i}'] = st.slider(f'Bobot Soal {i}', min_value=1, max_value=100, value=50, step=1)

data_df = pd.DataFrame(data)
data_df.insert(0, 'Soal', question_numbers)
st.write("Data yang diinput:")
st.write(data_df)

# Preprocess data
for col in data_df.columns:
    if col != 'Soal':  # Skip kolom 'Soal'
        data_df[col] = data_df[col].apply(preprocess_text)



# Tombol untuk memulai analisis
if st.button('Mulai Analisis'):
    st.write("Data Hasil Preprocessing:")
    st.write(data_df)

    # Validasi apakah terdapat teks yang valid untuk analisis
    texts_valid = data_df.drop(columns='Soal').apply(lambda col: col.str.strip().astype(bool).any()).any()

    if texts_valid:
        total_scores = []

        for i in range(1, num_questions + 1):
            result_tfidf = pembobotan(data_df, i)
            similarities = cosineSimilarity(result_tfidf, num_students)
            similarities_percent = convertToPercentage(similarities)
            st.header(f'Soal {i}')
            # Hitung skor untuk setiap mahasiswa
            scores = []
            for idx, similarity in enumerate(similarities_percent, start=1):
                st.subheader(f'Mahasiswa {idx}')
                st.write(f'Nilai Cosine Similarity Mahasiswa {idx}: {similarity:.2f}%')
                score = similarity * bobot_soal[f'Soal{i}'] / 100  # Menggunakan bobot soal relatif
                scores.append(score)
                st.write(f'Nilai Konversi kedalam bobot Mahasiswa {idx} pada Soal {i}: {score:.2f}')

            total_scores.append(scores)

        # Menghitung total skor untuk setiap mahasiswa
        final_scores = []
        for j in range(num_students):
            total_score_j = sum(total_scores[i][j] for i in range(num_questions))
            final_scores.append(total_score_j)

        st.subheader("Total Skor Akhir untuk Setiap Mahasiswa:")
        for idx, score in enumerate(final_scores, start=1):
            st.write(f'Mahasiswa {idx}: {score:.2f}')

    else:
        st.warning('Masukkan setidaknya satu teks yang valid untuk melakukan analisis.')
