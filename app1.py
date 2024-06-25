import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Mengunduh stopwords dan tokenizer NLTK jika belum diunduh
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
    texts = [row['Kunci Jawaban']] + [row['Jawaban']]

    # Preprocessing teks
    texts = [preprocess_text(text) for text in texts]

    # Menghitung TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Mendapatkan nama fitur (kata)
    terms = vectorizer.get_feature_names_out()

    # Membuat DataFrame dari matriks TF-IDF
    tfidf_df = pd.DataFrame(tfidf_matrix.T.toarray(), index=terms, columns=['Kunci Jawaban'] + [f'Jawaban'])
    return tfidf_df

# Fungsi menghitung cosine similarity
def cosineSimilarity(result_tfidf):
    similarities = []
    q_vector = result_tfidf['Kunci Jawaban'].values.reshape(1, -1)
    m_vector = result_tfidf['Jawaban'].values.reshape(1, -1)
    similarity = cosine_similarity(q_vector, m_vector)[0][0]
    similarities.append(similarity)
    return similarities

# Fungsi mengonversi similarity menjadi persentase
def convertToPercentage(similarities):
    similarities_percent = [similarity * 100 for similarity in similarities]
    return similarities_percent

# Contoh data soal dan kunci jawaban
data = {
    'Soal': [
        "Menurut anda apa yang dimaksud dengan struktur data?",
        "Apa perbedaan antara tipe data integer dan float?",
        "Jelaskan secara ringkas karakteristik array yang anda ketahui!"
    ],
    'Kunci Jawaban': [
        "Struktur data adalah suatu cara pengelolaan data mulai dari penyimpanan, pengorganisasian dan penyimpanan data di dalam media penyimpanan komputer agar dapat digunakan secara efisien",
        "Tipe data integer digunakan untuk merepresentasikan suatu bilangan bulat positif maupun negatif sedangkan tipe data float digunakan untuk merepresentasikan bilangan pecahan positif maupun negatif",
        "Array memiliki karakteristik berupa kumpulan data dengan tipe data yang sama atau bersifat homogen, memiliki dimensi 1, 2 dan 3 atau multidimensi serta dapat diakses secara acak"
    ],
    'Bobot':[35,35,30]
}

df = pd.DataFrame(data)

# Judul aplikasi
st.title('Penilaian Ujian Esai Otomatis')

# Input jawaban siswa
for i, question in enumerate(df['Soal']):
    st.subheader(f'Soal {i+1}:')
    st.write(question)
    df.loc[i,'Jawaban'] = st.text_area(f'Jawaban Anda untuk soal {i+1}', key=f'answer_{i}')

# Tombol untuk menyelesaikan dan menilai
if st.button('Selesai'):
    total_scores = []
    total_weight = 0
    for i, question in enumerate(df['Soal']):
        tfidf_df = pembobotan(df, question,)
        similarities = cosineSimilarity(tfidf_df)
        scores = convertToPercentage(similarities)
        weighted_score = scores[0] * (df.loc[i, 'Bobot'] / 100)
        total_scores.append(weighted_score)
        total_weight += df.loc[i, 'Bobot']
        st.write(f"Nilai Anda untuk soal {i+1} (berbobot {df.loc[i, 'Bobot']}%): {scores[0]:.2f}%")
    
    # Menampilkan nilai akhir berbobot
    final_score = sum(total_scores)
    st.write(f'Nilai akhir berbobot Anda: {final_score:.2f}')
    

# Jalankan aplikasi Streamlit dengan menjalankan perintah berikut di terminal:
# streamlit run nama_file.py
