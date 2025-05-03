import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Identificador de Locutor", layout="centered")
st.title("🔊 Identificador de Locutor por Voz")

st.markdown("Faça upload de amostras de voz rotuladas para treino e depois envie áudios para identificar o locutor.")

# Upload dos arquivos de treino e teste
amostras_treinamento = st.file_uploader("Amostras para Treinamento (nome no formato rosiane_01.wav)", type=["wav", "mp3"], accept_multiple_files=True)
amostras_teste = st.file_uploader("Amostras para Identificação", type=["wav", "mp3"], accept_multiple_files=True)

@st.cache_data
def extrair_mfcc(uploaded_file, n_mfcc=13):
    y, sr = librosa.load(uploaded_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

if st.button("🔍 Identificar Locutores"):
    if not amostras_treinamento or not amostras_teste:
        st.warning("Envie amostras de treino e teste primeiro.")
    else:
        X, y = [], []

        for file in amostras_treinamento:
            nome = os.path.splitext(file.name)[0]
            label = nome.split("_")[0]
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            features = extrair_mfcc(tmp_path)
            X.append(features)
            y.append(label)

        X = np.array(X)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        modelo = SVC(kernel='linear', probability=True)
        modelo.fit(X_scaled, y_encoded)

        resultados = []

        for file in amostras_teste:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            features = extrair_mfcc(tmp_path)
            entrada = scaler.transform([features])
            pred = modelo.predict(entrada)[0]
            prob = modelo.predict_proba(entrada).max()
            resultado = f"{file.name} → {le.inverse_transform([pred])[0]} ({prob*100:.1f}%)"
            resultados.append(resultado)

        st.success("Resultado da Identificação:")
        for r in resultados:
            st.write(r)

        # Gráfico PCA
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X_scaled)
        fig, ax = plt.subplots()
        for i, label in enumerate(np.unique(y_encoded)):
            ax.scatter(X_vis[y_encoded == label, 0], X_vis[y_encoded == label, 1], label=le.inverse_transform([label])[0])
        ax.set_title("PCA das Características Vocais")
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.legend()
        st.pyplot(fig)

        # PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Relatório de Identificação de Locutores", ln=True, align='C')
        pdf.ln(10)
        for linha in resultados:
            pdf.cell(200, 10, txt=linha, ln=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            with open(tmp_pdf.name, "rb") as f:
                st.download_button("📄 Baixar Relatório em PDF", f, file_name="relatorio_identificacao.pdf")
