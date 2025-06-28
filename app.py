import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ======================= PAGE CONFIG =======================
st.set_page_config(page_title="Klasifikasi Transaksi", layout="centered")
st.title("🔍 Aplikasi Klasifikasi Pola Transaksi Pelanggan")

# ======================= PILIH MENU ========================
menu = st.selectbox("📌 Pilih Mode Penggunaan", ["Input Manual", "Upload File"])

# Fungsi untuk melatih model
def train_model(df):
    X = df.drop("Target", axis=1)
    y = df["Target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, scaler, X_test, y_test, X.columns.tolist(), y.unique().tolist()

# ======================= INPUT MANUAL =======================
if menu == "Input Manual":
    st.header("📝 Form Input Data Transaksi")

    # Load data untuk pelatihan model
    df = pd.read_csv("data_clustering.csv")
    model, scaler, _, _, feature_names, class_labels = train_model(df)

    int_cols = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation', 'LoginAttempts', 'AgeGroup']
    float_cols = ['TransactionAmount', 'CustomerAge', 'TransactionDuration', 'AccountBalance']

    with st.form("form_manual"):
        user_input = []
        for col in feature_names:
            if col in int_cols:
                val = st.number_input(f"{col}", value=0, step=1, format="%d")
            elif col in float_cols:
                val = st.number_input(f"{col}", value=0.123456, step=0.000001, format="%.6f")
            else:
                val = st.number_input(f"{col}", value=0.0)
            user_input.append(val)
        submit = st.form_submit_button("🔮 Prediksi")

    if submit:
        user_np = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(user_np)
        pred_class = model.predict(scaled_input)[0]
        probas = model.predict_proba(scaled_input)[0]

        st.success(f"📌 Prediksi: **Kategori {pred_class}**")
        st.write("📊 Probabilitas Tiap Kategori:")
        prob_df = pd.DataFrame({
            "Kategori": [str(c) for c in class_labels],
            "Probabilitas": [f"{p*100:.2f}%" for p in probas]
        })
        st.table(prob_df)

# ======================= UPLOAD FILE =======================
elif menu == "Upload File":
    st.header("📁 Upload Dataset CSV")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Data Awal")
        st.dataframe(df.head())

        if 'Target' not in df.columns:
            st.error("❌ Kolom 'Target' tidak ditemukan dalam dataset.")
        else:
            model, scaler, X_test, y_test, feature_names, class_labels = train_model(df)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"🎯 Akurasi Model: {acc*100:.2f}%")

            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")
            fig_cm, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            st.pyplot(fig_cm)

            # Plot garis
            st.subheader("📈 Plot Garis Aktual vs Prediksi")
            fig_line, ax = plt.subplots()
            ax.plot(range(len(y_test)), y_test.values, label="Aktual", color="green")
            ax.plot(range(len(y_pred)), y_pred, label="Prediksi", color="red", linestyle="--")
            ax.set_title("Perbandingan Data Aktual vs Prediksi")
            ax.legend()
            st.pyplot(fig_line)

            # Bar Chart
            st.subheader("📊 Plot Batang Kategori Prediksi")
            fig_bar, ax = plt.subplots()
            pd.Series(y_pred).value_counts().sort_index().plot(kind="bar", ax=ax, color="orange")
            ax.set_xlabel("Kategori")
            ax.set_ylabel("Jumlah")
            st.pyplot(fig_bar)

            # Donut Chart
            st.subheader("🍩 Donut Chart Kategori Aktual")
            kelas_aktual = pd.Series(y_test).value_counts().sort_index()
            fig_donut = px.pie(
                values=kelas_aktual.values,
                names=[str(i) for i in kelas_aktual.index],
                hole=0.5,
                title="Distribusi Kategori Aktual"
            )
            st.plotly_chart(fig_donut)

            # Tree
            st.subheader("🌳 Pohon Keputusan")
            fig_tree, ax = plt.subplots(figsize=(16, 6))
            plot_tree(model, feature_names=feature_names, class_names=[str(i) for i in class_labels],
                      filled=True, rounded=True, fontsize=8, ax=ax)
            st.pyplot(fig_tree)
