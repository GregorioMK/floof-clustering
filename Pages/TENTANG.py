import streamlit as st

user_type = st.session_state.get("user_type")
if user_type == None:
    st.switch_page("BERANDA.py")

st.set_page_config(
    page_title="TENTANG",
    page_icon="📊",
)

def show_footer():
    st.markdown("""
        <hr style="margin-top: 50px;">
        <p style="text-align: center; color: #888; font-size: 14px;">
            © 2025 Gregorio Melvin Karnikov
        </p>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Tentang Website Ini</h1>", unsafe_allow_html=True)

# Deskripsi Website
st.markdown("""
    <div style='text-align: justify;'>
        <p>
            Website ini dibuat oleh <strong>Gregorio Melvin Karnikov</strong> untuk melakukan klasterisasi 
            wilayah rawan banjir di DKI Jakarta menggunakan algoritma <strong>K-Medoids</strong> dan 
            <strong>DBSCAN</strong>. Data bersumber dari BPBD DKI Jakarta dan dilengkapi dengan data 
            populasi dari Dukcapil guna mendukung perencanaan kesiapsiagaan bencana.
        </p>
        <p>
            Proyek ini merupakan bagian dari penelitian skripsi sebagai mahasiswa Teknik Informatika 
            Universitas Tarumanagara untuk memenuhi persyaratan kelulusan.
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()


st.markdown("<h2 style='text-align: center; margin-top: 30px;'>Dosen Pembimbing</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='margin-bottom: 10px;'>Pembimbing I</h3>
            <p style='font-size: 18px; font-weight: bold; margin: 5px 0;'>Prof. Dr. Ir. Dyah Erny Herwindiati, M.Si.</p>
            <p style='color: #666; margin: 5px 0;'>Universitas Tarumanagara</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3 style='margin-bottom: 10px;'>Pembimbing II</h3>
            <p style='font-size: 18px; font-weight: bold; margin: 5px 0;'>Novario Jaya Perdana, S.Kom., M.T.</p>
            <p style='color: #666; margin: 5px 0;'>Universitas Tarumanagara</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("<h2 style='text-align: center; margin-top: 30px;'>📚 Informasi Skripsi</h2>", unsafe_allow_html=True)

st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-top: 20px;'>
        <p style='margin: 10px 0;'><strong>Judul:</strong> Klasterisasi Wilayah Rawan Banjir di DKI Jakarta Menggunakan Algoritma K-Medoids dan DBSCAN</p>
        <p style='margin: 10px 0;'><strong>Tahun:</strong> 2025</p>
        <p style='margin: 10px 0;'><strong>Sumber Data:</strong> BPBD DKI Jakarta & Dinas Kependudukan dan Pencatatan Sipil DKI Jakarta</p>
        <p style='margin: 10px 0;'><strong>Teknologi:</strong> Python, Streamlit, PostgreSQL, Scikit-learn, Folium</p>
    </div>
""", unsafe_allow_html=True)

show_footer()