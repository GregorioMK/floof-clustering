import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import toml

# ===== HANYA SATU st.set_page_config() DI BARIS PERTAMA =====
st.set_page_config(
    page_title="Sistem Clustering Banjir", 
    page_icon="🌊",
    initial_sidebar_state="expanded",  # ← Ubah ke "expanded" agar sidebar selalu muncul
    layout="wide"
)

# ===== LOAD DATABASE CONFIGURATION =====
try:
    secrets = toml.load(".streamlit/secrets.toml")
    
    if "connection_string" in secrets["database"]:
        connection_string = secrets["database"]["connection_string"]
        from urllib.parse import urlparse
        parsed = urlparse(connection_string)
        db_host = parsed.hostname
        db_port = str(parsed.port) if parsed.port else "5432"
        db_name = parsed.path[1:]
        db_user = parsed.username
        db_password = parsed.password
    else:
        db_host = secrets["database"]["db_host"]
        db_port = secrets["database"]["db_port"]
        db_name = secrets["database"]["db_name"]
        db_user = secrets["database"]["db_user"]
        db_password = secrets["database"]["db_password"]
        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require"
        
except FileNotFoundError:
    st.error("❌ File secrets.toml tidak ditemukan")
    st.stop()
except KeyError as e:
    st.error(f"❌ Konfigurasi database tidak lengkap: {e}")
    st.stop()

# ===== INISIALISASI SESSION STATE =====
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.username = None

# ===== FUNGSI =====
def check_login(username, password):
    try:
        engine = create_engine(connection_string)
        query = "SELECT * FROM admin WHERE username = %s AND password = %s"
        result = pd.read_sql(query, engine, params=(username, password))
        engine.dispose()
        return len(result) > 0
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def show_footer():
    st.markdown("""
    <hr style='margin: 0.5rem 0;'>
    <div style='text-align: center; font-size: 0.9rem; padding: 0.5rem 0;'>
        © 2025 Gregorio Melvin Karnikov
    </div>
    """, unsafe_allow_html=True)

def logout():
    st.session_state.logged_in = False
    st.session_state.user_type = None
    st.session_state.username = None
    st.rerun()

# ===== HALAMAN LOGIN =====
if not st.session_state.logged_in:
    # Hide sidebar navigation saat belum login
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔐 Login System")
    st.write("Silakan pilih cara masuk ke aplikasi")
    
    login_option = st.radio(
        "Pilih opsi:",
        ["Login sebagai Admin", "Masuk sebagai Guest"],
        horizontal=True
    )
    
    if login_option == "Login sebagai Admin":
        st.subheader("🔑 Login Admin")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username and password:
                    if check_login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.user_type = "admin"
                        st.session_state.username = username
                        st.success("✅ Login berhasil!")
                        st.rerun()
                    else:
                        st.error("❌ Username atau password salah!")
                else:
                    st.warning("⚠️ Harap isi username dan password!")
    else:
        st.subheader("👤 Masuk sebagai Guest")
        st.info("Sebagai guest, Anda dapat melihat hasil clustering tetapi tidak dapat mengupload data.")
        
        if st.button("Masuk sebagai Guest", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.user_type = "guest"
            st.session_state.username = "Guest"
            st.success("✅ Berhasil masuk sebagai Guest!")
            st.rerun()

# ===== HALAMAN BERANDA (SETELAH LOGIN) =====
else:
    # Sidebar dengan logout
    with st.sidebar:
        st.write(f"👤 **User:** {st.session_state.username}")
        st.write(f"🔑 **Role:** {st.session_state.user_type.capitalize()}")
        st.divider()
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Konten Beranda
    st.title("🌊 Sistem Clustering Data Banjir")
    
    if st.session_state.user_type == "admin":
        st.success(f"Selamat datang, **{st.session_state.username}**! Anda login sebagai **Administrator**")
    else:
        st.info(f"Selamat datang, **{st.session_state.username}**! Anda login sebagai **Guest**")
    
    st.divider()
    
    # Gambar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("images/banjir.png", use_container_width=True, caption="Kejadian Banjir di DKI Jakarta")
        except:
            st.warning("⚠️ Gambar tidak ditemukan")
    
    st.divider()
    
    st.subheader("📖 Tentang Aplikasi")
    st.markdown("""
    ### Sistem Clustering Data Banjir dengan Algoritma K-Medoids
    
    Aplikasi ini merupakan sistem analisis data banjir yang menggunakan teknik *machine learning* 
    untuk mengelompokkan data banjir berdasarkan karakteristik tertentu.
    
    #### 🎯 Tujuan Aplikasi
    - Mengelompokkan data banjir berdasarkan kesamaan karakteristik
    - Mengidentifikasi pola dan tren kejadian banjir
    - Membantu dalam perencanaan mitigasi bencana banjir
    - Menyediakan visualisasi data yang mudah dipahami
    
    #### 🔧 Teknologi yang Digunakan
    - **Algoritma K-Medoids**: Metode clustering yang robust terhadap outlier
    - **MinMax Scaler**: Normalisasi data untuk meningkatkan akurasi clustering
    - **PCA**: Reduksi dimensi untuk visualisasi data
    - **Silhouette Score**: Evaluasi kualitas hasil clustering
    - **PostgreSQL**: Database untuk penyimpanan data
    - **Streamlit**: Framework untuk antarmuka aplikasi web
    
    #### 📊 Fitur Utama
    """)
    
    col_feature1, col_feature2 = st.columns(2)
    
    with col_feature1:
        st.markdown("""
        **Untuk Administrator:**
        - ✅ Upload dan import data banjir
        - ✅ Melakukan proses clustering
        - ✅ Normalisasi dan preprocessing data
        - ✅ Visualisasi hasil clustering 2D dan 3D
        - ✅ Evaluasi performa clustering
        - ✅ Export hasil clustering
        - ✅ Manajemen data dan pengguna
        """)
    
    with col_feature2:
        st.markdown("""
        **Untuk Guest:**
        - 👁️ Melihat hasil clustering
        - 👁️ Mengakses visualisasi data
        - 👁️ Membaca laporan analisis
        - 👁️ Download hasil clustering (terbatas)
        
        *Note: Guest tidak dapat mengubah data*
        """)
    
    show_footer()