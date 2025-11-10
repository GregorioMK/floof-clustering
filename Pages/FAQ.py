import streamlit as st

st.set_page_config(
    page_title="FAQ",
    page_icon="🔍",
)

def show_footer():
    st.markdown("""
    <hr style='margin: 0.5rem 0;'>
    <div style='text-align: center; font-size: 0.9rem; padding: 0.5rem 0;'>
        © 2025 Gregorio Melvin Karnikov
    </div>
    """, unsafe_allow_html=True)

user_type = st.session_state.get("user_type")
if user_type == None:
    st.switch_page("BERANDA.py")

st.markdown("<h1 style='text-align: center;'>FREQUENTLY ASKED QUESTION</h1>", unsafe_allow_html=True)

with st.expander("Apa tujuan dari website ini?"):
    st.write("untuk mengelompokkan kerawanan banjir di Provinsi DKI Jakarta di tingkat kecamatan.")

with st.expander("Data apa saja yang digunakan dalam website ini?"):
    st.write("Data yang digunakan bersumber dari data kejadian bencana banjir di Provinsi DKI Jakarta yang disediakan oleh Badan Penanggulangan Bencana Daerah (BPBD) DKI Jakarta untuk klasterisasi sedangkan data populasi disediakan oleh Dinas Kependudukan dan Pencatatan Sipil Provinsi DKI Jakarta sebagai data tambahan untuk membantu pihak terkait dalam merencanakan kesiapsiagaan bencana secara lebih tepat. Melalui data tersebut, pihak berwenang dapat menentukan jumlah perahu karet, posko, dan tenaga evakuasi yang dibutuhkan di setiap wilayah. Selain itu, data ini juga berperan penting dalam penyusunan strategi evakuasi prioritas bagi kelompok rentan seperti lansia dan penyandang disabilitas, serta dalam mengoptimalkan distribusi bantuan dan fasilitas darurat di daerah yang memiliki tingkat kerawanan banjir lebih tinggi.")

with st.expander("Bagaimana cara website ini melakukan pengelompokkan (clustering)?"):
    st.write("""
    1. **Mengambil Data**: Mengambil data yang sudah disimpan di database
    2. **Preprocessing**: Data dinormalisasi menggunakan MinMax Scaler
    3. **Clustering**:  Mengelompokkan data berdasarkan kesamaan dengan menggunakan Algoritma K-Medoids dan DBSCAN
    4. **Evaluasi**: Sistem menghitung Silhouette Score untuk mengukur kualitas cluster
    5. **Visualisasi**: Hasil clustering divisualisasikan dalam berbagai visualisasi
    6. **Analisis**: Pengguna dapat menganalisis karakteristik setiap cluster
    """)

with st.expander("Apa manfaat dari clustering hasil banjir?"):
    st.write("banjir.")

with st.expander("Bagaimana cara menggunakan website ini?"):
    st.write("Anda dapat login sebagai guest kemudian masuk ke fitur clustering dan memilih parameter.")

with st.expander("Apakah bisa mengubah data yang digunakan dalam website?"):
    st.write("Ya, Bisa jika anda adalah admin dan memiliki akses sebagai administrator.")

with st.expander("Bagaimana cara menghubungi developer jika menemui masalah?"):
    st.write("Anda dapat menghubungi developer melalui alamat email : gregorio.535220085@stu.untar.ac.id.")

# Footer
show_footer()