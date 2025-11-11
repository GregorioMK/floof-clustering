import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
import psycopg2
from sqlalchemy import create_engine
import toml

# Hide sidebar if guest
user_type = st.session_state.get("user_type")
if user_type == "guest":
    st.warning("👤 Anda login sebagai Guest. Halaman ini hanya dapat diakses oleh Admin.")
    st.stop()

if user_type is None:
    st.switch_page("BERANDA.py")
    st.rerun()

st.set_page_config(
    page_title="DATA",
    page_icon="📤",
)

def show_footer():
    st.markdown("""
    <hr style='margin: 0.5rem 0;'>
    <div style='text-align: center; font-size: 0.9rem; padding: 0.5rem 0;'>
        © 2025 Gregorio Melvin Karnikov
    </div>
    """, unsafe_allow_html=True)

st.title("📤 Update Data Banjir")

# Sidebar untuk database configuration dan upload
with st._main:
    
    # ===== LOAD DATABASE CONFIGURATION FROM secrets.toml =====
    try:
        secrets = toml.load(".streamlit/secrets.toml")
        db_host = secrets["database"]["db_host"]
        db_port = secrets["database"]["db_port"]
        db_name = secrets["database"]["db_name"]
        db_user = secrets["database"]["db_user"]
        db_password = secrets["database"]["db_password"]
    except FileNotFoundError:
        st.error("❌ File secrets.toml tidak ditemukan di folder .streamlit/")
        st.info("💡 Pastikan file secrets.toml ada di .streamlit/secrets.toml")
        st.stop()
    except KeyError as e:
        st.error(f"❌ Konfigurasi database tidak lengkap di secrets.toml: {e}")
        st.stop()
    # =========================================================
    
    # Upload data section
    st.header("Upload & Update Data")
    st.caption("Upload file untuk update data kecamatan di database")

    # ===== INFORMASI PENTING & DOWNLOAD TEMPLATE =====
    st.info("""
    ### 📋 Informasi Penting
    - ✅ File harus berisi **tepat 44 baris data kecamatan**
    - ✅ Nama kecamatan harus **valid dan sesuai** dengan database
    - ⚠️ **Hati-hati!** Data akan langsung **diupdate ke database**
    - 📥 Download template di bawah sebagai contoh format yang benar
    """)
    
    # Download Template Button

    template_path = "Dataset dengan demografi/2018.xlsx"
    with open(template_path, "rb") as file:
        template_data = file.read()
        
    st.download_button(
        label="📥 Download Template Excel (Contoh Format)",
        data=template_data,
        file_name="template_update_banjir.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download file contoh dengan format yang benar"
    )

    
    st.divider()
    
    upload_tahun = st.selectbox(
        "Tahun Target",
        options=list(range(2018, 2026)),
        key="upload_tahun"
    )
    
    uploaded_file = st.file_uploader(
        "Pilih File Excel/CSV",
        type=["xlsx", "xls", "csv"],
        help="File harus memiliki 44 baris kecamatan dengan header yang sama"
    )
    
    if uploaded_file is not None:
        try:
            # Baca file yang diupload
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            # Replace NaN dengan None
            df_upload = df_upload.where(pd.notna(df_upload), None)
            
            st.success(f"✅ File berhasil dibaca: {len(df_upload)} baris")
            
            # Validasi jumlah baris
            if len(df_upload) != 44:
                st.error(f"❌ Jumlah baris tidak sesuai! Ditemukan: {len(df_upload)}, Expected: 44")
            else:
                # Validasi kolom yang diperlukan
                required_cols = ["kecamatan", "jumlah_rw_terdampak", "jumlah_kk_terdampak", 
                                 "jumlah_jiwa_terdampak", "rata_ketinggian_air", "ketinggian_air_max",
                                 "jumlah_jiwa", "jumlah_disabilitas", "jumlah_lansia"]
                
                missing_cols = [col for col in required_cols if col not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"❌ Kolom tidak lengkap!")
                    st.write("Kolom yang hilang:", missing_cols)
                    st.write("Kolom yang ada:", df_upload.columns.tolist())
                else:
                    # === VALIDASI NAMA KECAMATAN ===
                    conn = None
                    engine = None
                    try:
                        # Koneksi ke database untuk ambil daftar kecamatan valid
                        if db_password:
                            connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                        else:
                            connection_string = f"postgresql://{db_user}@{db_host}:{db_port}/{db_name}"
                        
                        engine = create_engine(connection_string)
                        table_name = f"kejadian_{upload_tahun}"
                        
                        # Ambil daftar kecamatan yang ada di database
                        query_kecamatan = f"SELECT DISTINCT UPPER(TRIM(kecamatan)) as kecamatan FROM {table_name} ORDER BY kecamatan"
                        df_valid_kecamatan = pd.read_sql(query_kecamatan, engine)
                        valid_kecamatan_list = df_valid_kecamatan['kecamatan'].tolist()
                        
                        # Normalisasi nama kecamatan dari upload
                        df_upload['kecamatan_normalized'] = df_upload['kecamatan'].str.strip().str.upper()
                        upload_kecamatan_list = df_upload['kecamatan_normalized'].tolist()
                        
                        # Cek kecamatan yang tidak valid
                        invalid_kecamatan = [k for k in upload_kecamatan_list if k not in valid_kecamatan_list]
                        missing_kecamatan = [k for k in valid_kecamatan_list if k not in upload_kecamatan_list]
                        
                        # Tampilkan hasil validasi
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("📋 Total Kecamatan di Upload", len(upload_kecamatan_list))
                            st.metric("✅ Kecamatan Valid", len([k for k in upload_kecamatan_list if k in valid_kecamatan_list]))
                        
                        with col2:
                            st.metric("🗂️ Total Kecamatan di Database", len(valid_kecamatan_list))
                            st.metric("❌ Kecamatan Invalid", len(invalid_kecamatan))
                        
                        # Warning untuk kecamatan invalid
                        if invalid_kecamatan:
                            st.error(f"❌ Ditemukan {len(invalid_kecamatan)} nama kecamatan yang TIDAK ADA di database!")
                            with st.expander("🔍 Lihat Kecamatan Invalid"):
                                st.write(invalid_kecamatan)
                                st.info("💡 Kecamatan ini tidak akan di-update karena tidak ditemukan di database")
                        
                        # Warning untuk kecamatan yang hilang dari upload
                        if missing_kecamatan:
                            st.warning(f"⚠️ Ditemukan {len(missing_kecamatan)} kecamatan di database yang TIDAK ADA di file upload!")
                            with st.expander("🔍 Lihat Kecamatan yang Hilang"):
                                st.write(missing_kecamatan)
                                st.info("💡 Data kecamatan ini tidak akan diubah (tetap seperti semula)")
                        
                        # Jika semua valid, tampilkan success
                        if not invalid_kecamatan and not missing_kecamatan:
                            st.success("✅ Semua nama kecamatan valid dan lengkap!")
                        
                        # Preview data dengan status validasi
                        with st.expander("👀 Preview Data Upload (dengan Status Validasi)"):
                            df_preview = df_upload.copy()
                            df_preview['Status'] = df_preview['kecamatan_normalized'].apply(
                                lambda x: '✅ Valid' if x in valid_kecamatan_list else '❌ Invalid'
                            )
                            st.dataframe(df_preview[['kecamatan', 'Status', 'jumlah_rw_terdampak', 
                                                     'jumlah_kk_terdampak', 'jumlah_jiwa_terdampak']])
                        
                        # Tombol untuk update (hanya aktif jika tidak ada invalid)
                        if invalid_kecamatan:
                            st.warning("⚠️ Perbaiki nama kecamatan yang invalid terlebih dahulu sebelum melakukan update!")
                        else:
                            if st.button("🔄 Update Database", type="primary"):
                                conn = None
                                try:
                                    conn = engine.raw_connection()
                                    cursor = conn.cursor()
                                    
                                    updated_count = 0
                                    skipped_count = 0
                                    update_details = []

                                    # === FUNGSI HELPER ===
                                    def to_int(val):
                                        if val is None or pd.isna(val):
                                            return None
                                        try:
                                            return int(float(val))
                                        except (ValueError, TypeError):
                                            return None
                                        
                                    def to_float(val):
                                        if val is None or pd.isna(val):
                                            return None
                                        try:
                                            return float(val)
                                        except (ValueError, TypeError):
                                            return None
                                    
                                    with st.spinner(f"🔄 Updating data ke tabel {table_name}..."):
                                        progress_bar = st.progress(0)
                                        total_rows = len(df_upload)
                                        
                                        for idx, row in df_upload.iterrows():
                                            kecamatan_name = str(row['kecamatan']).strip().upper()
                                            
                                            # Skip jika kecamatan tidak valid
                                            if kecamatan_name not in valid_kecamatan_list:
                                                skipped_count += 1
                                                update_details.append({
                                                    'Kecamatan': row['kecamatan'],
                                                    'Status': '⏭️ Skipped',
                                                    'Reason': 'Tidak ditemukan di database'
                                                })
                                                continue
                                            
                                            # UPDATE query
                                            update_query = f"""
                                            UPDATE {table_name}
                                            SET 
                                                jumlah_rw_terdampak = %s,
                                                jumlah_kk_terdampak = %s,
                                                jumlah_jiwa_terdampak = %s,
                                                rata_ketinggian_air = %s,
                                                ketinggian_air_max = %s,
                                                jumlah_jiwa = %s,
                                                jumlah_disabilitas = %s,
                                                jumlah_lansia = %s
                                            WHERE UPPER(TRIM(kecamatan)) = %s
                                            """
                                            
                                            cursor.execute(update_query, (
                                                to_int(row['jumlah_rw_terdampak']),
                                                to_int(row['jumlah_kk_terdampak']),
                                                to_int(row['jumlah_jiwa_terdampak']),
                                                to_float(row['rata_ketinggian_air']),
                                                to_float(row['ketinggian_air_max']),
                                                to_int(row['jumlah_jiwa']),
                                                to_int(row['jumlah_disabilitas']),
                                                to_int(row['jumlah_lansia']),
                                                kecamatan_name
                                            ))
                                            
                                            if cursor.rowcount > 0:
                                                updated_count += cursor.rowcount
                                                update_details.append({
                                                    'Kecamatan': row['kecamatan'],
                                                    'Status': '✅ Updated',
                                                    'Reason': f'{cursor.rowcount} row(s) affected'
                                                })
                                            else:
                                                update_details.append({
                                                    'Kecamatan': row['kecamatan'],
                                                    'Status': '⚠️ Not Updated',
                                                    'Reason': 'Tidak ada perubahan atau kecamatan tidak ditemukan'
                                                })
                                            
                                            # Update progress bar
                                            progress_bar.progress((idx + 1) / total_rows)
                                        
                                        conn.commit()
                                        progress_bar.empty()
                                    
                                    cursor.close()
                                    conn.close()
                                    
                                    # === TAMPILKAN HASIL UPDATE ===
                                    st.success(f"🎉 Proses Update Selesai!")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("✅ Berhasil Update", updated_count, delta="rows")
                                    with col2:
                                        st.metric("⏭️ Skipped", skipped_count, delta="rows")
                                    with col3:
                                        st.metric("📊 Total Diproses", len(df_upload), delta="rows")
                                    
                                    # Detail hasil update
                                    with st.expander("📋 Detail Hasil Update per Kecamatan"):
                                        df_details = pd.DataFrame(update_details)
                                        st.dataframe(df_details, use_container_width=True)
                                    
                                    # Download laporan update
                                    csv_report = df_details.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="📥 Download Laporan Update",
                                        data=csv_report,
                                        file_name=f"laporan_update_{table_name}.csv",
                                        mime="text/csv"
                                    )
                                    
                                    if updated_count > 0:
                                        st.balloons()
                                        
                                        st.info(f"💡 Membersihkan cache untuk tahun {upload_tahun} dan Total (Agregasi)...")
                                        
                                        # Import fungsi dari CLUSTERING.py
                                        # Karena cache adalah global di Streamlit, cukup clear langsung
                                        try:
                                            # ✅ Clear cache untuk load_data dan get_data_hash
                                            from pages import CLUSTERING  # Sesuaikan dengan struktur folder Anda
                                            
                                            # Clear cache functions
                                            if hasattr(CLUSTERING, 'load_data'):
                                                CLUSTERING.load_data.clear()
                                            if hasattr(CLUSTERING, 'get_data_hash'):
                                                CLUSTERING.get_data_hash.clear()
                                            
                                            st.success(f"✅ Cache telah dibersihkan. Data terbaru akan diambil saat clustering berikutnya.")
                                        except:
                                            # Jika import gagal (struktur folder berbeda), gunakan clear manual
                                            # Cache akan auto-refresh karena checksum berubah
                                            st.success(f"✅ Data berhasil diupdate. Cache akan otomatis ter-refresh berdasarkan checksum database.")
                                    
                                except Exception as e:
                                    st.error(f"❌ Gagal update database: {str(e)}")
                                    if conn:
                                        conn.rollback()
                                finally:
                                    if conn:
                                        conn.close()
                        
                    except Exception as e:
                        st.error(f"❌ Gagal validasi kecamatan: {str(e)}")
                    finally:
                        if engine:
                            engine.dispose()
                            
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {str(e)}")

        st.divider()
        show_footer()

