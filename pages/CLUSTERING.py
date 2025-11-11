import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
import psycopg2
from sqlalchemy import create_engine
import folium
from streamlit_folium import st_folium
import json
import numpy as np
import toml
import hashlib

st.set_page_config(
    page_title="CLUSTERING",
    page_icon="🗺️",
)

st.title("Clustering Wilayah Rawan Banjir")
st.write("Pilih tipe data, tahun, dan parameter clustering untuk analisis Anda")

user_type = st.session_state.get("user_type")
if user_type == None:
    st.switch_page("BERANDA.py")

# Initialize session state
if 'clustering_result' not in st.session_state:
    st.session_state.clustering_result = None
if 'geojson_data' not in st.session_state:
    st.session_state.geojson_data = None
if 'last_params' not in st.session_state:
    st.session_state.last_params = None

# ===== LOAD DATABASE CONFIGURATION FROM secrets.toml =====

secrets = toml.load(".streamlit/secrets.toml")
db_host = secrets["database"]["db_host"]
db_port = secrets["database"]["db_port"]
db_name = secrets["database"]["db_name"]
db_user = secrets["database"]["db_user"]
db_password = secrets["database"]["db_password"]


# ===== FUNGSI UNTUK LABELING CLUSTER =====
CLUSTER_LABELS = {
    2: ['Rendah', 'Tinggi'],
    3: ['Rendah', 'Sedang', 'Tinggi'],
    4: ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi'],
    5: ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'],
    6: ['Sangat Rendah', 'Rendah', 'Cukup Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi'],
    7: ['Sangat Rendah', 'Rendah', 'Cukup Rendah', 'Sedang', 'Cukup Tinggi', 'Tinggi', 'Sangat Tinggi'],
}

def get_cluster_labels(n_clusters):
    """Dapatkan label sesuai jumlah cluster"""
    if n_clusters in CLUSTER_LABELS:
        return CLUSTER_LABELS[n_clusters]
    else:
        return [f'Cluster {i}' for i in range(n_clusters)]

def categorize_clusters(df):
    """
    Kategorisasi cluster berdasarkan tingkat keparahan banjir
    dari Rendah ke Tinggi berdasarkan rata-rata fitur per cluster
    """
    feature_cols = ["jumlah_rw_terdampak", "jumlah_kk_terdampak", 
                    "jumlah_jiwa_terdampak", "rata_ketinggian_air", 
                    "ketinggian_air_max"]
    
    available_cols = [col for col in feature_cols if col in df.columns]
    
    cluster_means = df.groupby('cluster')[available_cols].mean()
    cluster_means['skor_agregat'] = cluster_means.mean(axis=1)
    cluster_means = cluster_means.sort_values(by='skor_agregat').reset_index()
    
    clusters_without_noise = cluster_means[cluster_means['cluster'] != -1]
    n_clusters = len(clusters_without_noise)
    labels = get_cluster_labels(n_clusters)
    
    cluster_label_map = {}
    for i, row in clusters_without_noise.iterrows():
        cluster_label_map[row['cluster']] = labels[i]
    
    if -1 in df['cluster'].values:
        cluster_label_map[-1] = 'Noise/Outlier bernilai ekstrim'
    
    df['kategori'] = df['cluster'].map(cluster_label_map)
    
    return df, cluster_means


def plot_silhouette_analysis(X_scaled, cluster_labels, n_clusters):
    """
    Membuat silhouette plot untuk analisis kualitas cluster
    """
    # Filter out noise points for DBSCAN
    mask = cluster_labels != -1
    X_filtered = X_scaled[mask]
    labels_filtered = cluster_labels[mask]
    
    if len(np.unique(labels_filtered)) < 2:
        st.warning("⚠️ Tidak cukup cluster untuk analisis silhouette")
        return None
    
    # Hitung silhouette score
    silhouette_avg = silhouette_score(X_filtered, labels_filtered)
    sample_silhouette_values = silhouette_samples(X_filtered, labels_filtered)
    
    # Buat plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    y_lower = 10
    colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
    
    for i, color in zip(range(n_clusters), colors):
        # Ambil silhouette values untuk cluster i
        cluster_mask = labels_filtered == i
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_mask]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        
        # Label cluster di tengah
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), 
                fontsize=10, fontweight='bold')
        
        y_lower = y_upper + 10
    
    ax.set_title(f'Silhouette Plot untuk Setiap Cluster\n(Rata-rata Silhouette Score: {silhouette_avg:.3f})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster Label', fontsize=12)
    
    # Garis vertikal untuk rata-rata silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2, 
               label=f'Avg Score: {silhouette_avg:.3f}')
    ax.legend(loc='best')
    
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


#Get hash dari database untuk cache-busting
@st.cache_data(ttl=60, show_spinner=False)  # Cache hash hanya 1 menit
def get_data_hash(tipe, tahun_selected):
    """
    Mengambil hash dari checksum database untuk deteksi perubahan data.
    Hash ini digunakan untuk invalidate cache saat data berubah.
    """
    try:
        secrets = toml.load(".streamlit/secrets.toml")
        db_host = secrets["database"]["db_host"]
        db_port = secrets["database"]["db_port"]
        db_name = secrets["database"]["db_name"]
        db_user = secrets["database"]["db_user"]
        db_password = secrets["database"]["db_password"]
        
        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(connection_string)
        
        if tipe == "Per Tahun":
            table_name = f"kejadian_{tahun_selected}"
            # Ambil checksum data (sum semua nilai numerik sebagai fingerprint)
            query = f"""
            SELECT 
                COALESCE(SUM(
                    COALESCE(jumlah_rw_terdampak, 0) + 
                    COALESCE(jumlah_kk_terdampak, 0) + 
                    COALESCE(jumlah_jiwa_terdampak, 0) + 
                    COALESCE(rata_ketinggian_air, 0) + 
                    COALESCE(ketinggian_air_max, 0)
                ), 0) as checksum,
                COUNT(*) as row_count
            FROM {table_name}
            """
        else:
            # Untuk Total, ambil checksum gabungan semua tahun
            list_tahun = list(range(2018, 2026))
            union_queries = []
            for tahun in list_tahun:
                union_queries.append(f"""
                    SELECT 
                        COALESCE(SUM(
                            COALESCE(jumlah_rw_terdampak, 0) + 
                            COALESCE(jumlah_kk_terdampak, 0) + 
                            COALESCE(jumlah_jiwa_terdampak, 0) + 
                            COALESCE(rata_ketinggian_air, 0) + 
                            COALESCE(ketinggian_air_max, 0)
                        ), 0) as checksum,
                        COUNT(*) as row_count
                    FROM kejadian_{tahun}
                """)
            query = f"""
            SELECT 
                SUM(checksum) as checksum,
                SUM(row_count) as row_count
            FROM (
                {" UNION ALL ".join(union_queries)}
            ) subq
            """
        
        result = pd.read_sql(query, engine)
        engine.dispose()
        
        # Generate hash dari checksum + row count
        checksum_value = f"{result['checksum'].iloc[0]}_{result['row_count'].iloc[0]}"
        data_hash = hashlib.md5(checksum_value.encode()).hexdigest()
        
        return data_hash
        
    except Exception as e:
        # Jika gagal, return timestamp sebagai fallback
        import time
        fallback_hash = hashlib.md5(str(int(time.time())).encode()).hexdigest()
        st.warning(f"⚠️ Gagal mengambil data hash: {str(e)}. Menggunakan fallback.")
        return fallback_hash


@st.cache_data(ttl=600, show_spinner=False)  # Cache 10 menit
def load_data(tipe, tahun_selected, data_hash):
    """
    Mengambil data dari database berdasarkan tipe (Per Tahun atau Total).
    Parameter data_hash: Hash dari checksum database untuk cache-busting otomatis
    """
    # Baca konfigurasi database dari secrets.toml
    secrets = toml.load(".streamlit/secrets.toml")
    db_host = secrets["database"]["db_host"]
    db_port = secrets["database"]["db_port"]
    db_name = secrets["database"]["db_name"]
    db_user = secrets["database"]["db_user"]
    db_password = secrets["database"]["db_password"]
    
    
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_string)
    query = ""
    
    if tipe == "Per Tahun":
        if tahun_selected is None:
            st.error("Silakan pilih tahun terlebih dahulu.")
            return None
        table_name = f"kejadian_{tahun_selected}"
        query = f"SELECT * FROM {table_name} ORDER BY kecamatan ASC"
        
    else:
        list_tahun = list(range(2018, 2026))
        union_all_queries = []
        
        for tahun_loop in list_tahun:
            table_name = f"kejadian_{tahun_loop}"
            union_all_queries.append(
                f"""
                SELECT 
                    kecamatan, 
                    jumlah_rw_terdampak, 
                    jumlah_kk_terdampak, 
                    jumlah_jiwa_terdampak, 
                    rata_ketinggian_air, 
                    ketinggian_air_max 
                FROM {table_name}
                """
            )
        union_string = " UNION ALL ".join(union_all_queries)
        
        query = f"""
        WITH all_data AS (
            {union_string}
        ),
        aggregated_data AS (
            SELECT
                kecamatan,
                COALESCE(SUM(jumlah_rw_terdampak), 0) AS jumlah_rw_terdampak,
                COALESCE(SUM(jumlah_kk_terdampak), 0) AS jumlah_kk_terdampak,
                COALESCE(SUM(jumlah_jiwa_terdampak), 0) AS jumlah_jiwa_terdampak,
                COALESCE(AVG(rata_ketinggian_air), 0) AS rata_ketinggian_air,
                COALESCE(MAX(ketinggian_air_max), 0) AS ketinggian_air_max
            FROM all_data
            GROUP BY kecamatan
        ),
        demo_data_2025 AS (
            SELECT
                kecamatan,
                jumlah_jiwa,
                jumlah_disabilitas,
                jumlah_lansia
            FROM kejadian_2025
        )
        SELECT
            agg.*,
            demo.jumlah_jiwa,
            demo.jumlah_disabilitas,
            demo.jumlah_lansia
        FROM aggregated_data agg
        LEFT JOIN demo_data_2025 demo ON agg.kecamatan = demo.kecamatan
        ORDER BY agg.kecamatan ASC;  -- 
        """

    with st.spinner("Membaca data dari database..."):
        df = pd.read_sql(query, engine)
        engine.dispose()
    
    if df.empty:
        st.warning("⚠️ Data tidak ditemukan untuk parameter yang dipilih.")
        return None
            
    return df


def show_footer():
    st.markdown("""
    <hr style='margin: 0.5rem 0;'>
    <div style='text-align: center; font-size: 0.9rem; padding: 0.5rem 0;'>
        © 2025 Gregorio Melvin Karnikov
    </div>
    """, unsafe_allow_html=True)


# Pemetaan

geojson_path = os.path.join("KECAMATAN.geojson")

if os.path.exists(geojson_path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        st.session_state.geojson_data = json.load(f)


# Pilihan Tipe Data
tipe_data = st.radio(
    "Pilih Tipe Data",
    options=["Per Tahun", "Total (Agregasi)"],
    index=0,
    horizontal=True
)

# Dropdown tahun kondisional
tahun = None
if tipe_data == "Per Tahun":
    tahun = st.selectbox(
        "Pilih Tahun Data",
        options=list(range(2018, 2026)),
        index=0
    )
    st.write(f"Tahun yang dipilih: **{tahun}**")
else:
    st.write("Data agregasi dari **2018-2025** akan digunakan.")

# Radio button untuk metode clustering
if tipe_data == "Per Tahun" and tahun == 2025:
    st.warning("⚠️ **Metode K-Medoids tidak tersedia untuk tahun 2025**")
    st.info("""
    📌 **Alasan:**
    - Data tahun 2025 memiliki banyak kecamatan dengan nilai 0 (tidak terdampak banjir)
    - Hal ini menyebabkan data menjadi identik setelah normalisasi
    - K-Medoids kesulitan membentuk cluster yang valid dengan data seperti ini
    
    💡 **Alternatif yang tersedia:**
    - Gunakan **DBSCAN** yang lebih robust terhadap data dengan banyak nilai 0
    - Atau pilih **Total (Agregasi)** untuk analisis keseluruhan 2018-2025
    - Atau pilih tahun lain (2018-2024)
    """)
    metode = "DBSCAN"  # Set default ke DBSCAN
    st.success("✅ Menggunakan metode **DBSCAN** untuk tahun 2025")
else:
    metode = st.radio(
        "Pilih Metode Clustering",
        options=["K-Medoids", "DBSCAN"],
        horizontal=True
    )

st.divider()

# Fungsi untuk membuat peta dengan kategori
def create_cluster_map(df, geojson_data, metode_name):
    df['kecamatan_normalized'] = df['kecamatan'].str.upper().str.strip()
    
    m = folium.Map(
        location=[-6.2088, 106.8456],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    cluster_dict = dict(zip(df['kecamatan_normalized'], df['cluster']))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
              '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']
    
    def style_function(feature):
        kecamatan_name = feature['properties']['kecamatan'].upper().strip()
        cluster = cluster_dict.get(kecamatan_name, -1)
        
        if cluster == -1:
            color = '#cccccc'
        else:
            color = colors[cluster % len(colors)]
        
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }
    
    def highlight_function(feature):
        return {
            'fillColor': '#ffff00',
            'color': 'black',
            'weight': 3,
            'fillOpacity': 0.9
        }
    
    folium.GeoJson(
        geojson_data,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['kecamatan', 'kab_kota'],
            aliases=['Kecamatan:', 'Kota:'],
            localize=True
        ),
        popup=folium.GeoJsonPopup(
            fields=['kecamatan', 'kab_kota'],
            aliases=['Kecamatan:', 'Kota:']
        )
    ).add_to(m)
    
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
        <p style="margin: 0; font-weight: bold;">{metode_name} Clusters</p>
    '''
    
    unique_clusters = sorted(df['cluster'].unique())
    for cluster in unique_clusters:
        kategori_series = df[df['cluster'] == cluster]['kategori']
        if kategori_series.empty:
            continue
        kategori = kategori_series.iloc[0]
        count = len(df[df['cluster'] == cluster])
        
        if cluster >= 0:
            color = colors[cluster % len(colors)]
            legend_html += f'<p style="margin: 3px 0;"><i style="background:{color}; width: 18px; height: 18px; display: inline-block; margin-right: 5px;"></i>{kategori} ({count})</p>'
        else:
            legend_html += f'<p style="margin: 3px 0;"><i style="background:#cccccc; width: 18px; height: 18px; display: inline-block; margin-right: 5px;"></i>{kategori} ({count})</p>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Parameter berdasarkan metode yang dipilih
if metode == "K-Medoids":
    st.subheader("Parameter K-Medoids")
    
    k = st.slider(
        "Jumlah Cluster (k)",
        min_value=2,
        max_value=10,
        value=3,
        help="Tentukan jumlah cluster yang diinginkan"
    )
    
    max_iter = 300
    random_state = 42
    
    current_params = {'metode': metode, 'tipe_data': tipe_data, 'tahun': tahun, 'k': k}
    if st.session_state.last_params != current_params:
        if st.session_state.last_params is not None:
            st.session_state.clustering_result = None
    
    info_text = f"K-Medoids dengan {k} cluster"
    if tipe_data == "Per Tahun":
        info_text += f" pada data tahun {tahun}"
    else:
        info_text += " pada data agregasi (2018-2025)"
    st.info(f"📊 {info_text}")
    
    if st.button("🚀 Jalankan K-Medoids", type="primary"):
        
        # ✅ Dapatkan hash data terkini (menggantikan version_dummy)
        with st.spinner("🔍 Memeriksa versi data..."):
            data_hash = get_data_hash(tipe=tipe_data, tahun_selected=tahun)
        
        # ✅ Load data dengan hash
        df = load_data(
            tipe=tipe_data, 
            tahun_selected=tahun, 
            data_hash=data_hash
        )
        
        if df is not None and not df.empty:
            try:
                X = df[["jumlah_rw_terdampak", "jumlah_kk_terdampak", "jumlah_jiwa_terdampak", 
                        "rata_ketinggian_air", "ketinggian_air_max"]]
                
                with st.spinner("Menormalisasi data..."):
                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(X)
                
                with st.spinner(f"⚡ Menjalankan K-Medoids dengan {k} cluster..."):
                    kmedoids = KMedoids(
                        n_clusters=k, 
                        random_state=int(random_state), 
                        max_iter=int(max_iter),
                        metric="euclidean"
                    )
                    df["cluster"] = kmedoids.fit_predict(X_scaled)
                
                with st.spinner("Melakukan kategorisasi cluster..."):
                    df, cluster_means = categorize_clusters(df)
                
                score = silhouette_score(X_scaled, df["cluster"])
                
                st.session_state.clustering_result = {
                    'df': df,
                    'score': score,
                    'k': k,
                    'metode': 'K-Medoids',
                    'tipe_data': tipe_data,
                    'tahun': tahun,
                    'X_scaled': X_scaled,
                    'kmedoids': kmedoids,
                    'cluster_means': cluster_means
                }
                
                st.session_state.last_params = current_params
                                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat clustering: {str(e)}")

else:  # DBSCAN
    st.subheader("Parameter DBSCAN")
    
    epsilon = st.slider(
        "Epsilon (ε)",
        min_value=0.05,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Jarak maksimum antara dua sampel untuk dianggap sebagai tetangga"
    )
    
    min_pts = st.slider(
        "Min Points (MinPts)",
        min_value=2,
        max_value=10,
        value=5,
        help="Jumlah minimum sampel dalam neighborhood untuk membentuk core point"
    )
    
    metric = "euclidean"
    
    current_params = {'metode': metode, 'tipe_data': tipe_data, 'tahun': tahun, 'epsilon': epsilon, 'min_pts': min_pts}
    if st.session_state.last_params != current_params:
        if st.session_state.last_params is not None:
            st.session_state.clustering_result = None
    
    info_text = f"DBSCAN dengan ε={epsilon} dan MinPts={min_pts}"
    if tipe_data == "Per Tahun":
        info_text += f" pada data tahun {tahun}"
    else:
        info_text += " pada data agregasi (2018-2025)"
    st.info(f"📊 {info_text}")
    
    if st.button("🚀 Jalankan DBSCAN", type="primary"):
        
        # ✅ Dapatkan hash data terkini (menggantikan version_dummy)
        with st.spinner("🔍 Memeriksa versi data..."):
            data_hash = get_data_hash(tipe=tipe_data, tahun_selected=tahun)
        
        # ✅ Load data dengan hash
        df = load_data(
            tipe=tipe_data, 
            tahun_selected=tahun, 
            data_hash=data_hash
        )
        
        if df is not None and not df.empty:
            try:
                X = df[["jumlah_rw_terdampak", "jumlah_kk_terdampak", "jumlah_jiwa_terdampak", 
                        "rata_ketinggian_air", "ketinggian_air_max"]]
                
                with st.spinner("Menormalisasi data..."):
                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(X)
                
                with st.spinner(f"⚡ Menjalankan DBSCAN..."):
                    dbscan = DBSCAN(eps=epsilon, min_samples=int(min_pts), metric=metric)
                    df["cluster"] = dbscan.fit_predict(X_scaled)
                
                unique_clusters = set(df["cluster"])
                valid_clusters = unique_clusters - {-1}
                n_clusters = len(valid_clusters)
                n_noise = list(df["cluster"]).count(-1)
                
                if n_clusters == 0:
                    st.error("❌ DBSCAN tidak membentuk cluster sama sekali (semua data adalah noise)")
                    st.info("💡 **Saran:** Perbesar nilai Epsilon atau perkecil MinPts")
                    st.stop()
                
                if n_clusters < 2:
                    st.warning(f"⚠️ DBSCAN hanya membentuk {n_clusters} cluster ({len(df)-n_noise} data) dan {n_noise} noise")
                    st.info("💡 **Saran:** Sesuaikan parameter Epsilon atau MinPts untuk membentuk lebih banyak cluster")
                    score_text = "N/A (butuh > 1 cluster)"
                else:
                    mask = df["cluster"] != -1
                    try:
                        score = silhouette_score(X_scaled[mask], df[mask]["cluster"])
                        score_text = f"{score:.3f}"
                    except ValueError as e:
                        st.warning(f"⚠️ Tidak dapat menghitung silhouette score: {str(e)}")
                        score_text = "Null"
                
                with st.spinner("Melakukan kategorisasi cluster..."):
                    df, cluster_means = categorize_clusters(df)
                
                st.session_state.clustering_result = {
                    'df': df,
                    'score': score_text,
                    'epsilon': epsilon,
                    'min_pts': min_pts,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'metode': 'DBSCAN',
                    'tipe_data': tipe_data,
                    'tahun': tahun,
                    'X_scaled': X_scaled,
                    'cluster_means': cluster_means
                }
                
                st.session_state.last_params = current_params
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat clustering: {str(e)}")


# TAMPILKAN HASIL dari session state
if st.session_state.clustering_result is not None:
    result = st.session_state.clustering_result
    df = result['df']
    
    st.divider()
    st.subheader("📊 Hasil Clustering")
    
    # Metrik
    if result['metode'] == 'K-Medoids':
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Silhouette Score", f"{result['score']:.3f}")
        with col2:
            st.metric("Jumlah Cluster", result['k'])
    else:  # DBSCAN
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jumlah Cluster", result['n_clusters'])
        with col2:
            st.metric("Noise Points", result['n_noise'])
        with col3:
            st.metric("Silhouette Score", result['score'])
    
    # Tampilkan tabel statistik cluster dengan kategori
    st.subheader("📈 Karakteristik Setiap Kategori")
    cluster_stats = result['cluster_means'].copy()
    
    if 'cluster' in cluster_stats.columns:
        cluster_label_map = dict(zip(df['cluster'], df['kategori']))
        cluster_stats['kategori'] = cluster_stats['cluster'].map(cluster_label_map)
        
        cols = ['cluster', 'kategori'] + [col for col in cluster_stats.columns if col not in ['cluster', 'kategori']]
        cluster_stats = cluster_stats[cols]
        
        st.dataframe(cluster_stats.round(2), use_container_width=True)
    
    
    # ===== VISUALISASI SILHOUETTE =====
    st.divider()
    st.subheader("📉 Analisis Silhouette")
    
    st.markdown("""
    **Silhouette Analysis** mengukur seberapa baik setiap data point cocok dengan cluster-nya dibandingkan dengan cluster lain.
    - **Nilai mendekati +1**: Data sangat cocok dengan cluster-nya
    - **Nilai mendekati 0**: Data berada di perbatasan antar cluster
    - **Nilai negatif**: Data mungkin salah ditempatkan ke cluster yang salah
    """)
    
    # Hitung jumlah cluster yang valid (tanpa noise)
    if result['metode'] == 'DBSCAN':
        n_clusters_valid = result['n_clusters']
        cluster_labels = df['cluster'].values
    else:
        n_clusters_valid = result['k']
        cluster_labels = df['cluster'].values
    
    # Plot silhouette hanya jika ada cluster valid
    if n_clusters_valid >= 2:
        fig_silhouette = plot_silhouette_analysis(
            result['X_scaled'], 
            cluster_labels, 
            n_clusters_valid
        )
        if fig_silhouette:
            st.pyplot(fig_silhouette)
            
            # Interpretasi hasil
            if result['metode'] == 'K-Medoids':
                avg_score = result['score']
            else:
                # Untuk DBSCAN, hitung ulang atau ambil dari string
                if isinstance(result['score'], str):
                    if result['score'] != "N/A (butuh > 1 cluster)" and result['score'] != "Null":
                        avg_score = float(result['score'])
                    else:
                        avg_score = None
                else:
                    avg_score = result['score']
            
            if avg_score is not None and isinstance(avg_score, (int, float)):
                st.markdown("### 📊 Interpretasi Silhouette Score:")
                if avg_score >= 0.7:
                    st.success(f"✅ **Excellent** ({avg_score:.3f}): Struktur cluster sangat kuat dan jelas")
                elif avg_score >= 0.5:
                    st.success(f"✅ **Good** ({avg_score:.3f}): Struktur cluster yang baik dan reasonable")
                elif avg_score >= 0.3:
                    st.warning(f"⚠️ **Fair** ({avg_score:.3f}): Struktur cluster lemah, overlap mungkin terjadi")
                else:
                    st.error(f"❌ **Poor** ({avg_score:.3f}): Tidak ada struktur cluster substansial")
                
                st.info("💡 **Tips:** Jika silhouette score rendah, coba ubah jumlah cluster atau parameter clustering")
    else:
        st.warning("⚠️ Tidak cukup cluster untuk membuat analisis silhouette (minimal 2 cluster)")
    
    # Peta Clustering
    if st.session_state.geojson_data is not None:
        st.divider()
        st.subheader("🗺️ Visualisasi Peta Clustering")
        cluster_map = create_cluster_map(df, st.session_state.geojson_data, result['metode'])
        st_folium(cluster_map, width=800, height=600)
    
    # Tampilkan tabel hasil dengan kategori
    st.divider()
    st.subheader("📋 Hasil Clustering per Wilayah")
    
    # Hanya tambahkan kolom jika ada di dataframe (untuk Per Tahun vs Total)
    all_cols = ["kecamatan", "cluster", "kategori"]
    for col in ["jumlah_jiwa", "jumlah_disabilitas", "jumlah_lansia"]:
        if col in df.columns:
            all_cols.append(col)
            
    # Pastikan urutan benar
    ordered_cols = ["kecamatan"]
    if "jumlah_jiwa" in all_cols: ordered_cols.append("jumlah_jiwa")
    if "jumlah_disabilitas" in all_cols: ordered_cols.append("jumlah_disabilitas")
    if "jumlah_lansia" in all_cols: ordered_cols.append("jumlah_lansia")
    ordered_cols.extend(["cluster", "kategori"])
    
    st.dataframe(df[ordered_cols].sort_values("cluster"), use_container_width=True)
    
    # Visualisasi distribusi kategori
    st.divider()
    st.subheader("📊 Distribusi Kategori")
    col1, col2 = st.columns(2)
    
    with col1:
        kategori_counts = df['kategori'].value_counts().sort_index()
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        kategori_counts.plot(kind='bar', ax=ax_bar, color='steelblue')
        ax_bar.set_xlabel("Kategori", fontsize=12)
        ax_bar.set_ylabel("Jumlah Kecamatan", fontsize=12)
        ax_bar.set_title("Distribusi Kecamatan per Cluster", fontsize=14, fontweight='bold')
        ax_bar.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig_bar)
    
    with col2:
        fig_pie, ax_pie = plt.subplots(figsize=(8, 5))
        kategori_counts.plot(kind='pie', ax=ax_pie, autopct='%1.1f%%', startangle=90)
        ax_pie.set_ylabel("")
        ax_pie.set_title("Proporsi Cluster", fontsize=14, fontweight='bold')
        st.pyplot(fig_pie)
    
    # Visualisasi dengan PCA
    st.divider()
    st.subheader("📍 Visualisasi PCA 2D")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(result['X_scaled'])
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
              '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']
    
    unique_clusters = sorted(df['cluster'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cluster_id in unique_clusters:
        mask = df['cluster'] == cluster_id
        if cluster_id == -1:
            ax.scatter(
                X_pca[mask, 0], 
                X_pca[mask, 1], 
                c='#cccccc', 
                s=100, 
                alpha=0.6, 
                edgecolors='black',
                linewidths=0.5,
                label='Noise/Outlier'
            )
        else:
            kategori = df[mask]['kategori'].iloc[0]
            ax.scatter(
                X_pca[mask, 0], 
                X_pca[mask, 1], 
                c=colors[cluster_id % len(colors)], 
                s=100, 
                alpha=0.6,
                edgecolors='black',
                linewidths=0.5,
                label=f'{kategori} (Cluster {cluster_id})'
            )
    
    if result['metode'] == 'K-Medoids':
        medoids_pca = pca.transform(result['kmedoids'].cluster_centers_)
        ax.scatter(
            medoids_pca[:,0], 
            medoids_pca[:,1],
            c="red", 
            marker="X", 
            s=300, 
            edgecolors='black', 
            linewidths=2, 
            label="Medoids",
            zorder=5
        )
    
    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    
    if result['tipe_data'] == 'Total (Agregasi)':
        title_tahun = "Total (Agregasi 2018-2025)"
    else:
        title_tahun = f"Tahun {result['tahun']}"
    
    if result['metode'] == 'K-Medoids':
        ax.set_title(f"K-Medoids Clustering (k={result['k']}, {title_tahun})", 
                     fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"DBSCAN Clustering (ε={result['epsilon']}, MinPts={result['min_pts']}, {title_tahun})", 
                     fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Informasi variance explained oleh PCA
    st.caption(f"💡 PCA Component 1 menjelaskan {pca.explained_variance_ratio_[0]*100:.1f}% variance, "
               f"Component 2 menjelaskan {pca.explained_variance_ratio_[1]*100:.1f}% variance")

show_footer()