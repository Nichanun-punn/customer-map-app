import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import polyline
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import base64

# โหลดโลโก้ SCGP
with open("logo.png", "rb") as f:
    base64_logo = base64.b64encode(f.read()).decode()

# Header แบบโลโก้อย่างเดียว
st.markdown(f"""
    <style>
    .fixed-header {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 9999;
        background-color: white;
        padding: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}
    .fixed-header-content {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    .fixed-header-content img {{
        height: 50px;
    }}
    .spacer {{
        height: 1px;
    }}
    </style>

    <div class="fixed-header">
        <div class="fixed-header-content">
            <img src="data:image/png;base64,{base64_logo}">
        </div>
    </div>
    <div class="spacer"></div>
""", unsafe_allow_html=True)

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Customer Map Planner", layout="wide")

# 🔁 Navigation Menu
page = st.sidebar.radio("เลือกหน้า", ["📍 แผนที่ลูกค้า", "📊 ข้อมูลลูกค้า"])

factory_latlon = (13.543372, 100.663351)
default_excel_path = "Class.xlsx"

# ---------------- CACHE ---------------- #
@st.cache_data
def get_osrm_route(start, end):
    url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        coords = polyline.decode(data['routes'][0]['geometry'])
        distance_km = data['routes'][0]['distance'] / 1000
        return coords, distance_km
    return [], 0

@st.cache_data
def compute_clusters(df):
    features = df[['Lat', 'Long']].copy()
    best_k = 0
    best_score = -1
    best_model = None
    for k in range(2, min(len(df), 200)):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(features)
        score = silhouette_score(features, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_model = model
    df['Cluster'] = best_model.predict(features)
    return df

# ---------------- LOAD FILE ---------------- #
df = None
uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File (optional)", type=['xlsx'])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("✅ Loaded file from upload")
else:
    try:
        df = pd.read_excel(default_excel_path)
        st.sidebar.info(f"📥 Loaded default file: {default_excel_path}")
    except FileNotFoundError:
        st.sidebar.error(f"❌ ไม่พบไฟล์: {default_excel_path}")
        st.stop()

# ---------------- VALIDATE ---------------- #
required_cols = {'Name', 'Lat', 'Long', 'Class'}
if not required_cols.issubset(df.columns):
    st.error("❌ File must contain 'Name', 'Lat', 'Long', 'Class' columns.")
    st.stop()

df = compute_clusters(df)

# ---------------- หน้าแผนที่ ---------------- #
if page == "📍 แผนที่ลูกค้า":
    st.title("📍 Customer Mapping & Route Planning")

    col1, col2 = st.columns([2, 2])

    with col2:
        st.subheader("⚙️ ตั้งค่าการแสดงแผนที่")
        tile_option = st.selectbox("🗺️ เลือกโหมดแผนที่", [
            '☀️ โหมดกลางวัน (Positron)',
            '🌙 โหมดกลางคืน (Dark Matter)',
            '🌍 โหมดปกติ (OSM)'
        ])
        customer_option = st.selectbox("👤 เลือกลูกค้า", ['-- Show All Customers --'] + df['Name'].tolist())

    selected_name = None if customer_option == '-- Show All Customers --' else customer_option
    if selected_name not in df['Name'].values:
        selected_name = None

    highlight_name = None
    dist1 = dist2 = 0
    route_order = None

    def plot_map(selected_name=None, tile='CartoDB positron', nearby_highlight=None):
        global dist1, dist2, route_order

        from folium.plugins import Fullscreen
        m = folium.Map(location=[df['Lat'].mean(), df['Long'].mean()], zoom_start=10, tiles={
            '☀️ โหมดกลางวัน (Positron)': 'CartoDB positron',
            '🌙 โหมดกลางคืน (Dark Matter)': 'CartoDB dark_matter',
            '🌍 โหมดปกติ (OSM)': 'OpenStreetMap'
        }[tile])
        Fullscreen(position='topright').add_to(m)

        folium.Marker(factory_latlon, popup="🏠 Factory", icon=folium.Icon(color='blue')).add_to(m)
        bounds = []
        total_distance_km = 0

        if selected_name:
            selected = df[df['Name'] == selected_name].iloc[0]
            dest_latlon = (selected['Lat'], selected['Long'])
            folium.Marker(dest_latlon,
                        popup=f"📍 {selected_name}<br>Class: {selected['Class']}",
                        icon=folium.Icon(color='red')).add_to(m)

            cluster_members = df[(df['Cluster'] == selected['Cluster']) & (df['Name'] != selected_name)].copy()

            if cluster_members.empty:
                st.info("ℹ️ ไม่มีลูกค้าใกล้เคียงในคลัสเตอร์")
                nearest_df = pd.DataFrame(columns=['Name', 'Class', 'Distance_km'])
            else:
                cluster_members['Distance_km'] = cluster_members.apply(
                    lambda row: geodesic(dest_latlon, (row['Lat'], row['Long'])).km, axis=1)
                nearest_df = cluster_members.sort_values('Distance_km')

                for _, row in nearest_df.iterrows():
                    latlon = (row['Lat'], row['Long'])
                    color = 'green' if row['Name'] == nearby_highlight else 'gray'
                    popup = f"📍 {row['Name']}<br>Class: {row['Class']}"
                    folium.Marker(latlon, popup=popup, icon=folium.Icon(color=color)).add_to(m)

            if nearby_highlight:
                highlight_row = nearest_df[nearest_df['Name'] == nearby_highlight].iloc[0]
                highlight_latlon = (highlight_row['Lat'], highlight_row['Long'])

                route1 = [factory_latlon, dest_latlon, highlight_latlon, factory_latlon]
                d1 = [get_osrm_route(route1[i], route1[i + 1])[1] for i in range(3)]
                dist1 = sum(d1)

                route2 = [factory_latlon, highlight_latlon, dest_latlon, factory_latlon]
                d2 = [get_osrm_route(route2[i], route2[i + 1])[1] for i in range(3)]
                dist2 = sum(d2)

                final_route = route1 if dist1 <= dist2 else route2
                total_distance_km = min(dist1, dist2)
                route_order = f"โรงงาน ➝ **{selected_name}** ➝ **{highlight_name}** " if dist1 <= dist2 else f"โรงงาน ➝ **{highlight_name}** ➝ **{selected_name}**"

                for i in range(len(final_route) - 1):
                    coords, _ = get_osrm_route(final_route[i], final_route[i + 1])
                    folium.PolyLine(
                        coords,
                        color=['orange', 'green', 'blue'][i],
                        weight=4,
                        dash_array='5,5' if final_route[i + 1] == factory_latlon else None
                    ).add_to(m)
                    bounds.append(final_route[i + 1])
            else:
                coords1, d1 = get_osrm_route(factory_latlon, dest_latlon)
                coords2, d2 = get_osrm_route(dest_latlon, factory_latlon)
                dist1 = d1
                dist2 = d2
                total_distance_km = d1 + d2
                folium.PolyLine(coords1, color='orange', weight=4).add_to(m)
                folium.PolyLine(coords2, color='blue', weight=4, dash_array='5,5').add_to(m)
                bounds.extend([dest_latlon, factory_latlon])

            m.fit_bounds(bounds)
            return m, nearest_df[['Name', 'Class', 'Distance_km']], total_distance_km

        else:
            class_color_map = {
                "KAM": "darkred", "Diamond": "red", "Titanium": "lightred", "Platinum": "orange",
                "Gold": "darkblue", "Silver": "purple", "Bronze": "lightblue", "Zinc": "gray", "SUB-CONTRACTOR": "gray"
            }
            for _, row in df.iterrows():
                latlon = [row['Lat'], row['Long']]
                color = class_color_map.get(row['Class'], 'gray')
                folium.Marker(latlon,
                            popup=f"📍 {row['Name']}<br>Class: {row['Class']}",
                            icon=folium.Icon(color=color)).add_to(m)
                bounds.append(latlon)
            bounds.append(factory_latlon)
            m.fit_bounds(bounds)
            return m, pd.DataFrame(columns=['Name', 'Class', 'Distance_km']), 0

    with st.spinner("🔄 กำลังโหลดแผนที่และเส้นทาง..."):
        m, nearest_df, total_distance_km = plot_map(selected_name, tile_option)

    with col2:
        if not nearest_df.empty:
            st.subheader("👥 ลูกค้าที่อยู่ใกล้เคียง")
            st.markdown("""เส้นทาง: <br>
            <span style='color: orange;'>🟧 โรงงาน ➝ ลูกค้ารายแรก</span><br>
            <span style='color: green;'>🟩 ลูกค้าหลัก ➝ ลูกค้ารายที่สอง</span><br>
            <span style='color: blue;'>🔵 กลับโรงงาน</span>
            """, unsafe_allow_html=True)
            nearest_df_display = nearest_df[['Name', 'Class']].reset_index(drop=True)
            nearest_df_display.index += 1
            st.dataframe(nearest_df_display, use_container_width=True, hide_index=True, height=150)

            highlight_name = st.selectbox("🔎 เลือกลูกค้าใกล้เคียงเพื่อวางแผนเส้นทาง",
                                        ['--'] + nearest_df_display['Name'].tolist())
            if highlight_name == '--':
                highlight_name = None

    with col1:
        st.markdown("🔵 โรงงาน 🔴 ลูกค้าหลัก 🟢 ลูกค้าใกล้เคียงที่เลือก 🔘 ลูกค้าใกล้เคียงอื่น ๆ", unsafe_allow_html=True)
        with st.spinner("🔄 กำลังอัปเดตเส้นทาง..."):
            m, _, total_distance_km = plot_map(selected_name, tile_option, nearby_highlight=highlight_name)
        st_folium(m, use_container_width=True, height=770)

    with col2:
        if selected_name and total_distance_km > 0:
            if route_order:
                st.info(f" **เส้นทางที่สั้นที่สุด**: {route_order}")
            st.success(f"🚚 ระยะทางรวมทั้งหมด: **{total_distance_km:.2f} km**")

# ---------------- หน้าแสดงข้อมูลลูกค้า ---------------- #
elif page == "📊 ข้อมูลลูกค้า":
    st.title("📊 ข้อมูลลูกค้า")
    st.dataframe(df, use_container_width=True)

    # ปุ่มดาวน์โหลด
    st.download_button(
        "⬇️ ดาวน์โหลดข้อมูลลูกค้า",
        data=df.to_csv(index=False).encode('utf-8-sig'),
        file_name="customer_data.csv",
        mime="text/csv"
    )
