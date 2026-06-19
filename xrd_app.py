import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import re
import io

# --- コールバック関数（一括設定を個別設定に自動同期させる魔法） ---
def sync_bulk_color(idx, n_peaks):
    if f"bulk_col_{idx}" in st.session_state:
        new_color = st.session_state[f"bulk_col_{idx}"]
        for j in range(n_peaks):
            st.session_state[f"col_{idx}_{j}"] = new_color

def sync_bulk_size(idx, n_peaks):
    if f"bulk_siz_{idx}" in st.session_state:
        new_size = st.session_state[f"bulk_siz_{idx}"]
        for j in range(n_peaks):
            st.session_state[f"siz_{idx}_{j}"] = new_size

def sync_bulk_offset(idx, n_peaks):
    if f"bulk_off_{idx}" in st.session_state:
        new_off = st.session_state[f"bulk_off_{idx}"]
        for j in range(n_peaks):
            st.session_state[f"off_{idx}_{j}"] = new_off

# --- 【アップデート1】データ読み込みの高速化・万能化（キャッシュ機能付き） ---
@st.cache_data
def load_data(file_bytes):
    # pandasを使ってカンマ、スペース、タブ区切りを自動判別して読み込む
    content = file_bytes.decode('utf-8', errors='ignore')
    lines = [line for line in content.split('\n') if not line.startswith('*')]
    content_clean = '\n'.join(lines)
    
    df = pd.read_csv(io.StringIO(content_clean), sep=r'[,\s]+', engine='python', header=None)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    return x, y

# --- 【アップデート2】ノイズ除去と正規化処理のキャッシュ化 ---
@st.cache_data
def process_signal(y, apply_smooth, window, poly):
    if apply_smooth and len(y) > window:
        y_proc = savgol_filter(y, window, poly)
    else:
        y_proc = y
    # 正規化
    y_norm = (y_proc - np.min(y_proc)) / (np.max(y_proc) - np.min(y_proc))
    return y_norm

# --- 1. アプリの基本設定 ---
st.set_page_config(page_title="XRD Data Analyzer", layout="wide")
st.title("XRD Data Analyzer (Pro)")

# --- 2. サイドバー（4つのタブ） ---
tab1, tab2, tab3, tab4 = st.sidebar.tabs(["1. 軸/範囲", "2. ピーク", "3. 凡例", "4. 個別データ"])

# --- タブ1：軸と描画範囲の調整 ---
with tab1:
    st.header("軸と描画範囲の調整")
    offset = st.slider("Y軸オフセット (分離幅)", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    
    st.subheader("X軸の描画範囲")
    x_min = st.number_input("最小値 (2θ)", value=25.0, step=5.0)
    x_max = st.number_input("最大値 (2θ)", value=50.0, step=5.0)
    
    st.subheader("Y軸の描画範囲")
    y_min = st.number_input("最小値 (Intensity)", value=-0.1, step=0.1)
    y_max = st.number_input("最大値 (Intensity)", value=2.0, step=0.1)
    
    st.subheader("グラフの表示サイズ")
    fig_width = st.slider("グラフの横幅", min_value=4.0, max_value=20.0, value=10.0, step=0.5)
    fig_height = st.slider("グラフの縦幅", min_value=4.0, max_value=12.0, value=6.0, step=0.5)

# --- タブ2：全体ピーク検出設定 ---
with tab2:
    st.header("ピークの全体設定")
    show_peaks = st.checkbox("ピーク位置 (▼) を表示・編集する", value=True)
    prominence_val = st.slider("自動検出の感度 (ノイズ除去)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    
    st.markdown("---")
    st.subheader("スムージング (ノイズ除去)")
    apply_smoothing = st.checkbox("波形をなめらかにする", value=False)
    if apply_smoothing:
        smooth_window = st.slider("なめらかさの強さ (大きいほど強力)", min_value=3, max_value=51, value=11, step=2)
    else:
        smooth_window = 11

# --- タブ3：凡例の設定 ---
with tab3:
    st.header("凡例 (Legend) の設定")
    show_legend = st.checkbox("凡例を表示する", value=True)
    if show_legend:
        legend_loc = st.selectbox(
            "凡例の位置", 
            ["upper right (右上)", "upper left (左上)", "lower right (右下)", "lower left (左下)", "枠外 (右側)"]
        )
    else:
        legend_loc = None

default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# --- メイン画面（ファイルアップロードと描画） ---
uploaded_files = st.file_uploader("データファイルのアップロード (.ras, .csv, .txt)", accept_multiple_files=True)

if not uploaded_files:
    with tab4:
        st.info("データをアップロードすると、ここに個別設定が表示されます。")

if uploaded_files:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    with tab4:
        st.header("各データと個別ピークの詳細設定")

        for i, file in enumerate(uploaded_files):
            st.markdown(f"**■ データ {i+1}**")
            
            try:
                # ファイルの中身を読み込んでキャッシュ関数に渡す
                file_bytes = file.getvalue()
                x, y = load_data(file_bytes)
                
                # スムージングと正規化（これもキャッシュで爆速化）
                y_norm = process_signal(y, apply_smoothing, smooth_window, 2)
                y_shifted = y_norm + (i * offset)
                
                label_name = st.text_input("ラベル名", value=file.name, key=f"label_{i}")
                line_col_default = default_colors[i % len(default_colors)]
                
                # ピークの検出処理
                all_peaks = []
                if show_peaks:
                    peaks_auto, _ = find_peaks(y_norm, prominence=prominence_val)
                    
                    manual_peaks_str = st.text_input(
                        "手動でピークを追加 (スペースかカンマで複数入力OK)", 
                        placeholder="例: 28.5  30.1  35.2",
                        key=f"manual_{i}"
                    )
                    manual_indices = []
                    if manual_peaks_str:
                        for val_str in re.split(r'[,\s、，]+', manual_peaks_str):
                            if not val_str:
                                continue
                            try:
                                val = float(val_str.strip())
                                idx = np.abs(x - val).argmin()
                                manual_indices.append(idx)
                            except ValueError:
                                pass
                                
                    all_peaks = np.unique(np.concatenate((peaks_auto, manual_indices)).astype(int))
                
                num_peaks = len(all_peaks)

                # 一括設定UI
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    line_color = st.color_picker("波形(線)の色", line_col_default, key=f"line_color_{i}")
                with c2:
                    st.color_picker("一括マーカー色", value=line_col_default, key=f"bulk_col_{i}", 
                                    on_change=sync_bulk_color, args=(i, num_peaks))
                with c3:
                    st.number_input("一括サイズ", min_value=1, max_value=20, value=8, key=f"bulk_siz_{i}", 
                                    on_change=sync_bulk_size, args=(i, num_peaks))
                with c4:
                    st.number_input("一括高さ(浮き)", min_value=-0.5, max_value=1.5, value=0.05, step=0.01, key=f"bulk_off_{i}", 
                                    on_change=sync_bulk_offset, args=(i, num_peaks))
                
                ax.plot(x, y_shifted, label=label_name, color=line_color)
                
                final_peaks_data = [] # CSV出力用のデータ保管リスト
                
                if show_peaks:
                    with st.expander(f"ピークの編集 (計 {num_peaks} 個)"):
                        st.caption("左から: [表示/非表示] | [2θ値 (左右)] | [色] | [サイズ] | [高さ (上下)]")

                        for j, p in enumerate(all_peaks):
                            col_chk, col_val, col_col, col_siz, col_off = st.columns([0.5, 2, 1, 1, 1])
                            
                            with col_chk:
                                keep_peak = st.checkbox("✓", value=True, key=f"keep_{i}_{j}", label_visibility="collapsed")
                            with col_val:
                                edited_x = st.number_input("2θ", value=float(x[p]), step=0.05, format="%.2f", key=f"val_{i}_{j}", label_visibility="collapsed")
                            with col_col:
                                init_col = st.session_state.get(f"bulk_col_{i}", line_col_default)
                                ind_color = st.color_picker("Col", value=init_col, key=f"col_{i}_{j}", label_visibility="collapsed")
                            with col_siz:
                                init_siz = st.session_state.get(f"bulk_siz_{i}", 8)
                                ind_size = st.number_input("Size", min_value=1, max_value=20, value=init_siz, key=f"siz_{i}_{j}", label_visibility="collapsed")
                            with col_off:
                                init_off = st.session_state.get(f"bulk_off_{i}", 0.05)
                                ind_off = st.number_input("Offset", value=init_off, step=0.01, key=f"off_{i}_{j}", label_visibility="collapsed")
                                
                            if keep_peak:
                                target_idx = np.abs(x - edited_x).argmin()
                                ax.plot(x[target_idx], y_shifted[target_idx] + ind_off, "v", color=ind_color, markersize=ind_size)
                                
                                # 表示されているピークのデータをリストに追加
                                final_peaks_data.append({
                                    "2Theta": round(x[target_idx], 3),
                                    "Intensity(Norm)": round(y_norm[target_idx], 4)
                                })
                
                # --- 【アップデート3】CSVダウンロードボタンの追加 ---
                if final_peaks_data:
                    df_peaks = pd.DataFrame(final_peaks_data)
                    csv = df_peaks.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 {label_name} のピークリスト (CSV) をダウンロード",
                        data=csv,
                        file_name=f"{label_name}_peaks.csv",
                        mime="text/csv",
                        key=f"dl_{i}"
                    )
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"{file.name} の読み込みに失敗しました。ファイル形式を確認してください。")

    # --- 7. 軸の調整と装飾 ---
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max) 
    
    ax.set_xlabel(r"2$\theta$ [$^\circ$]", fontsize=12)
    ax.set_ylabel("Normalized Intensity [a.u.]", fontsize=12)
    ax.set_yticks([]) 
    
    if show_legend:
        if legend_loc == "枠外 (右側)":
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)
        else:
            loc_str = legend_loc.split(" ")[0] + " " + legend_loc.split(" ")[1]
            ax.legend(loc=loc_str, frameon=False, fontsize=10)
            
    plt.tight_layout()

    # --- 8. アプリ上に表示 ---
    st.pyplot(fig)