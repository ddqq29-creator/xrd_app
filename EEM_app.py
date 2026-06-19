# python -m streamlit run EEM_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import io

# --- どんなJASCOデータでも読み込む最強のパーサー ---
@st.cache_data
def load_jasco_eem(file_bytes):
    try:
        content = file_bytes.decode('shift_jis')
    except UnicodeDecodeError:
        content = file_bytes.decode('utf-8', errors='ignore')
        
    lines = content.split('\n')

    xydata_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "XYDATA":
            xydata_idx = i
            break

    if xydata_idx == -1:
        raise ValueError("no data")

    ex_line = lines[xydata_idx + 1].strip('\n\r').split('\t')
    ex_waves = []
    for x in ex_line:
        if x.strip() != '':
            try:
                ex_waves.append(float(x))
            except ValueError:
                pass
    ex_waves = np.array(ex_waves)

    em_waves = []
    z_matrix = []
    
    for line in lines[xydata_idx + 2:]:
        line_clean = line.strip('\n\r')
        if not line_clean:
            continue
            
        parts = line_clean.split('\t')
        if len(parts) < 2:
            break
            
        try:
            em = float(parts[0])
            z_row = []
            for x in parts[1:]:
                val = x.strip()
                if val == '':
                    z_row.append(0.0)
                else:
                    try:
                        z_row.append(float(val))
                    except ValueError:
                        z_row.append(0.0)
            
            if len(z_row) < len(ex_waves):
                z_row.extend([0.0] * (len(ex_waves) - len(z_row)))
            elif len(z_row) > len(ex_waves):
                z_row = z_row[:len(ex_waves)]
                
            em_waves.append(em)
            z_matrix.append(z_row)
            
        except ValueError:
            break

    if len(z_matrix) == 0:
        raise ValueError("データ部分が読み込めませんでした。")

    return np.array(ex_waves), np.array(em_waves), np.array(z_matrix)

# --- Jupyter Notebookの「論文用」軸設定を再現する関数 ---
def apply_publication_style(ax, title, xlabel, ylabel, xlim=None, ylim=None):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
        
    ax.tick_params(axis='both', which='major', direction='in',
                   top=True, bottom=True, right=True, left=True,
                   length=5, width=1.0, labelsize=10)
    
    ax.tick_params(axis='both', which='minor', direction='in',
                   top=True, bottom=True, right=True, left=False,
                   length=3, width=0.8)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


# --- 1. アプリの基本設定 ---
st.set_page_config(page_title="EEM Publication Plotter", layout="wide")
st.title("for EEM")

# --- 2. サイドバー（設定メニュー） ---
with st.sidebar:
    st.header("グラフ描画設定")
    cmap_choice = st.selectbox("カラーマップ (全体マップ用)", ['turbo', 'jet', 'viridis', 'plasma', 'inferno'])
    
    cut_scattering = st.checkbox("Z軸(強度)の上限をカット", value=True)
    if cut_scattering:
        z_max_limit = st.number_input("Z軸の最大値", min_value=10, max_value=20000, value=1000, step=50)
    else:
        z_max_limit = None
        
    st.markdown("---")
    st.header("軸の範囲")
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        # 初期値を230に変更
        ex_min = st.number_input("Ex 最小値", value=230, step=10)
    with col_ex2:
        # 初期値を500に変更
        ex_max = st.number_input("Ex 最大値", value=500, step=10)
        
    col_em1, col_em2 = st.columns(2)
    with col_em1:
        # 初期値を500に変更
        em_min = st.number_input("Em 最小値", value=500, step=10)
    with col_em2:
        # 初期値を800に変更
        em_max = st.number_input("Em 最大値", value=800, step=10)

    st.markdown("---")
    
    st.header("表示するグラフの選択 (複数可)")
    selected_plots = st.multiselect(
        "描画したいグラフを選んでください",
        [
            "2D 等高線マップ (Contour)", 
            "励起(Ex)波長を固定 ➡️ 蛍光スペクトル", 
            "蛍光(Em)波長を固定 ➡️ 励起スペクトル"
        ],
        default=["2D 等高線マップ (Contour)"]
    )
    
    target_ex, target_em = None, None
    if "励起(Ex)波長を固定 ➡️ 蛍光スペクトル" in selected_plots:
        target_ex = st.number_input("ターゲット 励起(Ex)波長 (nm)", value=300.0, step=5.0)
    if "蛍光(Em)波長を固定 ➡️ 励起スペクトル" in selected_plots:
        target_em = st.number_input("ターゲット 蛍光(Em)波長 (nm)", value=300.0, step=5.0)

# --- 3. メイン画面（データ表示） ---
uploaded_files = st.file_uploader("JASCO形式の3D蛍光スペクトルデータ(.txt)をアップロード", accept_multiple_files=True)

if uploaded_files:
    num_files = len(uploaded_files)
    
    # 全データを一括で先に読み込んでおく（処理の高速化）
    data_dict = {}
    for file in uploaded_files:
        try:
            Ex, Em, Z = load_jasco_eem(file.getvalue())
            if cut_scattering and z_max_limit is not None:
                Z_plot = np.clip(Z, a_min=None, a_max=z_max_limit)
            else:
                Z_plot = Z
            data_dict[file.name] = (Ex, Em, Z, Z_plot)
        except Exception as e:
            st.error(f"{file.name} の読み込みに失敗しました。詳細: {e}")

    st.markdown("---")

    # ----------------------------------------------------
    # ① 全体マップの横並び描画
    # ----------------------------------------------------
    if "2D 等高線マップ (Contour)" in selected_plots and data_dict:
        st.markdown("2D 等高線マップの比較")
        cols = st.columns(num_files) # アップロードされたファイル数に応じて画面を縦に分割
        
        for i, (file_name, (Ex, Em, Z, Z_plot)) in enumerate(data_dict.items()):
            with cols[i]:
                # 横並びで文字が潰れないように少し小さめのfigsizeに設定
                fig_map, ax_map = plt.subplots(figsize=(5, 4.5))
                contour = ax_map.contourf(Ex, Em, Z_plot, levels=100, cmap=cmap_choice)
                cbar = fig_map.colorbar(contour, ax=ax_map)
                cbar.set_label('Intensity (a.u.)', fontsize=11)
                cbar.ax.tick_params(direction='in')

                apply_publication_style(
                    ax_map, 
                    title=f"EEM - {file_name}", 
                    xlabel="Excitation Wavelength (nm)", 
                    ylabel="Emission Wavelength (nm)",
                    xlim=(ex_min, ex_max),
                    ylim=(em_min, em_max)
                )
                fig_map.tight_layout()
                st.pyplot(fig_map, use_container_width=True)
                
                buf_map = io.BytesIO()
                fig_map.savefig(buf_map, format="png", dpi=300, bbox_inches="tight")
                buf_map.seek(0)
                st.download_button(f"📥 保存 (PNG)", data=buf_map, file_name=f"{file_name}_map.png", mime="image/png", key=f"dl_map_{i}")

        st.markdown("---")

    # ----------------------------------------------------
    # ② Ex固定（蛍光スペクトル）の横並び描画
    # ----------------------------------------------------
    if "励起(Ex)波長を固定 ➡️ 蛍光スペクトル" in selected_plots and data_dict:
        st.markdown(f"蛍光スペクトル (Ex = {target_ex} nm) の比較")
        cols = st.columns(num_files)
        
        for i, (file_name, (Ex, Em, Z, Z_plot)) in enumerate(data_dict.items()):
            with cols[i]:
                fig_ex, ax_ex = plt.subplots(figsize=(5, 4))
                
                idx_ex = (np.abs(Ex - target_ex)).argmin()
                actual_ex = Ex[idx_ex]
                
                ax_ex.plot(Em, Z[:, idx_ex], color='blue', linewidth=1.5)
                
                apply_publication_style(
                    ax_ex, 
                    title=f"{file_name}\n(Ex = {actual_ex:.1f} nm)", 
                    xlabel="Emission Wavelength (nm)", 
                    ylabel="Intensity (a.u.)",
                    xlim=(em_min, em_max),
                    ylim=None
                )
                fig_ex.tight_layout()
                st.pyplot(fig_ex, use_container_width=True)
                
                buf_ex = io.BytesIO()
                fig_ex.savefig(buf_ex, format="png", dpi=300, bbox_inches="tight")
                buf_ex.seek(0)
                st.download_button(f"保存 (PNG)", data=buf_ex, file_name=f"{file_name}_Ex_{actual_ex:.1f}.png", mime="image/png", key=f"dl_ex_{i}")

        st.markdown("---")

    # ----------------------------------------------------
    # ③ Em固定（励起スペクトル）の横並び描画
    # ----------------------------------------------------
    if "蛍光(Em)波長を固定 ➡️ 励起スペクトル" in selected_plots and data_dict:
        st.markdown(f"励起スペクトル (Em = {target_em} nm) の比較")
        cols = st.columns(num_files)
        
        for i, (file_name, (Ex, Em, Z, Z_plot)) in enumerate(data_dict.items()):
            with cols[i]:
                fig_em, ax_em = plt.subplots(figsize=(5, 4))
                
                idx_em = (np.abs(Em - target_em)).argmin()
                actual_em = Em[idx_em]
                
                ax_em.plot(Ex, Z[idx_em, :], color='red', linewidth=1.5)
                
                apply_publication_style(
                    ax_em, 
                    title=f"{file_name}\n(Em = {actual_em:.1f} nm)", 
                    xlabel="Excitation Wavelength (nm)", 
                    ylabel="Intensity (a.u.)",
                    xlim=(ex_min, ex_max),
                    ylim=None
                )
                fig_em.tight_layout()
                st.pyplot(fig_em, use_container_width=True)
                
                buf_em = io.BytesIO()
                fig_em.savefig(buf_em, format="png", dpi=300, bbox_inches="tight")
                buf_em.seek(0)
                st.download_button(f"保存 (PNG)", data=buf_em, file_name=f"{file_name}_Em_{actual_em:.1f}.png", mime="image/png", key=f"dl_em_{i}")

        st.markdown("---")
else:
    st.info("データをアップロード")