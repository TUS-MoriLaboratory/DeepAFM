# src/utils/common.py

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

#データ変換
def txt_path_to_array(file_path):
    #画像の読み込み
    if file_path.endswith('.txt'):
        heights_array = []
        with open(file_path) as f:
            for line in f:
                line_list = [float(i) for i in line.split(',')]
                heights_array.append(line_list)
    
        return np.array(heights_array)

    elif file_path.endswith('.csv'):
        heights_array = np.loadtxt(file_path, delimiter=',')
        return np.array(heights_array)

    elif file_path.endswith('.npy'):
        heights_array = np.load(file_path)
        return heights_array

    elif file_path.endswith('.tsv'):
        heights_array = []
        with open(file_path) as f:
            for line in f:
                line_list = [float(i) for i in line.split()]
                heights_array.append(line_list)

        return np.array(heights_array)

def heights_max_min_scaling(heights_array):
    #リスト形式のheightsをまとめた二次元リスト
    #numpy配列に変換
    np_array = np.array(heights_array)

    #max-min スケーリング
    max_height = np.max(np_array)
    min_height = np.min(np_array)

    scaled_array = (np_array - min_height) / (max_height - min_height)

    return scaled_array, max_height, min_height

def inverse_scaling(scaled_val, max, min):
    #max, min: 最大最小値
    #scaled_val: max-min scaling後のデータ
    
    ori_data = scaled_val * (max - min) + min

    return ori_data


#データの可視化
def color_afmhot(value, min_val, max_val):
    #assert min_val <= value <= max_val, "Value must be between min_val and max_val"

    v = (value - min_val) / (max_val - min_val)

    if v < 0.625:
        R = np.floor(v * 256 / 0.625)
        G = np.floor(v * 256)
        return (int(R), int(G), 0)
    
    elif v < 0.75:
        u = (v - 0.625)
        if u < 0.0:
            return (255, 159, 0)
        elif u >= 0.125:
            return (255, 191, 0)
        G = np.floor(v * 256)
        return (255, int(G), 0)
    
    else:
        u = (v - 0.75)
        if u < 0.0:
            return (255, 191, 0)
        elif u >= 0.25:
            return (255, 255, 255)
        B = np.floor(u * 256 / 0.25)
        G = np.floor(v * 256)
        return (255, int(G), int(B))

# 2次元numpy配列に color_afmhot を適用する関数
def apply_colormap_to_array(heights_array, min_val, max_val):
    # 出力は (height, width, 3) となる RGB配列
    height, width = heights_array.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

    # それぞれの値に対して color_afmhot 関数を適用
    for i in range(height):
        for j in range(width):
            rgb_array[i, j] = color_afmhot(heights_array[i, j], min_val, max_val)
    
    return rgb_array

def resize_to_nxn(array, n):
    """
    入力のnumpy配列をn*nサイズに変更して出力

    Args:
        array: 変更する配列
        n: 出力したい画像サイズ
    """
    
    # 元の配列の形状を取得
    rows, cols = array.shape
    
    # n×nにするための開始位置を計算（元の配列の中心が n×n の中心になるように）
    row_start = max((n - rows) // 2, 0)
    col_start = max((n - cols) // 2, 0)
    
    # ゼロ、または元の配列の最小値で埋める場合は np.full を使う
    padded_array = np.full((n, n), fill_value=np.min(array), dtype=array.dtype)
    
    # 元の配列が n×n を超えている場合の切り取り（中心部分を切り出す）
    insert_row_start = max((rows - n) // 2, 0)
    insert_col_start = max((cols - n) // 2, 0)
    
    # 元の配列の切り出し（超えた部分はカット）
    resized_array = array[insert_row_start:insert_row_start + n,
                          insert_col_start:insert_col_start + n]
    
    # 作成した n×n 配列にリサイズした配列を挿入
    padded_array[row_start:row_start + resized_array.shape[0],
                 col_start:col_start + resized_array.shape[1]] = resized_array
    
    return padded_array

def array_to_image(
        heights_array, 
        ref_array=None, 
        image_size=None, 
        color_bar=False, 
        index=None, 
        axis='col', 
        save_path=None, 
        origin='upper',
        dpi=800, 
        unit='nm' # 'nm' or 'angstrom'
        ):

    # image_size: n(int)
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Rectangle

    # 単位変換
    heights_array = heights_array / (10.0 if unit=='nm' else 1.0)
    ref_array = ref_array / (10.0 if unit=='nm' else 1.0) if ref_array is not None else ref_array

    if image_size is not None:
        heights_array = resize_to_nxn(heights_array, image_size)
        ref_array = resize_to_nxn(ref_array, image_size) if ref_array is not None else ref_array
    
    # heights_array の最小・最大値で色付け
    min_val_heights = np.min(heights_array)
    max_val_heights = np.max(heights_array)
    rgb_heights = apply_colormap_to_array(heights_array, min_val_heights, max_val_heights)

    # ref_array がある場合は、ref の最小・最大値で色付け
    if ref_array is not None:
        min_val_ref = np.min(ref_array)
        max_val_ref = np.max(ref_array)
        rgb_ref = apply_colormap_to_array(ref_array, min_val_ref, max_val_ref)

    #colやrowを点線で囲んで強調
    if index:
        if axis=='row':
            total_row_nums = heights_array.shape[0]

            rect = Rectangle(
                (index, 0),               # 左下(x, y)
                1,                        # 幅
                total_row_nums,           # 高さ
                linewidth=1.5,
                edgecolor='white',
                linestyle='--',
                facecolor='none'
            )
            
        elif axis=='col':
            total_col_nums = heights_array.shape[1]

            rect = Rectangle(
                (0, index),               # 左下(x, y)
                total_col_nums,           # 幅
                1,                        # 高さ
                linewidth=1.5,
                edgecolor='white',
                linestyle='--',
                facecolor='none'
            )
    
    # カラーバー付きの場合、各配列ごとにカスタムカラーマップを作成
    if color_bar:
        n = 256  # カラーマップの段階数

        # heights 用のカスタムカラーマップ
        colors_heights = []
        for i in range(n):
            val = min_val_heights + (max_val_heights - min_val_heights) * i / (n - 1)
            color = color_afmhot(val, min_val_heights, max_val_heights)
            colors_heights.append(np.array(color) / 255.0)
        cmap_heights = mpl.colors.LinearSegmentedColormap.from_list("afmhot_custom_heights", colors_heights, N=n)
        norm_heights = mpl.colors.Normalize(vmin=min_val_heights, vmax=max_val_heights)
        
        # ref 用のカスタムカラーマップ（ref_array がある場合）
        if ref_array is not None:
            colors_ref = []
            for i in range(n):
                val = min_val_ref + (max_val_ref - min_val_ref) * i / (n - 1)
                color = color_afmhot(val, min_val_ref, max_val_ref)
                colors_ref.append(np.array(color) / 255.0)
            cmap_ref = mpl.colors.LinearSegmentedColormap.from_list("afmhot_custom_ref", colors_ref, N=n)
            norm_ref = mpl.colors.Normalize(vmin=min_val_ref, vmax=max_val_ref)
        else:
            cmap_ref = None
            norm_ref = None
    else:
        cmap_heights = 'viridis'
        norm_heights = None
        cmap_ref = 'viridis'
        norm_ref = None

    # 可視化：ref_array がある場合は 2 つのサブプロットを横に並べる
    if ref_array is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # 左側：ref_array の画像表示
        im1 = ax1.imshow(rgb_ref, cmap=cmap_ref, norm=norm_ref, origin=origin)
        ax1.axis('off')
        #ax1.set_title("Reference")
        if color_bar:
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar1 = fig.colorbar(
                im1, 
                cax=cax1, 
                orientation='vertical', 
                label='height(Å)' if unit=='angstrom' else 'height(nm)'
            )

            # カラーバーの詳細なフォントと文字サイズ設定
            fp1 = FontProperties(family='Times New Roman', size=20)
            for tl in cbar1.ax.yaxis.get_ticklabels():
                tl.set_fontproperties(fp1)
            
            cbar1.ax.yaxis.label.set_fontproperties(fp1)

            cbar1.set_label('height(Å)', fontproperties=fp1) if unit=='angstrom' else cbar1.set_label('height(nm)', fontproperties=fp1)
            cbar1.ax.tick_params(labelsize=20)  # 目盛りの数値
            cbar1.ax.yaxis.label.set_fontsize(20)  # ラベル
            cbar1.set_label('height(Å)', fontsize=20) if unit=='angstrom' else cbar1.set_label('height(nm)', fontsize=20)

        if index: # colやrowを強調する場合 
            ax1.add_patch(rect)
        
        # 右側：heights_array の画像表示
        im2 = ax2.imshow(rgb_heights, cmap=cmap_heights, norm=norm_heights, origin=origin)
        ax2.axis('off')
        #ax2.set_title("Heights")
        if color_bar:
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            cbar2 = fig.colorbar(
                im2, 
                cax=cax2, 
                orientation='vertical', 
                label='height (Å)'
                )

            fp2 = FontProperties(family='Times New Roman', size=20)
            for tl in cbar2.ax.yaxis.get_ticklabels():
                tl.set_fontproperties(fp2)
                    
            cbar2.ax.yaxis.label.set_fontproperties(fp2)
            cbar2.set_label('height(Å)', fontproperties=fp2) if unit=='angstrom' else cbar2.set_label('height(nm)', fontproperties=fp2)
            cbar2.ax.tick_params(labelsize=20)  # 目盛りの数値
            cbar2.ax.yaxis.label.set_fontsize(20)  # ラベル
            cbar2.set_label('height(Å)', fontsize=20) if unit=='angstrom' else cbar2.set_label('height(nm)', fontsize=20)

        if index: # colやrowを強調する場合 
            ax2.add_patch(rect)
    
    else:
        # ref_array がない場合は heights_array のみ表示
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(rgb_heights, cmap=cmap_heights, norm=norm_heights, origin=origin)
        ax.axis('off')
        if color_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(
                im, 
                cax=cax, 
                orientation='vertical', 
                label='height (Å)'
            )

            fp = FontProperties(family='Times New Roman', size=20)
            for tl in cbar.ax.yaxis.get_ticklabels():
                tl.set_fontproperties(fp)
            cbar.ax.yaxis.label.set_fontproperties(fp)

            cbar.set_label('height(Å)', fontproperties=fp) if unit=='angstrom' else cbar.set_label('height(nm)', fontproperties=fp)
            cbar.ax.tick_params(labelsize=20)  # 目盛りの数値
            cbar.ax.yaxis.label.set_fontsize(20)  # ラベル
            cbar.set_label('height(Å)', fontsize=20) if unit=='angstrom' else cbar.set_label('height(nm)', fontsize=20)

        if index: # colやrowを強調する場合 
            ax.add_patch(rect)
    
    # 画像の保存
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    else:
        plt.show()

def create_colormap_afmhot(min_val, max_val, color_func=color_afmhot, n=256, cmap_name="custom"):
    colors = []
    for i in range(n):
        val = min_val + (max_val - min_val) * i / (n - 1)
        color = color_func(val, min_val, max_val)
        colors.append(np.array(color) / 255.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    return cmap, norm

def add_colorbar(cmap, norm, ax, position='right', label=None, labelsize=16):
    """
    任意の位置にカラーバーを追加する共通関数。
    
    Args:
        cmap (mpl.colors.Colormap): 使用するカラーマップ
        vmin, vmax (float): データの最小値・最大値
        ax (matplotlib.axes.Axes): カラーバーを付けたい Axes
        position (str): 'right', 'left', 'top', 'bottom'
        label (str): カラーバーのラベル
        labelsize (int): ラベルおよび目盛りのフォントサイズ
    """
    divider = make_axes_locatable(ax)
    orientation = 'vertical' if position in ['right', 'left'] else 'horizontal'
    pad = 0.05 if orientation == 'vertical' else 0.10

    # カラーバー用の Axes を作成
    cax = divider.append_axes(position, size="5%", pad=pad)

    # カラーバー作成
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation=orientation)

    # ラベル設定
    if label:
        rotation = 90 if orientation == 'vertical' else 0
        labelpad = 15 if orientation == 'vertical' else 5
        cbar.set_label(label, labelpad=labelpad, rotation=rotation)

    # 目盛り設定
    cbar.ax.tick_params(labelsize=labelsize)
    return cbar


def array_difference_heatmap(
        ref_array, 
        array_list, 
        save_path=None, 
        text_list=None, 
        periodic_shift=None, 
        color_bar_position='right'
    ):
    """
    シンプルな差分ヒートマップ
    
    Args:
        ref_array: 参照配列
        array_list: 比較する配列のリスト
        title: タイトル
        save_path: 保存パス (オプション)
    """
    n = 256  # カラーマップの段階数

    # フォントを設定
    mpl.rcParams['font.family'] = 'serif'          # ファミリは serif にして
    mpl.rcParams['font.serif'] = ['Times New Roman']   # serif の候補を Times-Roman のみにする
    mpl.rcParams['font.size'] = 16                 # フォントサイズを 14 にする

    # 入力配列の形状が同じであることを
    for array in array_list:
        if ref_array.shape != array.shape:
            raise ValueError("Input arrays must have the same shape")

    N=len(array_list)
    # 絶対差分と最大値を計算
    abs_difference_list = [np.abs(ref_array - array) for array in array_list]
    max_abs_diff = np.max([np.max(diff) for diff in abs_difference_list])

    # 複数のaxesを作成
    if N > 1:
        if color_bar_position in ['right', 'left']:
            fig, ax = plt.subplots(1, 1+2*N, figsize=((1+2*N)*4, 6))
        else:
            fig, ax = plt.subplots(1, 1+2*N, figsize=((1+2*N)*4, 6))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    plt.subplots_adjust(wspace=0.35) 

    # 参照配列の可視化
    # ref_arrayの最小・最大値で色付け
    min_val_ref = np.min(ref_array)
    max_val_ref = np.max(ref_array)
    rgb_ref = apply_colormap_to_array(ref_array, min_val_ref, max_val_ref)

    # refのカラーマップ
    cmap_ref, norm_ref = create_colormap_afmhot(min_val_ref, max_val_ref, color_func=color_afmhot)

    if periodic_shift:
        rgb_ref = np.roll(rgb_ref, shift=periodic_shift, axis=(0, 1))

    im1 = ax[0].imshow(rgb_ref, cmap=cmap_ref, norm=norm_ref, origin='lower')
    ax[0].axis('off')

    add_colorbar(cmap_ref, norm_ref, ax[0], position=color_bar_position, label='height(Å)')

    ax[0].imshow(rgb_ref, origin='lower')

    # 各配列を描画
    for i, abs_diff in enumerate(abs_difference_list):
        heights_array = array_list[i]

        # 周期的シフト
        if periodic_shift:
            heights_array = np.roll(heights_array, shift=periodic_shift, axis=(0, 1))

        # 比較配列の可視化
         # heights_array の最小・最大値で色付け
        min_val_heights = np.min(heights_array)
        max_val_heights = np.max(heights_array)
        rgb_heights = apply_colormap_to_array(heights_array, min_val_heights, max_val_heights)

        # heightsのカラーマップ作成
        cmap_heights, norm_heights = create_colormap_afmhot(min_val_heights, max_val_heights, color_func=color_afmhot)
        im1 = ax[2*i+1].imshow(rgb_heights, cmap=cmap_heights, norm=norm_heights, origin='lower')
        ax[2*i+1].axis('off')
        add_colorbar(cmap_heights, norm_heights, ax[2*i+1], position=color_bar_position, label='height(Å)')

        # 絶対差分の可視化
        # カラーマップ
        im2 = ax[2*i+2].imshow(abs_diff, cmap='RdBu_r', origin='lower',
                    vmin=0, vmax=max_abs_diff)
        ax[2*i+2].axis('off')
        cmap_diff = plt.get_cmap('RdBu_r')
        norm_diff = mpl.colors.Normalize(vmin=0, vmax=max_abs_diff)
        add_colorbar(cmap_diff, norm_diff, ax[2*i+2], position=color_bar_position, label='Absolute Difference (Å)')

    # 画像の上にテキストを追加
    if text_list:
        for i in range(len(text_list)):
            text_ = text_list[i]
            if text_ is not None:
                h, w = heights_array.shape  # 画像サイズ
                idx = 0 if i==0 else 2*(i-1)+1
                ax[idx].text(
                    w // 2, 2, # h - 2
                    text_,
                    color='white',
                    fontsize=20,
                    ha='center',
                    va='bottom'
            )
                
    # 画像の保存
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=800)
        plt.close(fig)
    else:
        plt.show()