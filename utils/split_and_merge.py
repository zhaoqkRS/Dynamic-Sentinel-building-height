import numpy as np

# def split_to_tiles(data, window_size=128, overlap=2):
#     """
#     优化版分块函数（修复边缘溢出问题）
#     """
#     channels, height, width = data.shape
#     stride = window_size - overlap
#     tiles = []

#     # 确保最后一个tile起始位置不会导致越界
#     y_starts = []
#     y = 0
#     while y + window_size <= height:
#         y_starts.append(y)
#         y += stride
#     # 处理剩余部分（确保至少能覆盖到图像末端）
#     if y_starts[-1] + window_size < height:
#         y_starts.append(height - window_size)

#     x_starts = []
#     x = 0
#     while x + window_size <= width:
#         x_starts.append(x)
#         x += stride
#     if x_starts[-1] + window_size < width:
#         x_starts.append(width - window_size)

#     # 生成tiles
#     for y_start in y_starts:
#         for x_start in x_starts:
#             y_end = y_start + window_size
#             x_end = x_start + window_size
#             tile = data[:, y_start:y_end, x_start:x_end]
#             tiles.append((tile, (y_start, x_start)))
    
#     return tiles

# def merge_tiles(tile_list, target_shape, window_size=512, scale_factor=4):
#     target_c, target_h, target_w = target_shape
#     merged = np.zeros(target_shape, dtype=np.float32)
#     counter = np.zeros(target_shape, dtype=np.float32)

#     for tile, (orig_y_start, orig_x_start) in tile_list:
#         y_start = orig_y_start * scale_factor
#         x_start = orig_x_start * scale_factor
#         y_end = min(y_start + window_size, target_h)  # 使用目标高度
#         x_end = min(x_start + window_size, target_w)  # 使用目标宽度

#         # 调整tile的有效区域
#         h = y_end - y_start
#         w = x_end - x_start
#         if h <= 0 or w <= 0:
#             continue

#         # 校验tile的有效区域是否匹配
#         if tile.shape[1] < h or tile.shape[2] < w:
#             tile = tile[:, :h, :w]  # 自动裁剪tile（更鲁棒的写法）

#         merged[:, y_start:y_end, x_start:x_end] += tile[:, :h, :w]
#         counter[:, y_start:y_end, x_start:x_end] += 1

#     return np.divide(merged, counter, where=counter != 0, out=np.zeros_like(merged))

import numpy as np

import numpy as np

def split_to_tiles(data, window_size=128, overlap=20):
    """
    修改后的分块函数，增加重叠区域并填充不足的tile
    """
    channels, height, width = data.shape
    stride = window_size - overlap
    tiles = []

    def get_starts(dim_size):
        starts = []
        start = 0
        while start + window_size <= dim_size:
            starts.append(start)
            start += stride
        if start < dim_size:
            starts.append(max(0, dim_size - window_size))
        return starts

    y_starts = get_starts(height)
    x_starts = get_starts(width)

    for y_start in y_starts:
        for x_start in x_starts:
            y_end = y_start + window_size
            x_end = x_start + window_size
            tile = data[:, y_start:y_end, x_start:x_end]
            
            # 处理边界填充
            pad_y = window_size - (min(y_end, height) - y_start)
            pad_x = window_size - (min(x_end, width) - x_start)
            if pad_y > 0 or pad_x > 0:
                tile = np.pad(tile, ((0,0), (0,pad_y), (0,pad_x)), mode='constant')
            
            tiles.append((tile, (y_start, x_start)))
    
    return tiles

def merge_tiles(tile_list, target_shape, window_size=512, scale_factor=4, crop=40):
    """
    修改后的合并函数，支持边缘感知的加权融合
    """
    target_c, target_h, target_w = target_shape
    merged = np.zeros(target_shape, dtype=np.float32)
    weight_sum = np.zeros(target_shape, dtype=np.float32)

    def create_window(size):
        hann = np.hanning(size)
        return np.outer(hann, hann)

    full_window = create_window(window_size)

    for tile, (orig_y_start, orig_x_start) in tile_list:
        y_start = orig_y_start * scale_factor
        x_start = orig_x_start * scale_factor
        y_end = y_start + window_size
        x_end = x_start + window_size

        # 调整越界
        y_end = min(y_end, target_h)
        x_end = min(x_end, target_w)

        # 确定y轴裁剪参数
        if y_start == 0:
            write_y_start = 0
            write_y_end = y_end - crop
            tile_y_slice = slice(0, window_size - crop)
        elif y_end == target_h:
            write_y_start = y_start + crop
            write_y_end = y_end
            tile_y_slice = slice(crop, window_size)
        else:
            write_y_start = y_start + crop
            write_y_end = y_end - crop
            tile_y_slice = slice(crop, window_size - crop)

        # 确定x轴裁剪参数
        if x_start == 0:
            write_x_start = 0
            write_x_end = x_end - crop
            tile_x_slice = slice(0, window_size - crop)
        elif x_end == target_w:
            write_x_start = x_start + crop
            write_x_end = x_end
            tile_x_slice = slice(crop, window_size)
        else:
            write_x_start = x_start + crop
            write_x_end = x_end - crop
            tile_x_slice = slice(crop, window_size - crop)

        # 调整写入区域边界
        write_y_start = max(write_y_start, 0)
        write_y_end = min(write_y_end, target_h)
        write_x_start = max(write_x_start, 0)
        write_x_end = min(write_x_end, target_w)

        # 计算有效区域
        eff_height = write_y_end - write_y_start
        eff_width = write_x_end - write_x_start
        if eff_height <= 0 or eff_width <= 0:
            continue

        # 调整tile切片
        tile_y_slice = slice(tile_y_slice.start, tile_y_slice.start + eff_height)
        tile_x_slice = slice(tile_x_slice.start, tile_x_slice.start + eff_width)

        # 裁剪tile并应用窗口
        cropped_tile = tile[:, tile_y_slice, tile_x_slice]
        window = full_window[tile_y_slice, tile_x_slice]

        # 加权叠加
        merged[:, write_y_start:write_y_end, write_x_start:write_x_end] += cropped_tile * window
        weight_sum[:, write_y_start:write_y_end, write_x_start:write_x_end] += window

    # 归一化处理
    merged = np.divide(merged, weight_sum, where=weight_sum!=0)
    return merged