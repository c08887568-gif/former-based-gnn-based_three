## -------轨迹裁剪函数 -------
def crop_trace(arr, max_len,drop_rate):
    rows, cols = arr.shape
    cropped_blocks = []

    for i in range(0, rows, max_len):
        end_index = min(i + max_len, rows)
        cropped_block = arr[i:end_index, :]
        # 添加条件，仅保留长度大于等于 drop_rate * max_len 的子块
        if len(cropped_block) >= drop_rate * max_len:
            cropped_blocks.append(cropped_block)
    return cropped_blocks
## -------邻接矩阵裁剪函数 -------
def crop_adj(arr, max_len,drop_rate):
    rows, cols = arr.shape
    cropped_blocks = []

    for i in range(0, rows, max_len):
        end_index = min(i + max_len, rows)
        cropped_block = arr[i:end_index, i:end_index]
        # 添加条件，仅保留长度大于等于 drop_rate * max_len 的子块
        if len(cropped_block) >= drop_rate * max_len:
            cropped_blocks.append(cropped_block)
    return cropped_blocks
