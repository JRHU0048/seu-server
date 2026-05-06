"""
共享工具函数
"""

import os
from PIL import Image


def get_all_images(folder):
    """获取文件夹下所有图片的排序列表"""
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(os.path.join(folder, f))
    files.sort()
    return files


def load_image(img_path, size=384):
    """加载并缩放到统一尺寸"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size))
    return img


def is_skipped_class(cls_name, max_class=53):
    """
    根据文件夹名前缀数字跳过已处理过的类别。
    如 "005." → 5，若 1 ≤ num ≤ max_class 则跳过。
    """
    if "." not in cls_name:
        return False
    try:
        class_num = int(cls_name.split(".")[0])
        return 1 <= class_num <= max_class
    except ValueError:
        return False
