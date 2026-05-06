import os
from PIL import Image

# =====================================================
# 工具函数
# =====================================================
def get_all_images(folder):
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg",".jpeg",".png")):
            files.append(os.path.join(folder,f))
    files.sort()
    return files

def load_image(img_path, size=384):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size))
    return img

def is_skipped_class(cls_name):
    # 提取文件夹名开头的数字，例如 "005." → 5
    if "." in cls_name:
        num_part = cls_name.split(".")[0]
        try:
            class_num = int(num_part)
            return 1 <= class_num <= 53  # 跳过
        except:
            pass
    return False