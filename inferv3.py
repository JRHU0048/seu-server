# -*- coding: utf-8 -*-
# 这一版代码已经很好了，三张图拼成一张大图
import os
from pathlib import Path
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

# =====================================================
# 配置
# =====================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

INPUT_FOLDER = "/home/huyanhan/data/CUB_200_2011/images/train/003.Sooty_Albatross"
OUTPUT_FOLDER = "/home/huyanhan/data/CUB_200_2011/qwen_output/train/003.Sooty_Albatross"
MODEL_ID = "./Qwen-Image-Edit"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DTYPE = torch.bfloat16
STEPS = 28
CFG = 3.5
SEED = 42
SIZE = 384

NEG_PROMPT = """
collage, split image, multiple birds, duplicated bird,
three birds, blurry, low quality, deformed, cartoon
"""

# =====================================================
# 三视角 Prompt
# =====================================================
VIEW_PROMPTS = [
{
"name":"水平60度侧视",
"prompt":"""
The input contains three reference images of the SAME bird.

Use them only as references.

Generate ONE single bird only.

Keep same species, feather color, beak shape.

Rotate bird horizontally by 60 degrees.

Natural wildlife photo.

Do not generate collage.
Do not generate multiple birds.
"""
},

{
"name":"水平30度侧视",
"prompt":"""
The input contains three reference images of the SAME bird.

Use them only as references.

Generate ONE single bird only.

Rotate bird horizontally by 30 degrees.

Realistic bird photo.

Do not generate collage.
"""
},

{
"name":"斜45度俯视",
"prompt":"""
The input contains three reference images of the SAME bird.

Use them only as references.

Generate ONE single bird only.

Generate 45 degree top-down view.

Show back feathers and body.

Do not generate collage.
"""
}
]

# =====================================================
# 加载模型
# =====================================================
print("Loading model...")

pipe = QwenImageEditPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="balanced",
    max_memory={
        0:"46GiB",
        1:"46GiB"
    }
)

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

print("Loaded.")

# =====================================================
# 工具函数
# =====================================================
def get_all_images(folder):
    fs = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg",".png",".jpeg")):
            fs.append(os.path.join(folder,f))
    fs.sort()
    return fs


def merge_3_images_grid(img1,img2,img3,size=384):
    im1 = Image.open(img1).convert("RGB").resize((size,size))
    im2 = Image.open(img2).convert("RGB").resize((size,size))
    im3 = Image.open(img3).convert("RGB").resize((size,size))

    canvas = Image.new("RGB",(size*2,size*2),(255,255,255))

    canvas.paste(im1,(0,0))
    canvas.paste(im2,(size,0))
    canvas.paste(im3,(0,size))

    return canvas


# =====================================================
# 主逻辑
# =====================================================
def run():

    images = get_all_images(INPUT_FOLDER)
    total = len(images)

    print("图片数量:",total)

    for i in range(total):

        main_name = Path(images[i]).stem

        ref1 = images[i]
        ref2 = images[(i+1)%total]
        ref3 = images[(i+2)%total]

        print("\n生成:",main_name)

        input_img = merge_3_images_grid(ref1,ref2,ref3,SIZE)

        for vp in VIEW_PROMPTS:

            print(" ->",vp["name"])

            with torch.inference_mode():

                out = pipe(
                    image=input_img,
                    prompt=vp["prompt"],
                    negative_prompt=NEG_PROMPT,
                    true_cfg_scale=CFG,
                    num_inference_steps=STEPS,
                    generator=torch.manual_seed(SEED)
                ).images[0]

            save_path = os.path.join(
                OUTPUT_FOLDER,
                f"{main_name}_{vp['name']}.png"
            )

            out.save(save_path)
            print("Saved:",save_path)

            torch.cuda.empty_cache()


if __name__ == "__main__":
    run()