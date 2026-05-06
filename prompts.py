"""
共享 Prompt 库
所有模型的负面提示词和视角变换 prompt 都集中在此。
"""

NEG_PROMPT = (
    "collage, multiple birds, split image, duplicated bird, "
    "blurry, low quality, deformed, malformed, bad anatomy, "
    "extra wings, extra legs, extra body parts, fused limbs"
)

# ===========================================================================
# 单图输入 (Qwen-Image-Edit)
# ===========================================================================

# 视角变换（v5 在用）
VIEW_PROMPTS_SINGLE = [
    {
        "name": "view_60",
        "prompt": (
            "You are given a single reference image of a bird.\n"
            "Generate ONLY ONE bird.\n"
            "Keep the bird's identity, feather details, beak shape, and texture "
            "completely consistent with the reference.\n"
            "Change the camera viewing angle to rotate horizontally 60 degrees "
            "to the left around the bird, rather than flat image rotation.\n"
            "Keep the bird's natural standing posture, realistic body proportions "
            "and anatomical structure unchanged.\n"
            "Do NOT generate multiple birds.\n"
            "Do NOT generate collage.\n"
            "Do NOT deform or distort the bird's body.\n"
        )
    },
]

# 备选 prompt：环境替换（单图）
PROMPT_BG_REPLACE = {
    "name": "bg_replace_natural",
    "prompt": (
        "You are given a single reference image of a bird.\n"
        "Keep the bird identity consistent: same species, feather pattern, "
        "beak shape, and overall appearance.\n"
        "Replace the original background with a realistic natural outdoor "
        "environment such as forest, grassland, lake shore, wetland, or tree branches.\n"
        "Adjust the bird pose, lighting, shadow, color tone, feather direction, "
        "body contact, and perspective when necessary so the bird naturally fits "
        "the new environment.\n"
        "The bird should interact realistically with the scene, "
        "as if photographed there originally.\n"
        "Generate ONLY ONE bird.\n"
        "Do NOT make the bird look pasted, floating, cut out, or artificially inserted.\n"
        "Do NOT generate multiple birds.\n"
        "Do NOT generate collage.\n"
        "Do NOT distort the bird excessively.\n"
    )
}

# 备选 prompt：视角变换 + 环境替换（单图联合任务）
PROMPT_VIEW_BG = {
    "name": "view_bg_edit",
    "prompt": (
        "You are given a single reference image of a bird.\n"
        "Keep the same bird identity: same species, feather colors, beak shape, "
        "texture, and body characteristics.\n"
        "Rotate the bird horizontally by about 60 degrees to create a new side view.\n"
        "At the same time, replace the original background with a realistic "
        "natural outdoor environment such as forest, grassland, wetland, lakeside, "
        "rocks, or tree branches.\n"
        "Adjust the bird pose, body balance, lighting, shadow, feather direction, "
        "perspective, and color tone so the bird naturally matches the new environment.\n"
        "The final image should look like a real wildlife photograph taken from the new angle.\n"
        "Generate ONLY ONE bird.\n"
        "Do NOT make the bird look pasted, floating, duplicated, cropped, or artificial.\n"
        "Do NOT generate collage.\n"
        "Do NOT generate multiple birds.\n"
    )
}

# ===========================================================================
# 三图输入 (Qwen-Image-Edit-2511)
# ===========================================================================

VIEW_PROMPTS_MULTI = [
    {
        "name": "view_bg_edit_v1",
        "prompt": (
            "You are given THREE reference images of a bird.\n"
            "The FIRST image is the MAIN reference and defines the bird's identity: "
            "species, exact feather colors, beak shape, texture, pattern, body proportions, "
            "and all visual characteristics. The other two images are AUXILIARY ONLY.\n"
            "STRICT RULE: The resulting bird MUST EXACTLY match the FIRST image's appearance. "
            "Do NOT use any identity attribute (color, pattern, morphology) from the auxiliary "
            "images; if they differ, ignore them entirely and stay faithful to the FIRST image only.\n"
            "Task:\n"
            "- Rotate the camera horizontally about 60 degrees around the bird "
            "(keep its original standing posture).\n"
            "- Replace the background with a realistic natural outdoor environment "
            "(forest, grassland, wetland, lakeside, rocks, tree branches) "
            "that fits the bird's habitat.\n"
            "- Adjust pose, lighting, shadow, feather direction, and perspective "
            "so the result looks like a natural wildlife photograph with lighting "
            "consistent with the environment.\n"
            "- Use the AUXILIARY images ONLY to improve structure coherence, "
            "texture consistency, and to infer the 3D viewpoint — "
            "they must not alter the bird's identity in any way.\n"
            "Important prohibitions:\n"
            "- Do NOT blend, morph, or mix features from the auxiliary images into the bird.\n"
            "- Do NOT generate multiple birds.\n"
            "- Do NOT deform, distort, or change the bird's body shape or proportions.\n"
            "- Do NOT copy layout from the auxiliary images.\n"
            "Generate exactly ONE bird that looks like the bird in the FIRST image, "
            "just seen from a different angle."
        )
    }
]
