
NEG_PROMPT = (
    "collage, multiple birds, split image, duplicated bird, "
    "blurry, low quality, deformed, malformed, bad anatomy, "
    "extra wings, extra legs, extra body parts, fused limbs"
)

# =====================================================
# Prompt（主图 + 多图参考版）
# =====================================================
# v1
# VIEW_PROMPTS = [
# {
# "name":"view_bg_edit_v2",
# "prompt":"""
# You are given THREE reference images of a bird.

# The FIRST image is the main reference image.
# The other two images are auxiliary references.

# Keep the same bird identity as the FIRST image:
# same species, feather colors, beak shape, texture, and body characteristics.

# If there are attribute conflicts in bird features among the three reference images, 
# follow the majority voting principle to unify the bird's characteristics.

# Rotate the **camera viewing angle horizontally about 60 degrees around the bird**, 
# do NOT rotate the flat image itself, keep the bird's original standing posture unchanged.

# At the same time, replace the background with a realistic natural outdoor environment
# such as forest, grassland, wetland, lakeside, rocks, or tree branches.

# Do NOT change the bird's inherent feather color, pattern, and body proportion; 
# strictly preserve original anatomical structure.

# Use the other two reference images ONLY to improve structure, texture consistency,
# and viewpoint reasoning.

# Adjust pose, lighting, shadow, feather direction, and perspective
# so the result looks like a real wildlife photograph with natural and harmonious light matching the environment.

# Generate ONLY ONE bird.
# Do NOT generate multiple birds.
# Do NOT copy layout from auxiliary images.
# Do NOT deform, distort, stretch or alter the bird's body shape and proportions.
# """
# }
# ]

# v2
VIEW_PROMPTS = [
    {
        "name": "view_bg_edit_v1",
        "prompt": (
            "You are given THREE reference images of a bird.\n"
            "The FIRST image is the MAIN reference and defines the bird's identity: "
            "species, exact feather colors, beak shape, texture, pattern, body proportions, "
            "and all visual characteristics. The other two images are AUXILIARY ONLY.\n"
            "STRICT RULE: The resulting bird MUST EXACTLY match the FIRST image's appearance. "
            "Do NOT use any identity attribute (color, pattern, morphology) from the auxiliary images; "
            "if they differ, ignore them entirely and stay faithful to the FIRST image only.\n"
            "Task:\n"
            "- Rotate the camera horizontally about 60 degrees around the bird (keep its original standing posture).\n"
            "- Replace the background with a realistic natural outdoor environment "
            "(forest, grassland, wetland, lakeside, rocks, tree branches) that fits the bird's habitat.\n"
            "- Adjust pose, lighting, shadow, feather direction, and perspective "
            "so the result looks like a natural wildlife photograph with lighting consistent with the environment.\n"
            "- Use the AUXILIARY images ONLY to improve structure coherence, texture consistency, "
            "and to infer the 3D viewpoint — they must not alter the bird's identity in any way.\n"
            "Important prohibitions:\n"
            "- Do NOT blend, morph, or mix features from the auxiliary images into the bird.\n"
            "- Do NOT generate multiple birds.\n"
            "- Do NOT deform, distort, or change the bird's body shape or proportions.\n"
            "- Do NOT copy layout from the auxiliary images.\n"
            "Generate exactly ONE bird that looks like the bird in the FIRST image, just seen from a different angle."
        )
    }
]

# v3
# VIEW_PROMPTS = [
#     {
#         "name": "view_bg_edit_main_ref",
#         "prompt": (
#             "Given 3 reference images of a bird. "
#             "Image 1 is the PRIMARY reference — it defines the bird's exact species, colors, pattern, proportions. "
#             "Images 2 and 3 are AUXILIARY only: use them to improve texture consistency and 3D viewpoint, "
#             "but NEVER change the bird's appearance from Image 1. "
#             "Task: rotate the camera horizontally by ~60°, keep the bird's posture, "
#             "replace background with a realistic natural outdoor scene (forest/grassland/wetland etc.), "
#             "light and shadow must match the environment. "
#             "Strictly output ONE bird, no multi‑bird, no deformation, no feature mixing from aux images."
#         )
#     }
# ]