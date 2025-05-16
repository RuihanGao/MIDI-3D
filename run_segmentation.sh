# Note: need to manually provide the labels

# # 2025-05-01. Test on 4 simplified prompts
# python -m scripts.grounding_sam --image assets/data/midi_images_simplified_prompts/fruit_basket.jpeg --labels fruit basket --output ./output
# python -m scripts.grounding_sam --image assets/data/midi_images_simplified_prompts/headphone_stand.png --labels headphone stand --output ./output
# python -m scripts.grounding_sam --image assets/data/midi_images_simplified_prompts/stacked_cups.png --labels cup --output ./output
# python -m scripts.grounding_sam --image assets/data/midi_images_simplified_prompts/vase_table.png --labels vase cake plate fork table --output ./output

# 2025-05-15. Test on 13 prompts
CUDA_VISIBLE_DEVICES=0 python -m scripts.grounding_sam --image assets/data/midi_images_20250515/07_vase_table.png --labels vase table flower cake plate --output ./output
CUDA_VISIBLE_DEVICES=0 python -m scripts.grounding_sam --image assets/data/midi_images_20250515/08_fruit_basket.png --labels fruit basket --output ./output
CUDA_VISIBLE_DEVICES=0 python -m scripts.grounding_sam --image assets/data/midi_images_20250515/09_cups_plates.png --labels cup plate --output ./output
CUDA_VISIBLE_DEVICES=0 python -m scripts.grounding_sam --image assets/data/midi_images_20250515/10_toothbrush_soap.png --labels soap dish toothpate toothbrush razor --output ./output
CUDA_VISIBLE_DEVICES=0 python -m scripts.grounding_sam --image assets/data/midi_images_20250515/11_radio_books.png --labels radio cow book pencils cup --output ./output
CUDA_VISIBLE_DEVICES=0 python -m scripts.grounding_sam --image assets/data/midi_images_20250515/12_sofa_bear.png --labels sofa plushies cushions --output ./output
CUDA_VISIBLE_DEVICES=0 python -m scripts.grounding_sam --image assets/data/midi_images_20250515/13_pool_balls.png --labels pool balls toys --output ./output


