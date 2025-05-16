# # 2025-05-01. Test on 4 simplified prompts
# CUDA_VISIBLE_DEVICES=1 python -m scripts.inference_midi --rgb assets/data/midi_images_simplified_prompts/fruit_basket.jpeg --seg output/fruit_basket_seg.png --output-dir "./output" & 
# CUDA_VISIBLE_DEVICES=2 python -m scripts.inference_midi --rgb assets/data/midi_images_simplified_prompts/headphone_stand.png --seg output/headphone_stand_seg.png --output-dir "./output" &
# CUDA_VISIBLE_DEVICES=3 python -m scripts.inference_midi --rgb assets/data/midi_images_simplified_prompts/stacked_cups.png --seg output/stacked_cups_seg.png --output-dir "./output" 
# CUDA_VISIBLE_DEVICES=1 python -m scripts.inference_midi --rgb assets/data/midi_images_simplified_prompts/vase_table.png --seg output/vase_table_seg.png --output-dir "./output"


# 2025-05-15. Test on 13 prompts
CUDA_VISIBLE_DEVICES=0 python -m scripts.inference_midi --rgb assets/data/midi_images_20250515/07_vase_table.png --seg output/07_vase_table_seg.png --output-dir "./output"
CUDA_VISIBLE_DEVICES=0 python -m scripts.inference_midi --rgb assets/data/midi_images_20250515/08_fruit_basket.png --seg output/08_fruit_basket_seg.png --output-dir "./output"
CUDA_VISIBLE_DEVICES=0 python -m scripts.inference_midi --rgb assets/data/midi_images_20250515/09_cups_plates.png --seg output/09_cups_plates_seg.png --output-dir "./output"
CUDA_VISIBLE_DEVICES=0 python -m scripts.inference_midi --rgb assets/data/midi_images_20250515/10_toothbrush_soap.png --seg output/10_toothbrush_soap_seg.png --output-dir "./output"
CUDA_VISIBLE_DEVICES=0 python -m scripts.inference_midi --rgb assets/data/midi_images_20250515/11_radio_books.png --seg output/11_radio_books_seg.png --output-dir "./output"
CUDA_VISIBLE_DEVICES=0 python -m scripts.inference_midi --rgb assets/data/midi_images_20250515/12_sofa_bear.png --seg output/12_sofa_bear_seg.png --output-dir "./output"
CUDA_VISIBLE_DEVICES=0 python -m scripts.inference_midi --rgb assets/data/midi_images_20250515/13_pool_balls.png --seg output/13_pool_balls_seg.png --output-dir "./output"