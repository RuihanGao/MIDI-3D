# Note: need to manually provide the labels
python -m scripts.grounding_sam --image assets/data/midi_images_simplified_prompts/fruit_basket.jpeg --labels fruit basket --output ./output
python -m scripts.grounding_sam --image assets/data/midi_images_simplified_prompts/headphone_stand.png --labels headphone stand --output ./output
python -m scripts.grounding_sam --image assets/data/midi_images_simplified_prompts/stacked_cups.png --labels cup --output ./output
python -m scripts.grounding_sam --image assets/data/midi_images_simplified_prompts/vase_table.png --labels vase cake plate fork table --output ./output