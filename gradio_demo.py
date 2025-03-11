import os
import random
import tempfile
from typing import Any, List, Union

import gradio as gr
import numpy as np
import torch
from gradio_image_prompter import ImagePrompter
from gradio_litmodel3d import LitModel3D
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor

from midi.pipelines.pipeline_midi import MIDIPipeline
from scripts.grounding_sam import plot_segmentation, segment
from scripts.inference_midi import run_midi

# import spaces

# Constants
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "VAST-AI/MIDI-3D"

MARKDOWN = """
## Image to 3D Scene with [MIDI-3D](https://huanngzh.github.io/MIDI-Page/)
1. Upload an image, and draw bounding boxes for each instance by holding and dragging the mouse. Then clik "Run Segmentation" to generate the segmentation result. <b>Ensure instances should not be too small and bounding boxes fit snugly around each instance.</b>
2. <b>Check "Do image padding" in "Generation Settings" if instances in your image are too close to the image border.</b> Then click "Run Generation" to generate a 3D scene from the image and segmentation result.
3. If you find the generated 3D scene satisfactory, download it by clicking the "Download GLB" button.
"""

EXAMPLES = [
    [
        {
            "image": "assets/example_data/Cartoon-Style/00_rgb.png",
        },
        "assets/example_data/Cartoon-Style/00_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Cartoon-Style/01_rgb.png",
        },
        "assets/example_data/Cartoon-Style/01_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Cartoon-Style/03_rgb.png",
        },
        "assets/example_data/Cartoon-Style/03_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/00_rgb.png",
        },
        "assets/example_data/Realistic-Style/00_seg.png",
        42,
        False,
        True,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/01_rgb.png",
        },
        "assets/example_data/Realistic-Style/01_seg.png",
        42,
        False,
        True,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/02_rgb.png",
        },
        "assets/example_data/Realistic-Style/02_seg.png",
        42,
        False,
        False,
    ],
    [
        {
            "image": "assets/example_data/Realistic-Style/05_rgb.png",
        },
        "assets/example_data/Realistic-Style/05_seg.png",
        42,
        False,
        False,
    ],
]

os.makedirs(TMP_DIR, exist_ok=True)

# Prepare models
## Grounding SAM
segmenter_id = "facebook/sam-vit-base"
sam_processor = AutoProcessor.from_pretrained(segmenter_id)
sam_segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(
    DEVICE, DTYPE
)
## MIDI-3D
local_dir = "pretrained_weights/MIDI-3D"
snapshot_download(repo_id=REPO_ID, local_dir=local_dir)
pipe: MIDIPipeline = MIDIPipeline.from_pretrained(local_dir).to(DEVICE, DTYPE)
pipe.init_custom_adapter(
    set_self_attn_module_names=[
        "blocks.8",
        "blocks.9",
        "blocks.10",
        "blocks.11",
        "blocks.12",
    ]
)


# Utils
def split_rgb_mask(rgb_image, seg_image):
    if isinstance(rgb_image, str):
        rgb_image = Image.open(rgb_image)
    if isinstance(seg_image, str):
        seg_image = Image.open(seg_image)
    rgb_image = rgb_image.convert("RGB")
    seg_image = seg_image.convert("L")

    rgb_array = np.array(rgb_image)
    seg_array = np.array(seg_image)

    label_ids = np.unique(seg_array)
    label_ids = label_ids[label_ids > 0]

    instance_rgbs, instance_masks, scene_rgbs = [], [], []

    for segment_id in sorted(label_ids):
        # Here we set the background to white
        white_background = np.ones_like(rgb_array) * 255

        mask = np.zeros_like(seg_array, dtype=np.uint8)
        mask[seg_array == segment_id] = 255
        segment_rgb = white_background.copy()
        segment_rgb[mask == 255] = rgb_array[mask == 255]

        segment_rgb_image = Image.fromarray(segment_rgb)
        segment_mask_image = Image.fromarray(mask)
        instance_rgbs.append(segment_rgb_image)
        instance_masks.append(segment_mask_image)
        scene_rgbs.append(rgb_image)

    return instance_rgbs, instance_masks, scene_rgbs


@torch.no_grad()
@torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
def run_segmentation(image_prompts: Any, polygon_refinement: bool) -> Image.Image:
    rgb_image = image_prompts["image"].convert("RGB")

    # pre-process the layers and get the xyxy boxes of each layer
    if len(image_prompts["points"]) == 0:
        gr.Error("Please draw bounding boxes for each instance on the image.")
    boxes = [
        [
            [int(box[0]), int(box[1]), int(box[3]), int(box[4])]
            for box in image_prompts["points"]
        ]
    ]

    # run the segmentation
    detections = segment(
        sam_processor,
        sam_segmentator,
        rgb_image,
        boxes=[boxes],
        polygon_refinement=polygon_refinement,
    )
    seg_map_pil = plot_segmentation(rgb_image, detections)

    torch.cuda.empty_cache()

    return seg_map_pil


@torch.no_grad()
@torch.autocast(device_type=DEVICE, dtype=torch.bfloat16)
def run_generation(
    rgb_image: Any,
    seg_image: Union[str, Image.Image],
    seed: int,
    randomize_seed: bool = False,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    do_image_padding: bool = False,
):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    if not isinstance(rgb_image, Image.Image) and "image" in rgb_image:
        rgb_image = rgb_image["image"]

    scene = run_midi(
        pipe,
        rgb_image,
        seg_image,
        seed,
        num_inference_steps,
        guidance_scale,
        do_image_padding,
    )

    _, tmp_path = tempfile.mkstemp(suffix=".glb", prefix="midi3d_", dir=TMP_DIR)
    scene.export(tmp_path)

    torch.cuda.empty_cache()

    return tmp_path, tmp_path, seed


# Demo
with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                image_prompts = ImagePrompter(label="Input Image", type="pil")
                seg_image = gr.Image(
                    label="Segmentation Result", type="pil", format="png"
                )

            with gr.Accordion("Segmentation Settings", open=False):
                polygon_refinement = gr.Checkbox(label="Polygon Refinement", value=True)
            seg_button = gr.Button("Run Segmentation")

            with gr.Accordion("Generation Settings", open=False):
                do_image_padding = gr.Checkbox(label="Do image padding", value=False)
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=50,
                )
                guidance_scale = gr.Slider(
                    label="CFG scale",
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=7.0,
                )
            gen_button = gr.Button("Run Generation", variant="primary")

        with gr.Column():
            model_output = LitModel3D(label="Generated GLB", exposure=1.0, height=500)
            download_glb = gr.DownloadButton(label="Download GLB", interactive=False)

    with gr.Row():
        gr.Examples(
            examples=EXAMPLES,
            fn=run_generation,
            inputs=[image_prompts, seg_image, seed, randomize_seed, do_image_padding],
            outputs=[model_output, download_glb, seed],
            cache_examples=False,
        )

    seg_button.click(
        run_segmentation,
        inputs=[
            image_prompts,
            polygon_refinement,
        ],
        outputs=[seg_image],
    ).then(lambda: gr.Button(interactive=True), outputs=[gen_button])

    gen_button.click(
        run_generation,
        inputs=[
            image_prompts,
            seg_image,
            seed,
            randomize_seed,
            num_inference_steps,
            guidance_scale,
            do_image_padding,
        ],
        outputs=[model_output, download_glb, seed],
    ).then(lambda: gr.Button(interactive=True), outputs=[download_glb])


demo.launch()
