# MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation

## 🏠 [Project Page](https://huanngzh.github.io/MIDI-Page/) | [Paper](https://arxiv.org/abs/2412.03558) | [Model](https://huggingface.co/VAST-AI/MIDI-3D) | [Online Demo](https://huggingface.co/spaces/VAST-AI/MIDI-3D)

MIDI is a 3D generative model for single image to compositional 3D scene generation. Unlike existing methods that rely on reconstruction or retrieval techniques or recent approaches that employ multi-stage object-by-object generation, MIDI extends pre-trained image-to-3D object generation models to multi-instance diffusion models, enabling the simultaneous generation of multiple high-quality 3D instances with accurate spatial relationships and high generalizability.

## 🌟 Features

* **High Quality:** It produces diverse 3D scenes at high quality with intricate shape.
* **High Generalizability:** It generalizes to real image and stylized image inputs although trained only on synthetic data.
* **High Efficiency:** It generates 3D scenes from segmented instance images, without lengthy steps or time-consuming per-scene optimization.

## 🔥 Updates

* [2025-03] Release model weights, gradio demo, inference scripts of MIDI-3D.

## 🔨 Installation

Clone the repo first:

```Bash
git clone https://github.com/VAST-AI-Research/MIDI-3D.git
cd MIDI-3D
```

(Optional) Create a fresh conda env:

```Bash
conda create -n midi python=3.10
conda activate midi
```

Install necessary packages (torch > 2):

```Bash
# pytorch (select correct CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# other dependencies
pip install -r requirements.txt
```

## 💡 Usage

The following running scripts will automatically download model weights from [VAST-AI/MIDI-3D](https://huggingface.co/VAST-AI/MIDI-3D) to local directory `pretrained_weights/MIDI-3D`.

### Launch Demo

```Bash
python gradio_demo.py
```

**Important!!** Please check out our instructional video!

https://github.com/user-attachments/assets/4fc8aea4-010f-40c7-989d-6b1d9d3e3e09

**The web demo is also available on [Hugging Face Spaces](https://huggingface.co/spaces/VAST-AI/MIDI-3D)!**

### Inference Scripts

If running MIDI with command lines, you need to obtain the segmentation map of the scene image firstly. We provide a script to run [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) in `scripts/grounding_sam.py`. The following example command will produce a segmentation map in the `./segmentation.png`.

```Bash
python -m scripts.grounding_sam --image assets/example_data/Cartoon-Style/04_rgb.png --labels lamp sofa table dog --output ./
```

Then you can run MIDI with the rgb image and segmentation map, using our provided inference script `scripts/inference_midi.py`. The following command will save the generated 3D scene `output.glb` in the output dir.

```Bash
python -m scripts.inference_midi --rgb assets/example_data/Cartoon-Style/00_rgb.png --seg assets/example_data/Cartoon-Style/00_seg.png --output-dir "./"
```

**Important!!!**

* We recommend using the [interactive demo](#launch-demo) to get a segmentation map of moderate granularity.
* If instances in your image are too close to the image border, please add `--do-image-padding` to the running scripts of MIDI.

## Citation

```
@article{huang2024midi,
  title={MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation},
  author={Huang, Zehuan and Guo, Yuanchen and An, Xingqiao and Yang, Yunhan and Li, Yangguang and Zou, Zixin and Liang, Ding and Liu, Xihui and Cao, Yanpei and Sheng, Lu},
  journal={arXiv preprint arXiv:2412.03558},
  year={2024}
}
```
