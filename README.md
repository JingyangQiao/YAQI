<h1 align="center">
  <img src="assets/YAQI.png" alt="MIA Title" width="60"/><br>YAQI: Yielding Any-Modality Question-Answering Incremental Assistant
</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ComingSoon-b5212f.svg?logo=arxiv)](https://arxiv.org) [![Models](https://img.shields.io/badge/Models-YAQI-yellow?logo=huggingface)](https://huggingface.co/jingyang/YAQI-Vicuna-v1.5-13B) [![Datasets](https://img.shields.io/badge/Datasets-ModalEvolve-purple?logo=huggingface)](https://huggingface.co/datasets/jingyang/YAQI-Data) [![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) [![powered-by-sii](https://img.shields.io/badge/Powered%20By-SII-blue?style=plastic)](https://sii-group.com/fr-FR/sii-sud-ouest)

</div>

## 🚀 Latest News

- **[June 6, 2026]**:  🌈 Full stack is here. Whole Training and Evaluation Codebase, Models and Datasets have been published. Check them out!

## 📌 Overview

🤖 **YAQI (Yielding Any-Modality Question-Answering Incremental Assistant)** is a full-parameter modality continual learning framework for Omni-LMMs, developed by a joint team from the Shanghai Innovation Institute (SII) and East China Normal University (ECNU). YAQI is designed to transform multimodal models from "static multimodal learners" into "continually evolving modality assistants." Instead of relying on unrealistic one-stage PEFT adaptation, YAQI establishes a realistic two-stage modality evolution paradigm consisting of:
- **Pre-Training Stage**: A unified modality alignment stage that incrementally integrates new modalities into the shared semantic space.
- **Instruction-Tuning Stage**: A full-parameter multimodal reasoning stage that enables fine-grained cross-modal understanding and generation.
- **Mirror Replay Mechanism**: A mutual-information-guided replay strategy that preserves modality-wise learning states using only compact replay memory.

🌟 **Key Highlights**
- 🧬 **From Modality Learning to Modality Evolution**: YAQI moves beyond static multimodal alignment and enables Omni-LMMs to continuously evolve across text, image, audio, video, and music modalities under realistic sequential training scenarios.
- 🔬 **Realistic Full-Parameter Continual Learning**: Unlike previous PEFT-based works, YAQI reveals the true forgetting behaviors of Omni-LMMs under full-parameter tuning, including semantic drifting and instruction unfollowing.
- 🧠 **Learning-State-Aware Replay**: Mirror does not simply replay representative data distributions. Instead, it selects replay exemplars that best reflect the model’s current modality learning state through mutual information modeling and loss-aware replay selection.
- ⚡ **Efficient Yet Powerful**: Mirror significantly reduces computational overhead compared with traditional continual learning methods while consistently achieving state-of-the-art performance across multiple 7B–34B Omni-LMMs.
- 📊 **Comprehensive Benchmarking**: YAQI introduces the challenging Modal-Evolve benchmark, containing 3.2M multimodal samples across five modalities, providing a unified evaluation platform for modality continual learning research.


## 🛠️ Environment

* Python >= 3.10
* Pytorch == 2.0.1
* CUDA Version >= 11.7
* Install required packages:
```bash
git clone https://github.com/JingyangQiao/YAQI
cd YAQI
conda create -n yaqi python=3.10 -y
conda activate yaqi
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```

## 🧬 Modal-Evolve Benchmark
1.Please download the following datasets: Text, Image, Audio, Video, Music.

Dataset: 🤗 [Modal-Evolve](https://huggingface.co/datasets/jingyang/YAQI-Data)

2.After downloading all of them, organize the data as follows:
```
├── Dataset
|   └── Text
|    	└── codeqa
|           └── codeqa_test.jsonl
|       └── gsm8k
|           └── gsm8k_test.jsonl
|       └── text_instruct_3k.json
|   └── CC3M
|    	└── images
|       └── chat.json
|   └── Image
|    	└── coco
|           └── train2017
|    	└── COCO2014
|           └── val2014
|           └── train2014
|           └── coco_test.jsonl
|           └── vg_test.jsonl
|    	└── gqa
|           └── images
|           └── gqa_test.jsonl
|    	└── ocr_vqa
|           └── images
|           └── ocrvqa_test.jsonl
|    	└── textvqa
|           └── test_images
|           └── train_images
|           └── textvqa_test.jsonl
|    	└── vg
|           └── VG_100K
|           └── VG_100K_2
|    	└── llava_v1_5_image_624k.json
|   └── LAION-Audio-1M
|    	└── flash_15_2_random_snippets_0
|       └── ...
|    	└── flash_15_2_random_snippets_352
|       └── train.json
|   └── Audio
|    	└── Chord
|           └── Audio
|           └── chord_test.jsonl
|    	└── clothoaqa
|           └── train
|           └── test
|           └── clothoaqa_test.jsonl
|       └── EmoMusic
|           └── music_data
|           └── emotion_test.jsonl
|       └── MusicCaps
|           └── music_data
|           └── musiccaps_test.jsonl
|       └── Nsynth
|           └── audio
|           └── instrument_test.jsonl
|       └── tacos
|           └── train
|           └── test
|           └── tacos_test.jsonl
|    	└── audio_instruct_80k.json
|   └── OpenVid
|    	└── OpenVidHD
|    	└── annotations
|    	    └── chat.json
|   └── Video
|    	└── 0_30_s_academic_v0_1
|    	└── 0_30_s_activitynetqa
|    	└── 0_30_s_nextqa
|    	└── 0_30_s_perceptiontest
|    	└── 0_30_s_youtube_v0_1
|    	└── 30_60_s_academic_v0_1
|    	└── 30_60_s_activitynetqa
|    	└── 30_60_s_nextqa
|    	└── 30_60_s_perceptiontest
|    	└── 30_60_s_youtube_v0_1
|    	└── mc_test.jsonl
|    	└── oe_test.jsonl
|    	└── llava_v1_5_video_730k.json
|   └── Music
|       └── music_instruct_10k.json
```

## 🔍 YAQI Model Checkpoints

YAQI-Vicuna-v1.5-13b: 🤗 [checkpoints](https://huggingface.co/jingyang/YAQI-Vicuna-v1.5-13B)


## 📖 CLI Inference

```bash
CUDA_VISIBLE_DEVICES=0 python -m videollava.serve.cli --model-path "./checkpoints/YAQI-Vicuna-v1.5-13b" --file "path/to/your/file"
```

## 💻 Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/anyllava-llama-7b-mirror/finetune_text.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./mirror/mirror_text.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/anyllava-llama-7b-mirror/pretrain_image.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/anyllava-llama-7b-mirror/finetune_image.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./mirror/mirror_image.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/anyllava-llama-7b-mirror/pretrain_audio.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/anyllava-llama-7b-mirror/finetune_audio.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./mirror/mirror_audio.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/anyllava-llama-7b-mirror/pretrain_video.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/anyllava-llama-7b-mirror/finetune_video.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./mirror/mirror_video.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/anyllava-llama-7b-mirror/finetune_music.sh
```

## 💡 Testing

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/anyllava-eval/all.sh
```

## 👍 Acknowledgement

* [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) The codebase we built upon and it is an efficient large language and visual assistant.
* [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) An open source five modalities language-based retrieval framework.

## ⚖️ License

Released under the MIT License.

## ✏️ Citation

```BibTeX
@misc{qiao2026yaqi,
  title={Yielding Any-Modality Question-Answering Incremental Assistant},
  author={Jingyang Qiao and Chengwei Chen and Zhizhong Zhang and Xin Tan and Jingyu Gong and Yuan Xie},
  year={2026},
  note={Manuscript under review}
}
