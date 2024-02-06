# LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation

PyTorch code of the ACM MM'23 paper "LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation". 

## Introduction

In the text-to-image generation field, recent remarkable progress in Stable Diffusion makes it possible to generate rich kinds of novel photorealistic images. However, current models still face misalignment issues (e.g., problematic spatial relation understanding and numeration failure) in complex natural scenes, which impedes the high-faithfulness text-to-image generation. Although recent efforts have been made to improve controllability by giving fine-grained guidance (e.g., sketch and scribbles), this issue has not been fundamentally tackled since users have to provide such guidance information manually. In this work, we  strive to synthesize high-fidelity images that are semantically aligned with a given textual prompt without any guidance. Toward this end, we propose a coarse-to-fine paradigm to achieve layout planning and image generation. Concretely, we first generate the coarse-grained layout conditioned on a given textual prompt via in-context learning based on Large Language Models. Afterward, we propose a fine-grained object-interaction diffusion method to synthesize high-faithfulness images conditioned on the prompt and the automatically generated layout. Extensive experiments demonstrate that our proposed method outperforms the state-of-the-art models in terms of layout and image generation.

![model](/assets/framework.png)

## Dataset: COCO-NSS1K

By filtering, scrutinizing, and sampling from captions of COCO 2014, we built a new benchmark called [COCO-NSS1K](https://github.com/LayoutLLM-T2I/LayoutLLM-T2I/tree/main/data) to evaluate the Numerical reasoning, Spatial and Semantic Relation understanding of text-to-image generative models. 

|               | #Num | #Avg.bbox | #Avg.Cap.Len | Caption Examples                                             |
| ------------- | ---- | --------- | ------------ | ------------------------------------------------------------ |
| **Numerical** | 155  | 6.23      | 9.55         | • two old cell phones and a wooden table. <br />• two plates some food and a fork knife and spoon. |
| **Spatial**   | 200  | 5.35      | 10.25        | • a large clock tower next to a small white church.<br />• a bowl with some noodles inside of it. |
| **Semantic**  | 200  | 7.10      | 10.62        | • a train on a track traveling through a countryside.<br />• a living room filled with couches, chairs, TV, and windows. |
| **Mixed**     | 188  | 6.94      | 10.76        | • one motorcycle rider riding going up the mountain, two going down.<br />• a group of three bathtubs sitting next to each other. |
| **Null**      | 200  | 6.17      | 9.62         | • a kitchen scene complete with a dishwasher, sink, and an oven.<br />• a person with a hat and some ski poles. |
| **Total**     | 943  | 6.35      | 10.18        | -                                                            |

## Getting Started

### Installation

**1. Download repo and create environment**

```bash
https://github.com/LayoutLLM-T2I/LayoutLLM-T2I.git
conda create -n layoutllm_t2i python=3.8
conda activate layoutllm_t2i
pip install -r requirements.txt
```

**2. Download and prepare the pretrained weights**

This model includes a policy model and a [GLIGEN](https://github.com/gligen/GLIGEN)-based relation-aware diffusion model. The policy weights can be downloaded [here](https://drive.google.com/file/d/1t7M-uqgB5GMATJGEe2sM7oZX_ex4EysE/view?usp=sharing)and saved in `POLICY_CKPT`. The diffusion model weights are downloaded [here (Baidu)](https://pan.baidu.com/s/1mHyEljbq45Komzp3Iduw8g?pwd=mzac) or [here (Huggingface)](https://huggingface.co/leigangqu/LayoutLLM-T2I/tree/main) and saved in  `DIFFUSION_CKPT`. 

### Text-to-Image Generation

Download the [candidate example file](https://drive.google.com/file/d/14bZ7bOcLG5P9b6mu_StWOv4MyzBIqbfs/view?usp=sharing) in which each instance was randomly sampled from COCO2014, and save it in `CANDIDATE_PATH`.  Obtain `OPENAI_API_KEY` from the [openai platform](https://platform.openai.com/api-keys). 

Run the generation code: 

```bash
export OPENAI_API_KEY=OPENAI_API_KEY

python txt2img.py --folder generation_samples
    --prompt PROMPT
    --policy_ckpt_path POLICY_CKPT
    --diff_ckpt_path DIFFUSION_CKPT
    --cand_path CANDIDATE_PATH
    --num_per_prompt 1
```

`PROMPT` denotes your prompt. 

### Training

To train the the policy network, we first download images from COCO2014 and the [sampled training examples](https://drive.google.com/file/d/1pVEE9TeV1dpz43sxyek6N0m9C8poO46H/view?usp=sharing). Alternatively, one may sample some examples by himself. Besides, download the [weights](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth) of the aesthetic predictor pre-trained on the LAION dataset.  

Run the training code: 

```bash
export OPENAI_API_KEY=OPENAI_API_KEY

python -u train_rl.py
    --gpu GPU_ID
    --exp EXPERIMENT_NAME
    --img_dir IMAGE_DIR
    --sampled_data_dir DATA_PATH
    --diff_ckpt_path DIFFUSION_CKPT 
    --aesthetic_ckpt AESTHETIC_CKPT
```

## Reference

```
@inproceedings{qu2023layoutllm,
  title={LayoutLLM-T2I: Eliciting Layout Guidance from LLM for Text-to-Image Generation},
  author={Qu, Leigang and Wu, Shengqiong and Fei, Hao and Nie, Liqiang and Chua, Tat-Seng},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={643--654},
  year={2023}
}
```
