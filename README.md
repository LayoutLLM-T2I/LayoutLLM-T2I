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

## Model

Coming soon.





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

