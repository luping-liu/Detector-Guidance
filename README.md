# Detector Guidance for Multi-Object Text-to-Image Generation

[Luping Liu](https://luping-liu.github.io/)<sup>1</sup>, Zijian Zhang<sup>1</sup>, [Yi Ren](https://rayeren.github.io/)<sup>2</sup>, Rongjie Huang<sup>1</sup>, Xiang Yin<sup>2</sup>, Zhou Zhao<sup>1</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>ByteDance AI Lab

In this work, we introduce Detector Guidance (DG), which integrates a latent object detection model to separate different objects during the generation process. More precisely, DG first performs latent object detection on cross-attention maps (CAMs) to obtain object information. Based on this information, DG then masks conflicting prompts and enhances related prompts by manipulating the following CAMs. Human evaluations demonstrate that DG provides an 8-22% advantage in preventing the amalgamation of conflicting concepts and ensuring that each object possesses its unique region without any human involvement and additional iterations.

Code will be released soon.

![](images/compare1.jpg)
