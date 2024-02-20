# ICLR 2024 3D Generation

ICLR2024 paper list on 3D generation with brief introduction about each paper

## <font color=#0abab5>Oral</font>

### DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation (8 8 8 10) 

**Authors**: Jiaxiang Tang , Jiawei Ren, Hang Zhou , Ziwei Liu , Gang Zeng

<details span>
<summary><b>Abstract</b></summary>
Recent advances in 3D content creation mostly leverage optimization-based 3D generation via score distillation sampling (SDS). Though promising results have been exhibited, these methods often suffer from slow per-sample optimization, limiting their practical usage. In this paper, we propose DreamGaussian, a novel 3D content generation framework that achieves both efficiency and quality simultaneously. Our key insight is to design a generative 3D Gaussian Splatting model with companioned mesh extraction and texture refinement in UV space. In contrast to the occupancy pruning used in Neural Radiance Fields, we demonstrate that the progressive densification of 3D Gaussians converges significantly faster for 3D generative tasks. To further enhance the texture quality and facilitate downstream applications, we introduce an efficient algorithm to convert 3D Gaussians into textured meshes and apply a fine-tuning stage to refine the details. Extensive experiments demonstrate the superior efficiency and competitive generation quality of our proposed approach. Notably, DreamGaussian produces high-quality textured meshes in just 2 minutes from a single-view image, achieving approximately 10 times acceleration compared to existing methods.
</details> 

[📄Paper](https://openreview.net/pdf?id=UyNXMqnN3c) [🌐Project](https://dreamgaussian.github.io/) [💻Code ](https://github.com/dreamgaussian/dreamgaussian) #object_generation #texture_refinement #diffusion #3DGS #SDS

#### Pipeline

![Pasted image 20240130164900](ICLR2024 image\Pasted image 20240130164900.png)

#### Mesh Extraction

**Marching cubes** (2D case as an example)
![Pasted image 20240130185811](ICLR2024 image\Pasted image 20240130185811.png)

#### UV Mapping

![Pasted image 20240130190225](ICLR2024 image\Pasted image 20240130190225.png)

### <a id="LRM">LRM: Large Reconstruction Model for Single Image to 3D (8 8 8 10)</a>

**Authors**: Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, Hao Tan

<details span>
<summary><b>Abstract</b></summary>
We propose the first Large Reconstruction Model (LRM) that predicts the 3D model of an object from a single input image within just 5 seconds. In contrast to many previous methods that are trained on small-scale datasets such as ShapeNet in a category-specific fashion, LRM adopts a highly scalable transformer-based architecture with 500 million learnable parameters to directly predict a neural radiance field (NeRF) from the input image. We train our model in an end-to-end manner on massive multi-view data containing around 1 million objects, including both synthetic renderings from Objaverse and real captures from MVImgNet. This combination of a high-capacity model and large-scale training data empowers our model to be highly generalizable and produce high-quality 3D reconstructions from various testing inputs including real-world in-the-wild captures and images from generative models.
</details>

[📄Paper](https://openreview.net/pdf?id=sllU8vvsFF) [🌐Project](https://scalei3d.github.io/LRM/) #object_generation #triplane #NeRF



#### Pipeline

Image → Image feature → Triplane tokens (Triplane Nerf)
![Pasted image 20240130212144](ICLR2024 image\Pasted image 20240130212144.png)
A fully trained large transformer decoder can convert a single image to its corresponding triplane

**Ralated works:**

* TensoRF: Tensorial Radiance Fields (ECCV2022, Triplane NeRF) [PAPER](https://arxiv.org/abs/2203.09517)
* EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks (CVPR2022, stylegan generator → image feature → triplane feature → volume rendering → stylegan discriminator) [PAPER](https://arxiv.org/pdf/2112.07945.pdf)!![Pasted image 20240131220609](ICLR2024 image\Pasted image 20240131220609.png)

## <font color=#0abab5>Spotlight</font>

### DMV3D: Denoising Multi-view Diffusion Using 3D Large Reconstruction Model (6 8 8 10)

**Authors**: Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, Kai Zhang

<details span>
<summary><b>Abstract</b></summary>
We propose DMV3D, a novel 3D generation approach that uses a transformerbased 3D large reconstruction model to denoise multi-view diffusion. Our reconstruction model incorporates a triplane NeRF representation and can denoise noisy multi-view images via NeRF reconstruction and rendering, achieving singlestage 3D generation in ∼30s on single A100 GPU. We train DMV3D on largescale multi-view image datasets of highly diverse objects using only image reconstruction losses, without accessing 3D assets. We demonstrate state-ofthe-art results for the single-image reconstruction problem where probabilistic modeling of unseen object parts is required for generating diverse reconstructions with sharp textures. We also show high-quality text-to-3D generation results outperforming previous 3D diffusion models
</details>

[📄Paper](https://openreview.net/pdf?id=H4yQefeXhp) [🌐Project](https://justimyhxu.github.io/projects/dmv3d/) #object_generation #triplane #diffusion #multi-view_diffusion #viewpoint_information #NeRF 

#### Pipeline

![Pasted image 20240131221017](ICLR2024 image\Pasted image 20240131221017.png)
Use <a href="#LRM">LRM</a> to replace the UNet of diffusion model.
End-to-end training, during inference stage, once the multi-view images are fully denoised, our model offers a clean triplane NeRF, enabling 3D generation.

* Multi-view images as input, and add noise on different images with the same schedule                                                                            $\mathcal{I}=\{\mathbf{I}_1, \ldots, \mathbf{I}_N\}$, $\mathcal{I}_t=\{\sqrt{\bar{\alpha}_t} \mathbf{I}+\sqrt{1-\bar{\alpha}_t} \epsilon_{\mathbf{I}} \mid \mathbf{I} \in \mathcal{I}\}$
* Use LRM decoder to convert <font color=pink>noisy</font> multi-view images into <font color=pink>triplane tokens</font>
* Rendering <font color=pink>denoised</font> multi-view images from the triplane NeRF

### Adding 3D Geometry Control to Diffusion Models (5 5 6 8)

**Authors**: Wufei Ma, Qihao Liu, Jiahao Wang, Xiaoding Yuan, Angtian Wang, Yi Zhang, Zihao Xiao, Guofeng Zhang, Beijia Lu, Ruxiao Duan, Yongrui Qi, Adam Kortylewski, Yaoyao Liu, Alan Yuille

<details span>
<summary><b>Abstract</b></summary>
Diffusion models have emerged as a powerful method of generative modeling across a range of fields, capable of producing stunning photo-realistic images from natural language descriptions. However, these models lack explicit control over the 3D structure in the generated images. Consequently, this hinders our ability to obtain detailed 3D annotations for the generated images or to craft instances with specific poses and distances. In this paper, we propose a simple yet effective method that incorporates 3D geometry control into diffusion models. Our method exploits ControlNet, which extends diffusion models by using visual prompts in addition to text prompts. We generate images of the 3D objects taken from 3D shape repositories (e.g., ShapeNet and Objaverse), render them from a variety of poses and viewing directions, compute the edge maps of the rendered images, and use these edge maps as visual prompts to generate realistic images. With explicit 3D geometry control, we can easily change the 3D structures of the objects in the generated images and obtain ground-truth 3D annotations automatically. This allows us to improve a wide range of vision tasks, e.g., classification and 3D pose estimation, in both in-distribution (ID) and out-of-distribution (OOD) settings. We demonstrate the effectiveness of our method through extensive experiments on ImageNet-100, ImageNet-R, PASCAL3D+, ObjectNet3D, and OOD-CV. The results show that our method significantly outperforms existing methods across multiple benchmarks, e.g., 3.8 percentage points on ImageNet-100 using DeiT-B and 3.5 percentage points on PASCAL3D+ & ObjectNet3D using NeMo.
</details>

[📄Paper](https://openreview.net/pdf?id=XlkN11Xj6J) #object_generation #diffusion #ControlNet #spatial_information

#### Pipeline

![Pasted image 20240218224703](ICLR2024 image\Pasted image 20240218224703.png)

* Get a <font color=pink>CAD model</font> from the 3D shape repository(e.g., ShapeNet and Objaverse)
* Render them from a variety of poses and viewing directions, then get the <font color=pink>canny edge</font> $\mathcal{E}_{3 \mathrm{D}}$ of the rendered image 
* Using <font color=pink>ControlNet</font> to add 3D geometry information $\mathcal{E}_{3 \mathrm{D}}$

### SyncDreamer: Generating Multiview-consistent Images from a Single-view Image (6 8 8 8 10)

**Authors**: Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, Wenping Wang

<details span>
<summary><b>Abstract</b></summary>
In this paper, we present a novel diffusion model called SyncDreamer that generates multiview-consistent images from a single-view image. Using pretrained large-scale 2D diffusion models, recent work Zero123 demonstrates the ability to generate plausible novel views from a singleview image of an object. However, maintaining consistency in geometry and colors for the generated images remains a challenge. To address this issue, we propose a synchronized multiview diffusion model that models the joint probability distribution of multiview images, enabling the generation of multiview-consistent images in a single reverse process. SyncDreamer synchronizes the intermediate states of all the generated images at every step of the reverse process through a 3D-aware feature attention mechanism that correlates the corresponding features across different views. Experiments show that SyncDreamer generates images with high consistency across different views, thus making it well-suited for various 3D generation tasks such as novel-view synthesis, text-to-3D, and image-to-3D.
</details>

[📄Paper](https://arxiv.org/pdf/2309.03453.pdf) [🌐Project](https://liuyuan-pal.github.io/SyncDreamer/) [💻Code](https://github.com/liuyuan-pal/SyncDreamer) #object_generation #diffusion #multi-view_diffusion #viewpoint_information #spatial_information

#### Pipeline

![Pasted image 20240219203031](ICLR2024 image\Pasted image 20240219203031.png)

* Given the noisy 4 images from target views, we can get a spatial volume to represent these 4 images
* Pretrained zero123 model concatenates the input view $y$ with the noisy target view ${x_t}^{(n)}$ as the input to UNet. The viewpoint information $\Delta v^{(n)}$ and CLIP feature as the condition
* Also, construct a view frustum volume of target view from the spitial volume to enforce consistency among multiple generated views.
  **Ralated works:**
* Zero-1-to-3: Zero-shot One Image to 3D Object(ICCV2023) [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf)

## <font color=#0abab5>Poster</font>

### MVDream: Multi-view Diffusion for 3D Generation (6 6 6 8)

**Authors**: Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, Xiao Yang

<details span>
<summary><b>Abstract</b></summary>
We introduce MVDream, a multi-view diffusion model that is able to generate consistent multi-view images from a given text prompt. Learning from both 2D and 3D data, a multi-view diffusion model can achieve the generalizability of 2D diffusion models and the consistency of 3D renderings. We demonstrate that such a multi-view prior can serve as a generalizable 3D prior that is agnostic to 3D representations. It can be applied to 3D generation via Score Distillation Sampling, significantly enhancing the consistency and stability of existing 2D-lifting methods. It can also learn new concepts from a few 2D examples, akin to DreamBooth, but for 3D generation.
</details>

[📄Paper](https://arxiv.org/pdf/2308.16512.pdf) [🌐Project](https://mv-dream.github.io/) [💻Code](https://github.com/bytedance/MVDream) #object_generation #diffusion #multi-view_diffusion #viewpoint_information 

#### Pipeline

![Pasted image 20240219205842](ICLR2024 image\Pasted image 20240219205842.png)

![Pasted image 20240219205854](ICLR2024 image\Pasted image 20240219205854.png)

* Connecting all different views together and doing <font color=pink>3D self-attention</font> to generate consistent multi-view image at once
* Add camera embeddings to time embeddings as residuals
* Support multi-view Dreambooth

### Magic123: One Image to High-Quality 3D Object Generation Using Both 2D and 3D Diffusion Priors (5 5 8 8)

**Authors**: Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, Bernard Ghanem

<details span>
<summary><b>Abstract</b></summary>
We present “Magic123”, a two-stage coarse-to-fine approach for high-quality, textured 3D meshes generation from a single unposed image in the wild using both 2D and 3D priors. In the first stage, we optimize a neural radiance field to produce a coarse geometry. In the second stage, we adopt a memory-efficient differentiable mesh representation to yield a high-resolution mesh with a visually appealing texture. In both stages, the 3D content is learned through reference view supervision and novel views guided by a combination of 2D and 3D diffusion priors. We introduce a single trade-off parameter between the 2D and 3D priors to control exploration (more imaginative) and exploitation (more precise) of the generated geometry. Additionally, we employ textual inversion and monocular depth regularization to encourage consistent appearances across views and to prevent degenerate solutions, respectively. Magic123 demonstrates a significant improvement over previous image-to-3D techniques, as validated through extensive experiments on synthetic benchmarks and diverse real-world images
</details>

[📄Paper](https://arxiv.org/pdf/2306.17843.pdf) [🌐Project](https://guochengqian.github.io/project/magic123/) [💻Code](https://github.com/guochengqian/Magic123) #object_generation #diffusion #SDS #viewpoint_information #texture_refinement 

#### Pipeline

![Pasted image 20240219221404](ICLR2024 image\Pasted image 20240219221404.png)

* In coarse stage, do SDS on both 2D diffusion model(SD) and 3D diffusion model(zero123)
  ![Pasted image 20240219221727](ICLR2024 image\Pasted image 20240219221727.png)
* In fine stage, do refinement on DMTet Mesh
  **Ralated works:**
* Zero-1-to-3: Zero-shot One Image to 3D Object(ICCV2023) [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Zero-1-to-3_Zero-shot_One_Image_to_3D_Object_ICCV_2023_paper.pdf)

### Instant3D: Fast Text-to-3D with Sparse-view Generation and Large Reconstruction Model(6 8 8)

**Authors**: Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, FujunLuan, YinghaoXu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, Sai Bi

<details span>
<summary><b>Abstract</b></summary>
Text-to-3D with diffusion models has achieved remarkable progress in recent years. However, existing methods either rely on score distillation-based optimiza tion which suffer from slow inference, low diversity and Janus problems, or are feed-forward methods that generate low-quality results due to the scarcity of 3D training data. In this paper, we propose Instant3D, a novel method that generates high-quality and diverse 3D assets from text prompts in a feed-forward manner. We adopt a two-stage paradigm, which first generates a sparse set of four struc tured and consistent views from text in one shot with a fine-tuned 2D text-to-image diffusion model, and then directly regresses the NeRF from the generated images with a novel transformer-based sparse-view reconstructor. Through extensive ex periments, we demonstrate that our method can generate diverse 3D assets of high visual quality within 20 seconds, which is two orders of magnitude faster than previous optimization-based methods that can take 1 to 10 hours.
</details>

[📄Paper](https://arxiv.org/pdf/2311.06214.pdf) [🌐Project](https://instant-3d.github.io/) #object_generation #diffusion #multi-view_diffusion #triplane #NeRF 

#### Pipeline

![Pasted image 20240220113850](ICLR2024 image\Pasted image 20240220113850.png)

![Pasted image 20240220113831](ICLR2024 image\Pasted image 20240220113831.png)

* By dividing a picture into four Gaussian blobs, the 2D diffusion model can generate pictures from 4 viewpoints at once.
* The architecture ot the Transformer-based reconstructor is just the same as <a href="#LRM">LRM</a>

### DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior (5 6 6 8)

**Authors**: Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, Yebin Liu 

<details span>
<summary><b>Abstract</b></summary>
We present DreamCraft3D, a hierarchical 3D content generation method that produces high-fidelity and coherent 3D objects. We tackle the problem by leveraging a 2D reference image to guide the stages of geometry sculpting and texture boosting. A central focus of this work is to address the consistency issue that existing works encounter. To sculpt geometries that render coherently, we perform score distillation sampling via a view-dependent diffusion model. This 3D prior, alongside several training strategies, prioritizes the geometry consistency but compromises the texture fidelity. We further propose bootstrapped score distillation to specifically boost the texture. We train a personalized diffusion model, Dreambooth, on the augmented renderings of the scene, imbuing it with 3D knowledge of the scene being optimized. The score distillation from this 3D-aware diffusion prior provides view-consistent guidance for the scene. Notably, through an alternating optimization of the diffusion prior and 3D scene representation, we achieve mutually reinforcing improvements: the optimized 3D scene aids in training the scene-specific diffusion model, which offers increasingly view-consistent guidance for 3D optimization. The optimization is thus bootstrapped and leads to substantial texture boosting. With tailored 3D priors throughout the hierarchical generation, DreamCraft3D generates coherent 3D objects with photorealistic renderings, advancing the state-of-the-art in 3D content generation.
</details>

[📄Paper](https://arxiv.org/pdf/2310.16818.pdf) [🌐Project](https://mrtornado24.github.io/DreamCraft3D/) [💻Code](https://github.com/deepseek-ai/DreamCraft3D) #object_generation #diffusion #viewpoint_information #SDS #texture_refinement 

![Pasted image 20240220130314](ICLR2024 image\Pasted image 20240220130314.png)

* In coarse stage, do SDS on both 2D diffusion model and 3D diffusion model(zero123)
* In refinement stage, finetune the diffusion model with the multi-view texture-augmented images, using [DreamBooth](https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf). And use this finetuned model to gradually optimize the textures (Hope the score function of the optimized 3D scene match the score function of the DreamBooth model)
  $\nabla_\theta \mathcal{L}_{\mathrm{BSD}}(\phi, g(\theta))=\mathbb{E}_{t, \boldsymbol{\epsilon}, c}[\omega(t)(\boldsymbol{\epsilon}_{\text {DreamBooth }}(\boldsymbol{x}_t ; y, t, r_{t^{\prime}}(\boldsymbol{x}), c)-\boldsymbol{\epsilon}_{\text {lora }}(\boldsymbol{x}_t ; y, t, \boldsymbol{x}, c)) \frac{\partial \boldsymbol{x}}{\partial \theta}]$
  Compare with the [ProlificDreamer](https://proceedings.neurips.cc/paper_files/paper/2023/file/1a87980b9853e84dfb295855b425c262-Paper-Conference.pdf)(Hope the score function of the optimized 3D scene match the score function of the pretrained model))
  $\nabla_\theta \mathcal{L}_{\mathrm{VSD}}(\phi, g(\theta))=\mathbb{E}_{t, \boldsymbol{\epsilon}}[\omega(t)(\boldsymbol{\epsilon}_{\text {Pretrained}}(\boldsymbol{x}_t ; y, t)-\boldsymbol{\epsilon}_{\text {lora }}(\boldsymbol{x}_t ; y, t, x, c)) \frac{\partial \boldsymbol{x}}{\partial \theta}]$

### SWEETDREAMER: ALIGNING GEOMETRIC PRIORS IN 2D DIFFUSION FOR CONSISTENT TEXT-TO-3D (5 5 6 8)

**Authors**: Weiyu Li, Rui Chen, Xuelin Chen, Ping Tan 

<details span>
<summary><b>Abstract</b></summary>
It is inherently ambiguous to lift 2D results from pre-trained diffusion models to a 3D world for text-to-3D generation. 2D diffusion models solely learn view agnostic priors and thus lack 3D knowledge during the lifting, leading to the multi view inconsistency problem. We find that this problem primarily stems from ge ometric inconsistency, and avoiding misplaced geometric structures substantially mitigates the problem in the final outputs. Therefore, we improve the consis tency by aligning the 2D geometric priors in diffusion models with well-defined 3D shapes during the lifting, addressing the vast majority of the problem. This is achieved by fine-tuning the 2D diffusion model to be viewpoint-aware and to produce view-specific coordinate maps of canonically oriented 3D objects. In our process, only coarse 3D information is used for aligning. This “coarse” alignment not only resolves the multi-view inconsistency in geometries but also retains the ability in 2D diffusion models to generate detailed and diversified high-quality ob jects unseen in the 3D datasets. Furthermore, our aligned geometric priors (AGP) are generic and can be seamlessly integrated into various state-of-the-art pipelines, obtaining high generalizability in terms of unseen shapes and visual appearance while greatly alleviating the multi-view inconsistency problem. Our method repre sents a new state-of-the-art performance with a 85+% consistency rate by human evaluation, while many previous methods are around 30%.
</details>

[📄Paper](https://arxiv.org/pdf/2310.02596.pdf) [🌐Project](https://sweetdreamer3d.github.io/) [💻Code (not yet)](https://github.com/wyysf-98/SweetDreamer) #object_generation #diffusion #spatial_information #viewpoint_information #SDS

#### Pipeline

![Pasted image 20240220172411](ICLR2024 image\Pasted image 20240220172411.png)

![Pasted image 20240220172448](ICLR2024 image\Pasted image 20240220172448.png)

* In first stage, fine-tune a 2D diffusion model to generate viewpoint conditioned canonical coordinates maps(CCM)
* In the SDS stage, render both CCM and rgb image from the 3D representation(Nerf or DMTet), and do use both orginal and fine-tuned 2D diffusion models to optimize the 3D reprensentation.

### TEXT-TO-3D WITH CLASSIFIER SCORE DISTILLATION (5 6 8 8)

**Authors**: Xin Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Song-Hai Zhang, Xiaojuan Qi

<details span>
<summary><b>Abstract</b></summary>
Text-to-3D generation has made remarkable progress recently, particularly with methods based on Score Distillation Sampling (SDS) that leverages pre-trained 2Ddiffusion models. While the usage of classifier-free guidance is well acknowl edged to be crucial for successful optimization, it is considered an auxiliary trick rather than the most essential component. In this paper, we re-evaluate the role of classifier-free guidance in score distillation and discover a surprising finding: the guidance alone is enough for effective text-to-3D generation tasks. We name this method Classifier Score Distillation (CSD), which can be interpreted as using an implicit classification model for generation. This new perspective reveals new in sights for understanding existing techniques. We validate the effectiveness of CSD across a variety of text-to-3D tasks including shape generation, texture synthesis, and shape editing, achieving results superior to those of state-of-the-art methods.
</details>

[📄Paper](https://arxiv.org/pdf/2310.19415.pdf) [🌐Project](https://xinyu-andy.github.io/Classifier-Score-Distillation/) [💻Code](https://github.com/CVMI-Lab/Classifier-Score-Distillation) #object_generation #diffusion #SDS

#### Pipeline

In original SDS, the gradient is expressed as$\nabla_\theta \mathcal{L}_{\mathrm{SDS}}=\mathbb{E}_{t, \epsilon, \mathbf{c}}[w(t)(\epsilon_\phi(\mathbf{x}_t ; y, t)-\epsilon) \frac{\partial \mathbf{x}}{\partial \theta}]$
And can be expressed as
$\epsilon_{\phi}(\mathbf{x}_t ; y, t)-\epsilon =\delta_{x}(\mathbf{x}_{t} ; y, t) = \underbrace{[\epsilon_\phi(\mathbf{x}_t ; y, t)-\epsilon]}_{\delta_x^{\text {gen }}}+\omega \cdot \underbrace{[\epsilon_\phi(\mathbf{x}_t ; y, t)-\epsilon_\phi(\mathbf{x}_t ; t)]}_{\delta_x^{\text {cls }}}$
![Pasted image 20240220211004](ICLR2024 image\Pasted image 20240220211004.png)
The authors find that

* The gradient norm of the gen erative prior is several times larger than that of the classifier score in Fig(a)
* However, to generate high quality results, a large guidance weight must be set (e.g., ω = 40), as shown in Fig(b). When incorporating both components, the large guid ance weight actually causes the gradient from the classifier score to dominate the optimization di rection.
* Moreover, the optimization process fails whenrelying solely on the generative component, as indicated by setting ω = 0
  So they introduced to use classifier score sistillation(only consider$\delta_x^{\text {cls }}$) to align the rendered noisy image and the text y.

### DreamTime: An Improved Optimization Strategy for Diffusion-Guided 3D Generation (3 6 8 8)

**Authors**: Yukun Huang, Jianan Wang, Yukai Shi, Boshi Tang, Xianbiao Qi, Lei Zhang 

<details span>
<summary><b>Abstract</b></summary>
Text-to-image diffusion models pre-trained on billions of image-text pairs have recently enabled 3D content creation by optimizing a randomly initialized dif ferentiable 3D representation with score distillation. However, the optimization process suffers slow convergence and the resultant 3D models often exhibit two limitations: (a) quality concerns such as missing attributes and distorted shape and textures; (b) extremely low diversity comparing to text-guided image synthesis. In this paper, we show that the conflict between the 3D optimization process and uniform timestep sampling in score distillation is the main reason for these limi tations. To resolve this conflict, we propose to prioritize timestep sampling with monotonically non-increasing functions, which aligns the 3D optimization pro cess with the sampling process of diffusion model. Extensive experiments show that our simple redesign significantly improves 3D content creation with faster convergence, better quality and diversity.
</details>

[📄Paper](https://openreview.net/pdf?id=1bAUywYJTU) #object_generation #diffusion #SDS 

#### Pipeline

![Pasted image 20240220212231](ICLR2024 image\Pasted image 20240220212231.png)

![Pasted image 20240220212243](ICLR2024 image\Pasted image 20240220212243.png)
