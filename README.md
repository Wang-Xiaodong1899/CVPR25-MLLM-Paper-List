# CVPR25-MLLM-Paper-List
CVPR 2025 Multimodal Large Language Models Paper List

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
  - [Image LLMs](#image-llms)
  - [Video LLMs](#video-llms)
  - [Unified LLMs](#unified-llms)
  - [Other Modalities](#other-modalities)
  - [Preference Optimization](#preference-optimization)
  - [Jailbreak](#jailbreak)
  - [Benchmarks](#benchmarks)
  - [Retrieval](#retrieval)


## Image LLMs
- **LLaVA-Critic**: Learning to Evaluate Multimodal Models [Paper](https://arxiv.org/abs/2410.02712) [Page](https://llava-vl.github.io/blog/2024-10-03-llava-critic/)
- **Img-Diff**: Contrastive Data Synthesis for Multimodal Large Language Models [Paper](https://arxiv.org/abs/2408.04594) [Code](https://github.com/modelscope/data-juicer/tree/ImgDiff)
- **FlashSloth**: Lightning Multimodal Large Language Models via Embedded Visual Compression [Paper](https://arxiv.org/abs/2412.04317) [Code](https://github.com/codefanw/FlashSloth)
- **BlueLM-V-3B**: Algorithm and System Co-Design for Multimodal Large Language Models on Mobile Devices [Paper](https://arxiv.org/abs/2411.10640v1) [Code]()
- **Insight-V**: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models [Paper](https://arxiv.org/abs/2411.14432) [Code](https://github.com/dongyh20/Insight-V)
- **Critic-V**: VLM Critics Help Catch VLM Errors in Multimodal Reasoning [Paper](https://arxiv.org/abs/2411.18203)
- **Mono-InternVL**: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training [Paper](https://arxiv.org/abs/2410.08202) [Code](https://internvl.github.io/blog/2024-10-10-Mono-InternVL/)
- **DivPrune**: Diversity-based Visual Token Pruning for Large Multimodal Models [Paper](https://arxiv.org/abs/2503.02175) [Code](https://github.com/vbdi/divprune)
- **ODE**: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models [Paper](https://arxiv.org/abs/2409.09318)
- Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering [Paper](https://arxiv.org/abs/2411.16863) [Code](https://github.com/aimagelab/ReflectiVA)
- **AGLA**: Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention [Paper](https://arxiv.org/abs/2406.12718) [Code](https://github.com/Lackel/AGLA)
- **ICT**: Image-Object Cross-Level Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models [Paper](https://arxiv.org/abs/2411.15268v1)
- Can Large Vision-Language Models Correct Grounding Errors By Themselves? [Paper](https://openreview.net/pdf?id=fO1xnmW8T6)
- **Molmo and PixMo**: Open Weights and Open Data for State-of-the-Art Vision-Language Models [Paper](https://arxiv.org/abs/2409.17146) [Page](https://molmo.allenai.org/blog)
- **Nullu**: Mitigating Object Hallucinations in Large Vision-Language Models via HalluSpace Projection [Paper](https://arxiv.org/abs/2412.13817) [Code](https://github.com/Ziwei-Zheng/Nullu)
- **HiRes-LLaVA**: Restoring Fragmentation Input in High-Resolution Large Vision-Language Models [Paper](https://arxiv.org/abs/2407.08706)
- Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting and Mitigating Object Hallucinations via Attention Lens [Paper](https://arxiv.org/abs/2411.16724)
- **HoVLE**: Unleashing the Power of Monolithic Vision-Language Models with Holistic Vision-Language Embedding [Paper](https://arxiv.org/abs/2412.16158)
- **VoCo-LLaMA**: Towards Vision Compression with Large Language Models [Paper](https://github.com/Yxxxb/VoCo-LLaMA?tab=readme-ov-file) [Code](https://github.com/Yxxxb/VoCo-LLaMA?tab=readme-ov-file)




## Video LLMs
- **Apollo**: An Exploration of Video Understanding in Large Multi-Modal Models [Paper](https://arxiv.org/abs/2412.10360) [Page](https://apollo-lmms.github.io/)
- **VideoRefer Suite**: Advancing Spatial-Temporal Object Understanding with Video LLM [Paper](https://arxiv.org/abs/2501.00599) [Code](https://github.com/CircleRadon/VideoRefer-suite)
- **Seq2Time**: Sequential Knowledge Transfer for Video LLM Temporal Grounding [Paper](https://arxiv.org/abs/2411.16932)
- On the Consistency of Video Large Language Models in Temporal Comprehension [Paper](https://arxiv.org/abs/2411.12951) [Code](https://github.com/minjoong507/Consistency-of-Video-LLM)
- **DyCoke**: Dynamic Compression of Tokens for Fast Video Large Language Models [Paper](https://arxiv.org/abs/2411.15024) [Code](https://github.com/KD-TAO/DyCoke)
- **PAVE**: Patching and Adapting Video Large Language Models [Paper](https://drive.google.com/file/d/1whMeSxRh1BiUlunBTz26-7MTjv2K7cRF/view) [Code](https://github.com/dragonlzm/PAVE)
- **DynFocus**: Dynamic Cooperative Network Empowers LLMs with Video Understanding [Paper](https://arxiv.org/abs/2411.12355) [Code](https://github.com/Simon98-AI/DynFocus/tree/main)
- M-LLM Based Video Frame Selection for Efficient Video Understanding [Paper](https://arxiv.org/abs/2502.19680)
- Adaptive Keyframe Sampling for Long Video Understanding [Paper](https://arxiv.org/abs/2502.21271) [Code](https://github.com/ncTimTang/AKS)
- **VISTA**: Enhancing Long-Duration and High-Resolution Video Understanding by Video Spatiotemporal Augmentation [Paper](https://arxiv.org/abs/2412.00927) [Code](https://github.com/TIGER-AI-Lab/VISTA)
- **LLaVA-ST**: A Multimodal Large Language Model for Fine-Grained Spatial-Temporal Understanding [Paper](https://arxiv.org/abs/2501.08282) [Code](https://github.com/appletea233/LLaVA-ST)
- Unlocking Video-LLM via Agent-of-Thoughts Distillation [Paper](https://arxiv.org/abs/2412.01694v1) [Page](https://zhengrongz.github.io/AoTD/)
- **STEP**: Enhancing Video-LLMs’ Compositional Reasoning by Spatio-Temporal Graph-guided Self-Training [Paper](https://arxiv.org/abs/2412.00161)
- **PVC**: Progressive Visual Token Compression for Unified Image and Video Processing in Large Vision-Language Models [Paper](https://arxiv.org/abs/2412.09613) [Code](https://github.com/OpenGVLab/PVC?tab=readme-ov-file)


## Unified LLMs
- **Janus**: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation [Paper](https://arxiv.org/abs/2410.13848) [Code](https://github.com/deepseek-ai/Janus)
- **JanusFlow**: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation [Paper](https://arxiv.org/abs/2411.07975) [Code](https://github.com/deepseek-ai/Janus)
- **MMAR**: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling [Paper](https://arxiv.org/abs/2410.10798) [Code](https://github.com/ydcUstc/MMAR)
- **CoMM**: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation [Paper](https://arxiv.org/abs/2406.10462) [Code](https://github.com/HKUST-LongGroup/CoMM)
- **WeGen**: A Unified Model for Interactive Multimodal Generation as We Chat [Paper](https://arxiv.org/abs/2503.01115v1) [Code](https://github.com/hzphzp/WeGen)
- **SynerGen-VL**: Towards Synergistic Image Understanding and Generation with Vision Experts and Token Folding [Paper](https://arxiv.org/abs/2412.09604)
- **TokenFlow**: Unified Image Tokenizer for Multimodal Understanding and Generation [Paper](https://arxiv.org/abs/2412.03069) [Code](https://github.com/ByteFlow-AI/TokenFlow)
## Other Modalities
- **Thinking in Space**: How Multimodal Large Language Models See, Remember and Recall Spaces [Paper](https://arxiv.org/abs/2412.14171) [Code](https://github.com/vision-x-nyu/thinking-in-space)
- **EventGPT**: Event Stream Understanding with Multimodal Large Language Models [Paper](https://github.com/XduSyL/EventGPT) [Code](https://github.com/XduSyL/EventGPT)

## Preference Optimization
- **Task Preference Optimization**: Improving Multimodal Large Language Models Performance with Vision Task Alignment [Paper](https://github.com/OpenGVLab/TPO) [Code](https://github.com/OpenGVLab/TPO)
- **SymDPO**: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization [Paper](https://arxiv.org/abs/2411.11909) [Code](https://github.com/APiaoG/SymDPO)
- Continual SFT Matches Multimodal RLHF with Negative Supervision [Paper](https://arxiv.org/abs/2411.14797)
- Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key [Paper](https://arxiv.org/abs/2501.09695v2) [Code](https://github.com/zhyang2226/OPA-DPO)

## Jailbreak
- **Immune**: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment [Paper](https://arxiv.org/abs/2411.18688) [Code](https://github.com/itsvaibhav01/Immune)
- Distraction is All You Need for Multimodal Large Language Model Jailbreaking [Paper](https://arxiv.org/abs/2502.10794)
- **Playing the Fool**: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy [Paper](https://openreview.net/pdf?id=rgiIZ3pcZY)
- Exploring Visual Vulnerabilities via Multi-Loss Adversarial Search for Jailbreaking Vision-Language Models [Paper](https://arxiv.org/abs/2411.18000)

## Retrieval
- **GME**: Improving Universal Multimodal Retrieval by Multimodal LLMs [Paper](https://arxiv.org/abs/2412.16855) [Code](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/blob/main/gme_inference.py)
- Retrieval-Augmented Personalization for Multimodal Large Language Models [Paper](https://arxiv.org/abs/2410.13360) [Code](https://github.com/Hoar012/RAP-MLLM)



## Benchmarks
- **Video-MME**: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis [Paper](https://arxiv.org/abs/2405.21075) [Code](https://github.com/BradyFU/Video-MME)
- **MLVU**: Benchmarking Multi-task Long Video Understanding [Paper](https://arxiv.org/abs/2406.04264) [Code](https://github.com/JUNJIE99/MLVU)
- **MMVU**: Measuring Expert-Level Multi-Discipline Video Understanding [Paper](https://arxiv.org/abs/2501.12380) [Code](https://github.com/yale-nlp/MMVU)
- **MV-MATH**: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts [Paper](https://arxiv.org/abs/2502.20808) [Page](https://eternal8080.github.io/MV-MATH.github.io/)
- **VideoAutoArena**: An Automated Arena for Evaluating Large Multimodal Models in Video Analysis through User Simulation [Paper](https://arxiv.org/abs/2411.13281) [Code](https://github.com/VideoAutoArena/VideoAutoArena)
- **OVBench**: How Far is Your Video-LLMs from Real-World Online Video Understanding? [Paper](https://arxiv.org/pdf/2501.05510) [Code](https://github.com/JoeLeelyf/OVO-Bench?tab=readme-ov-file)
- **ECBench**: Can Multi-modal Foundation Models Understand the Egocentric World? A Holistic Embodied Cognition Benchmark [Paper](https://arxiv.org/abs/2501.05031) [Code](https://github.com/Rh-Dang/ECBench)
- Localizing Events in Videos with Multimodal Queries [Paper](https://arxiv.org/abs/2406.10079) [Code](https://github.com/icq-benchmark/icq-benchmark)
- **ICQ**: Is `Right' Right? Enhancing Object Orientation Understanding in Multimodal Language Models through Egocentric Instruction Tuning [Paper](https://arxiv.org/abs/2411.16761) [Code](https://github.com/jhCOR/EgoOrientBench)
- **VidHalluc**: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding [Paper](https://arxiv.org/abs/2412.03735) [Page](https://vid-halluc.github.io/)
- **VidComposition**: Can MLLMs Analyze Compositions in Compiled Video? [Paper](https://arxiv.org/abs/2411.10979v1) [Code](https://github.com/yunlong10/VidComposition)
- Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly [Paper](https://arxiv.org/abs/2406.10638)
- **VL-RewardBench**: A Challenging Benchmark for Vision-Language Generative Reward Models [Paper](https://arxiv.org/abs/2411.17451) [Page](https://vl-rewardbench.github.io/)
- Benchmarking Large Vision-Language Models via Directed Scene Graph for Comprehensive Image Captioning [Paper](https://arxiv.org/abs/2412.08614) [Code](https://github.com/LuFan31/CompreCap)
- **OpenING**: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation [Paper](https://arxiv.org/abs/2411.18499) [Code](https://github.com/LanceZPF/OpenING)
