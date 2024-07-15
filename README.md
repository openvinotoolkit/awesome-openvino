# Awesome OpenVINO ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

A curated list of OpenVINO based AI projects. The most exciting community projects based on OpenVINO are highlighted here. Explore a rich assortment of OpenVINO-based projects, libraries, and tutorials that cover a wide range of topics, from model optimization and deployment to real-world applications in various industries. 

This repository is a collaborative effort, continuously updated to provide you with the latest and most valuable resources for maximizing the potential of OpenVINO in your projects. If you want your project to appear in this list, please create a Pull Request or contact @DimaPastushenkov. 
Inspired by [Awesome oneAPI](https://github.com/oneapi-community/awesome-oneapi)

If your project is featured in this Awesome OpenVINO list, you are welcome to use the 'Mentioned in Awesome' badge on your project's repository. [![Mentioned in Awesome OpenVINO](https://awesome.re/mentioned-badge-flat.svg)](https://github.com/openvinotoolkit/awesome-openvino)


## What is OpenVINO 
OpenVINO™ is an open-source toolkit for AI inference optimization and deployment.
* Enhances deep learning performance in computer vision, automatic speech recognition, natural language processing, and other common tasks. 
* Utilize models trained with popular frameworks such as TensorFlow and PyTorch while efficiently reducing resource demands. 
* Deploy seamlessly across a spectrum of Intel® platforms, spanning from edge to cloud.


## Further resources:

* OpenVINO [GitHub repo](https://github.com/openvinotoolkit/openvino).

* To download OpenVINO toolkit, go [here](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html).

* A collection of ready-to-run Jupyter notebooks for learning and experimenting with the OpenVINO™ toolkit- [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks).
  

## Table of content 
1. [Generative AI](#Generative-AI)
2. [AI Computer Vision](#AI-Computer-Vision)
3. [AI Audio](#AI-Audio)
4. [OpenVINO API extentions](#OpenVINO-API-extentions)
5. [Natural Language Processing](#Natural-Language-Processing)
6. [Multimodal projects](#Multimodal-projects)
7. [Miscellaneous](#Miscellaneous)
8. [Educational](#Educational)


### Generative AI 
* [Stable Diffusion web UI](https://github.com/openvinotoolkit/stable-diffusion-webui/) - This is a repository for a browser interface based on Gradio library for Stable Diffusion
* [stable_diffusion.openvino](https://github.com/bes-dev/stable_diffusion.openvino) - This GitHub project provides an implementation of text-to-image generation using stable diffusion on Intel CPU or GPU. It requires Python 3.9.0 and is compatible with OpenVINO.
* [Fast SD](https://github.com/rupeshs/fastsdcpu) - FastSD CPU is a faster version of Stable Diffusion on CPU. Based on Latent Consistency Models and Adversarial Diffusion Distillation.[Read blog post about Fast Stable Diffusion on CPU using FastSD and OpenVINO.](https://nolowiz.com/fast-stable-diffusion-on-cpu-using-fastsd-cpu-and-openvino/)
* [OpenVINO™ AI Plugins for GIMP](https://github.com/intel/openvino-ai-plugins-gimp) - Provides a set of OpenVINO based plugins that add AI features to GIMP (GNU IMAGE
MANIPULATION PROGRAM)
* [OpenVINO Code - VSCode extension for AI code completion with OpenVINO](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/openvino_code) - VSCode extension for helping developers writing code with AI code assistant.
* [Enhancing Customer Service with Real-Time Sentiment Analysis: Leveraging LLMs and OpenVINO for Instant Emotional Insights](https://github.com/samontab/llm_sentiment) - The integration of LLMs with sentiment analysis models, further optimised by OpenVINO.
* [OV_SD_CPP](https://github.com/yangsu2022/OV_SD_CPP) - The pure C++ text-to-image pipeline, driven by the OpenVINO native API for Stable Diffusion v1.5 with LMS Discrete Scheduler.
* [QuickStyle](https://github.com/Y-T-G/QuickStyle) - A simple stylizing app utilizing OpenVINO to stylize common objects in images.
* [QuickPainter](https://github.com/Y-T-G/QuickPainter) - A simple inpainting app utilizing OpenVINO to remove common objects from images.
* [BlurAnything](https://github.com/Y-T-G/Blur-Anything) - An adaptation of the excellent Track Anything project which is in turn based on Meta's Segment Anything and XMem.
* [Stable Diffusion 2.1 on Intel ARC](https://github.com/jfsunx/OVSD21) - A simple and easy-to-use demo to run Stable Diffusion 2.1 for Intel ARC graphics card based on OpenVINO.
* [AI Video Builder](https://github.com/jediknight813/ai_video_builder) - Make videos with AI images from YouTube videos.

### AI Computer Vision
* [Visioncom](https://github.com/cabelo/visioncom) Visioncom is based on open_model_zoo project demo, the assisted communication system employs advanced computer vision technologies, using the OpenCV and OpenVINO libraries, to provide an interactive solution for patients with Amyotrophic Lateral Sclerosis (ALS).
* [BMW-IntelOpenVINO-Detection-Inference-API](https://github.com/BMW-InnovationLab/BMW-IntelOpenVINO-Detection-Inference-API) - This is a repository for an object detection inference API using OpenVINO, supporting both Windows and Linux operating systems
* [yolov5_export_cpu](https://github.com/SamSamhuns/yolov5_export_cpu) - The project provides documentation on exporting YOLOv5 models for fast CPU inference using Intel's OpenVINO framework
* [LidarObjectDetection-PointPillars](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/End-to-end-Workloads/LidarObjectDetection-PointPillars) (C++ based, requires AI toolkit and OpenVINO). demonstrates how to perform 3D object detection and classification using input data (point cloud) from a LIDAR sensor.
* [Image Processing with OpenVINO](https://github.com/AbhiLegend/Image-Processing-with-OpenVINO)
* [Implementing GAN with OpenVINO](https://github.com/AbhiLegend/GanOpenVINO)
* [RapidOCR](https://github.com/RapidAI/RapidOCR)
* [Pedestrian fall detection](https://github.com/guojin-yan/OpenVINO-CSharp-API/tree/csharp3.0/tutorial_examples/PP-Human_Fall_Detection) - Pedestrian fall detection. Deploying PP-Human based on OpenVINO C # API
* [OpenVINO Tennis Posture](https://github.com/salvino72/openvino-Tennis-Posture/) - Deciphering Tennis Posture with Artificial Intelligence
* [Cigarette Detection](https://github.com/Leviathanlzx/cgr_detection) - The project begins by YOLOv8-pose detecting human body positions and extracting skeletal information from images. Based on the skeletal poses, it assesses the elbow angles and the distance between hands and mouths for each individual. If successful, the RTDETR model is employed to detect cigarettes at the mouth zone.
* [FastSAM_Awesome_OpenVINO](https://github.com/zhg-SZPT/FastSAM_Awsome_Openvino) - The Fast Segment Anything Model(FastSAM) is a CNN Segment Anything Model trained by only 2% of the SA-1B dataset published by SAM authors. The FastSAM achieve a comparable performance with the SAM method at 50× higher run-time speed.
* [Computer Vision Models As Service](https://github.com/mohammad-oghli/CV-Models-Service) - implements different Computer Vision Deep Learning Models as a service.
* [Dance-with: Dance with your friends with the right pose!](https://github.com/bgb10/dance-with) - Dance-with corrects your dance posture using multi-person OpenPose, 2D pose estimation Deep Learning model.
* [Target-Person-Tracking-System](https://github.com/simpleis6est/Target-Person-Tracking-System) - Integration of face recognition and person tracking.
* [Metin2 Bot](https://github.com/Tigerly1/metin2bot) - bots for video game Metin2.
* [Machine control](https://github.com/5sControl/machine-control) - industrial machine surveillance system designed  to help increase efficiency of processes.
* [MeetingCam](https://github.com/nengelmann/MeetingCam) - Run your AI and CV algorithms in online meetings such as Zoom, Meets or Teams!
* [Virtual-Tryon](https://github.com/LZHMS/Virtual-Tryon) - Use AI to try on clothes with your pictures.
* [DepthAI Experiments](https://github.com/njnrn/depthai-experiments) - A collections of projects done with DepthAI.
* [Project Babble](https://github.com/SummerSigh/ProjectBabble) - Mouth tracking project designed to work with any existing VR headset.
* [Group Pose](https://github.com/Michel-liu/GroupPose-Paddle) - A Simple Baseline for End-to-End Multi-person Pose Estimation.
* [Frigate](https://github.com/blakeblackshear/frigate) - NVR With Realtime Object Detection for IP Cameras.
* [CGD OpenVINO Demo](https://github.com/sammysun0711/CGD_OpenVINO_Demo) - Efficient Inference and Quantization of CGD for Image Retrieval.
* [Risk package detection](https://github.com/AJV009/risk-package-detection) - Threat Detection and Unattended Baggage Detection with Associated Person Tracking.
* [YOLOv7-Intel](https://github.com/karnikkanojia/yolov7-intel) - Object Detection For Autonomous Vehicles.
* [Cerberus](https://github.com/gerardocipriano/Cerberus-Dog-Breed-Classification-and-Body-Localization-PyTorch) - Dog Breed Classification and Body Localization.
* [Criminal Activity recognition](https://github.com/ayush9304/Criminal-Activity-Video-Surveillance-using-Deep-Learning) - Criminal Activity Video Surveillance.
* [RapidOCR on OpenVINO GPU](https://github.com/jaggiK/rapidocr_openvinogpu) - A modified verison of RapidOCR to support OpenVINO GPU.
* [Yolov9 with OpenVINO](https://github.com/ahsan-raazaa/yolov9-openvino) - C++ and python implementation of YOLOv9 using OpenVINO
* [OpenVINO-Deploy](https://github.com/wxxz975/OpenVINO-Deploy) - A repository showcasing the deployment of popular object detection AI algorithms using the OpenVINO C++ API for efficient inference.
* [Clip-Chinese](https://github.com/towhee-io/examples/blob/main/image/text_image_search/3_build_chinese_image_search_engine.ipynb) - Chinese image-text similarity matching tasks, leverage OpenVINO and the Towhee embedding library.



### AI Audio
* [Whisper OpenVINO](https://github.com/zhuzilin/whisper-openvino)
  
### OpenVINO API extentions
* [OpenVINO™ C# API](https://github.com/guojin-yan/OpenVINO-CSharp-API) 
* [OpenVINO Java API](https://github.com/Hmm466/OpenVINO-Java-API)
* [OpenVINO LabVIEW API](https://github.com/wangstoudamire/lv_yolov8_openvino)
* [OpenVINO.net](https://github.com/sdcb/OpenVINO.NET)
* [OpenVINO-rs](https://github.com/intel/openvino-rs)
* [OpenVINO-GO Client](https://github.com/AbhiLegend/DrugLiphphilicty) - The end goal is to utilize a Go client to facilitate user-friendly batch processing of molecular data, interfacing with a Flask server that employs OpenVINO for optimized lipophilicity predictions and molecular visualization



### Natural Language Processing
* [Japanese chatbot Youri](https://github.com/yas-sim/openvino_japanese_chatbot_youri-7b-chat) - LLM Japanese chatbot demo program using Intel OpenVINO toolkit.
* [OpenVINO GPT-Neo](https://github.com/yousseb/ov-gpt-neo) - a port of GPT-Neo that uses OpenVINO.
* [Resume-Based Interview Preparation Tool](https://github.com/serinryu/interviewhelper_openvino) - The Resume-Based Interview Preparation Tool is a software application designed to streamline the interview process by helping interviewers generate relevant and meaningful questions based on a candidate's resume or portfolio page.

### Multimodal projects
* [Scene Explorer for kids](https://github.com/AJV009/explore-scene-w-object-detection) - Integration of a chat bot with an object detection algorithm.
* [Indoor Care Chatbot](https://github.com/AJV009/indoor-care-chatbot) - An Elderly Indoor Care Chatbot with an object detection algorithm.
* [SA2](https://github.com/LHBuilder/SA-Segment-Anything) - Vision-Oriented MultiModal AI.

### openSUSE 
* [OpenVINO Support](https://en.opensuse.org/SDB:Install_OpenVINO) This initiative generated openVINO compatibility with the openSUSE Linux platform. Because dependencies were added to tools and libraries for software development using C/C++ and other compilation directives for the programming language.

### Educational
* [NTUST Edge AI 2023 Artificial Intelligence and Edge Computing Practice ](https://github.com/OmniXRI/NTUST_EdgeAI_2023) - Educational meterials about AI and Edge Computing Practice GNU IMAGE
MANIPULATION PROGRAM

### Miscellaneous
* [JAX: Artificial Intelligence for everyone.](https://github.com/cabelo/jax) - JAX (Just an Artificial Intelligence Extended) is an optimized version of the openSUSE Linux image to work with openVINO. This platform was designed to facilitate the access and development of AI applications.
* [Shared Memory for AI inference](https://github.com/aiblockly/aixbroad_code_example) - Shared memory interface between OpenVINO and CODESYS. It allows to exchange variable between Control Application, written in IEC and OpenVINO Application, which performs inference
* [webnn-native](https://github.com/webmachinelearning/webnn-native)- WebNN Native is an implementation of the Web Neural Network API, providing building blocks, headers, and backends for ML platforms including DirectML, OpenVINO, and XNNPACK.
* [NVIDIA GPU Plugin](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/nvidia_plugin) - allows to perform deep neural networks inference on NVIDIA GPUs using CUDA, using OpenVINO API.
* [Token Merging for Stable Diffusion running with OpenVINO](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/token_merging) - An OpenVINO adopted version of Token Merging method.
* [Drug Discovery “Lipophilicity” using OpenVINO toolkit](https://github.com/AbhiLegend/DrugDisOpenVINO)- Finding Lipophilicity of peptides, proteins and molecules.
* [OpenVINO Quantization](https://github.com/AbhiLegend/OpenVinoQuantization)- Image Quantization Classification using STL 10 Dataset.
* [who_what_benchmark](https://github.com/andreyanufr/who_what_benchmark) - Simple and quick accuracy test for compressed, quantized, pruned, distilled LLMs from [NNCF](https://github.com/openvinotoolkit/nncf), Bitsandbytes, GPTQ, and BigDL-LLM.
* [OpenVINO with Docker](https://github.com/jonathanyeh0723/openvino-with-docker) - Dockerizing OpenVINO applications.
* [OpenVINO AICG Samples](https://github.com/sammysun0711/OpenVINO_AIGC_Samples) - A collection of samples for NLP and Image Generation.
* [OpenVINO Model Server k8s Terraform](https://github.com/dummyuser42/openvino-model-server-k8s-terraform) - Deploying Kubernetes cluster via Terraform as well as deploying and hosting a OpenVINO Model Server on it.

* [Application of Vision Language Models with ROS 2](https://github.com/nilutpolkashyap/vlms_with_ros2_workshop) -  Dives into vision-language models for Robotics applications using ROS 2 and Intel OpenVINO toolkit.

### Related Communities
See [Awesome oneAPI](https://github.com/oneapi-community/awesome-oneapi) for leading oneAPI and SYCL projects across diverse industries.
