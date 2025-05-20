[中文](README-CN.md)|[English](README.md)

# parakeet-tdt-0.6b-v2 的 ComfyUI 节点

准确而快速的自动语音识别（ASR）模型. 专为高质量的英语转录而设计，支持标点符号、大写和准确的时间戳预测. 

![](https://github.com/billwuhao/ComfyUI_parakeet-tdt/blob/main/images/20250520010321.png)

## 用法

- 快速添加字幕:
![](https://github.com/billwuhao/ComfyUI_parakeet-tdt/blob/main/images/2025-05-20_10-36-53.png)

## 📣 更新

[2025-05-20]⚒️: 发布 v1.0.0。

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_parakeet-tdt.git
cd ComfyUI_parakeet-tdt
pip install -r requirements.txt

# linux
pip install nemo_toolkit['asr']

# windows
git clone https://github.com/NVIDIA/NeMo
cd NeMo
pip install '.[asr]'
```
如果报错: `RuntimeError: CUDA error: operation not supported`. 在 ComfyUI 启动参数添加 `--disable-cuda-malloc`, 例如:
```
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --disable-cuda-malloc
```

## 模型下载

- [parakeet-tdt-0.6b-v2.nemo](https://huggingface.co//nvidia/parakeet-tdt-0.6b-v2/blob/main/parakeet-tdt-0.6b-v2.nemo): 下载放到 `ComfyUI/models/TTS` 目录下.

## 鸣谢

[NeMo](https://github.com/NVIDIA/NeMo)