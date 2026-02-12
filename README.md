# ComfyUI-Seans-OmniTag
<img width="1319" height="1007" alt="Screenshot 2026-02-12 152534" src="https://github.com/user-attachments/assets/0acb3d61-bb10-4ca0-a32b-d31dd14fdeec" />

# 🛠️ Sean's OmniTag: The Ultimate LTX-2 Dataset Tool

**Sean's OmniTag** is a powerhouse ComfyUI node designed specifically for creators building datasets for **LTX-Video (LTX-2)**, **Flux**, and high-fidelity video LoRAs. It automates the most painful parts of data prep: resampling, resizing, visual captioning, and audio transcription.

## 🚀 Why use this?

Stop wasting time building a spiderweb of nodes just to prep a dataset. Sean's OmniTag is a "One-and-Done" solution. Whether you are dropping in a folder of high-res images or a full-length video, this single node handles the extraction, the 24 FPS resampling, the smart-scaling, the visual captioning, and the Whisper transcription in one smooth motion. Best of all? Despite its power, it is highly optimized. Thanks to 4-bit quantization, it cruises along using only ~7GB of VRAM, 


🎥 Video Segmentation (The "Magic" for Video Training)
target_fps: Sets the frame rate of the exported video clips.

video_segment_seconds: How long each training clip should be (e.g., 5.0 seconds).

segment_skip: This determines how much of the video is skipped between segments. A skip of 10 means the node grabs a 5-second clip, skips ahead 50 Seconds, and grabs another, ensuring your dataset has visual variety.

video_max_segments: The total number of clips to extract from a single long video. This prevents one long movie from overwhelming your entire dataset.



    📐 Deep Dive: Resolution & Aspect Ratio Logic
One of the most powerful features of Sean's OmniTag Processor is how it handles varied input sizes. Whether you are throwing 4K vertical TikToks or old 4:3 home movies at it, the node ensures the output is optimized for AI training.

1. The "Smart Resize" System
Instead of stretching or squashing your media, the node uses a Longest-Edge Scaling method.

How it works: The node looks at your target_resolution (e.g., 768) and identifies which side of your image/video is the longest.

The Math: It calculates a scaling factor based on that longest side and applies it to the entire frame.

Result: If you set the resolution to 768:

A 1920x1080 (Landscape) video becomes 768x432.

A 1080x1920 (Portrait) video becomes 432x768.

A 1024x1024 (Square) image becomes 768x768.
 
    * 
* **📂 Batch Workflow:** Point it at a folder of images or a single long-form video, and it will churn out paired `.png/.mp4` and `.txt` files ready for training.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/seanhan19911990-source/ComfyUI-Seans-OmniTag.git
