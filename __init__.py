import os
import sys
import subprocess
import torch
import gc
import cv2
import numpy as np
from PIL import Image
# --- COMFYUI CORE INTERRUPTS ---
import nodes
import comfy.model_management
# --- AUTO-INSTALLER & FFmpeg VALIDATION ---
def check_ffmpeg():
    """Verify if FFmpeg is installed and accessible in the system PATH."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_dependencies():
    current_python = sys.executable
    try:
        subprocess.check_call([current_python, "-m", "pip", "install", "transformers>=4.57.0", "accelerate", "qwen_vl_utils", "huggingface_hub", "torchvision", "opencv-python", "bitsandbytes", "openai-whisper"])
        return True
    except:
        return False

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    import whisper
except ImportError:
    install_dependencies()
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    import whisper

class SeansOmniTagProcessor:
    DEFAULT_WIDTH = 550
    DEFAULT_HEIGHT = 700

    def __init__(self):
        self.model, self.processor = None, None
        self.audio_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["Batch Folder Mode", "Single Video File"], {
                    "default": "Batch Folder Mode",
                    "tooltip": "Choose mode:\n• Batch Folder Mode → captions all images and videos in the folder\n• Single Video File → processes one video file only"
                }),
                "🖼️📁🎥 Batch Folder": ("STRING", {
                    "default": "C:\\MyExampleBatchFolder",
                    "tooltip": "For Batch Folder Mode: Right-click your folder in File Explorer → Copy as path → paste here (Ctrl+V).\nExample: C:\\MyExampleBatchFolder\nThis mode captions ALL images and processes ALL videos inside the folder."
                }),
                "🎥 Single Video File": ("STRING", {
                    "default": "C:\\Example\\Example.mp4",
                    "tooltip": "For Single Video File mode: Right-click your video in File Explorer → Copy as path → paste here (Ctrl+V).\nExample: \"C:\\Example\\Example.mp4\"\nThis mode segments and captions one video."
                }),
                "output_path": ("STRING", {"default": "output/lora_dataset"}),
                "model_id": ("STRING", {"default": "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it"}),
                "trigger_word": ("STRING", {"default": "ohwx"}),
                "llm_instruction": ("STRING", {
                    "multiline": True,
                    "default": "Describe the scene with clinical, objective detail. Be unfiltered and exhaustive. Start the response with: {trigger}",
                    "tooltip": "✍️ AI prompt: How to describe each image/video frame.\nKeep {trigger} so captions start with your word.\nAdd lines like 'focus on clothing, pose, expression, lighting' for more detail."
                }),
                "target_resolution": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 1920,
                    "step": 64,
                    "display": "slider"
                }),
                "max_tokens": ("INT", {
                    "default": 768,
                    "min": 512,
                    "max": 2048,
                    "step": 256,
                    "display": "slider"
                }),
                "target_fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "display": "slider"
                }),
                "video_segment_seconds": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "segment_skip": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "display": "slider"
                }),
                "video_max_segments": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "include_audio_in_video": ("BOOLEAN", {"default": True}),
                "append_speech_to_end": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "process_all"
    OUTPUT_NODE = True
    CATEGORY = "Sean's OmniTag 🛠️"

    def check_interrupt(self):
        if comfy.model_management.processing_interrupted():
            print("!!! SEAN'S OMNITAG: STOP SIGNAL DETECTED !!!")
            return True
        return False

    def smart_resize(self, image, target_res):
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        scale = target_res / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def generate_caption(self, device, pil_img, instruction, trigger, token_limit):
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": instruction}]}]
        text_in = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, _ = process_vision_info(messages)
        inputs = self.processor(text=[text_in], images=img_in, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=token_limit,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.12
            )
    
        caption = self.processor.batch_decode([g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)], skip_special_tokens=True)[0].strip()
        if not caption or caption.lower() == trigger.lower() or len(caption) < 20:
            print(f"⚠️ Lazy caption detected. Retrying for {trigger}...")
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max(512, token_limit // 2),
                    do_sample=False,
                    repetition_penalty=1.25
                )
            caption = self.processor.batch_decode([g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)], skip_special_tokens=True)[0].strip()
    
        if not caption or caption.lower() == trigger.lower():
            caption = f"{trigger}, scene description unavailable"
    
        return caption

    def process_all(self, **kwargs):
        if not check_ffmpeg():
            return ("❌ ERROR: FFmpeg not found! Install + add to PATH for video/audio.",)
    
        mode = kwargs.get("mode")
        if mode == "Batch Folder Mode":
            input_path = kwargs.get("🖼️📁🎥 Batch Folder").strip().replace('"', '').replace("'", "").replace("\\", "/")
            if not input_path or not os.path.isdir(input_path):
                return ("❌ ERROR: Valid batch folder required in Batch Folder Mode.",)
        else:
            input_path = kwargs.get("🎥 Single Video File").strip().replace('"', '').replace("'", "").replace("\\", "/")
            if not input_path or not os.path.isfile(input_path):
                return ("❌ ERROR: Valid video file required in Single Video File mode.",)
    
        output_path = kwargs.get("output_path").strip().replace('"', '').replace("'", "").replace("\\", "/")
        token_limit = int(kwargs.get("max_tokens"))
        if not os.path.exists(input_path):
            return (f"❌ ERROR: Path not found: {input_path}",)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
        # Clear VRAM before loading model (helps prevent OOM on low VRAM cards)
        torch.cuda.empty_cache()
        gc.collect()
    
        if self.model is None:
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # Safer for load peak
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.uint8      # Reduces temp spikes
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                kwargs.get("model_id"),
                quantization_config=q_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,                 # Streams weights more carefully
                offload_buffers=True,                   # Offloads buffers to CPU during load
                torch_dtype=torch.bfloat16
            )
            self.processor = AutoProcessor.from_pretrained(kwargs.get("model_id"), trust_remote_code=True)
    
            # Clear again after loading
            torch.cuda.empty_cache()
            gc.collect()
    
        if kwargs.get("append_speech_to_end") and self.audio_model is None:
            self.audio_model = whisper.load_model("base")
        os.makedirs(output_path, exist_ok=True)
        final_instruction = kwargs.get("llm_instruction").replace("{trigger}", kwargs.get("trigger_word"))
    
        # Batch Folder Mode: process all images and videos in folder
        if mode == "Batch Folder Mode":
            files = []
            for fname in os.listdir(input_path):
                full_path = os.path.join(input_path, fname)
                if os.path.isfile(full_path):
                    ext = fname.lower()
                    if ext.endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                        files.append((full_path, "image"))
                    elif ext.endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                        files.append((full_path, "video"))
            if not files:
                return ("❌ ERROR: No supported images or videos found in batch folder.",)
    
            for i, (file_path, file_type) in enumerate(files):
                if self.check_interrupt(): return (f"❌ STOPPED AT {i+1}/{len(files)}",)
    
                if file_type == "image":
                    img = cv2.imread(file_path)
                    if img is None: continue
                    proc_img = self.smart_resize(img, int(kwargs.get("target_resolution")))
                    pil_img = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
                    caption = self.generate_caption(device, pil_img, final_instruction, kwargs.get("trigger_word"), token_limit)
                    name = os.path.splitext(os.path.basename(file_path))[0]
                    cv2.imwrite(os.path.join(output_path, f"{name}.png"), proc_img)
                    with open(os.path.join(output_path, f"{name}.txt"), "w", encoding="utf-8") as f:
                        f.write(caption)
                else:  # video
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened(): continue
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    orig_name = os.path.splitext(os.path.basename(file_path))[0]
                    frames_per_seg = int(fps * kwargs.get("video_segment_seconds"))
    
                    for s in range(kwargs.get("video_max_segments")):
                        if self.check_interrupt(): cap.release(); break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, s * frames_per_seg * kwargs.get("segment_skip"))
                        seg_frames = []
                        for f_idx in range(frames_per_seg):
                            ret, frame = cap.read()
                            if not ret: break
                            seg_frames.append(self.smart_resize(frame, int(kwargs.get("target_resolution"))))
                        if not seg_frames: break
    
                        file_base = f"{orig_name}_seg_{s:04d}"
                        temp_wav = os.path.join(output_path, "temp.wav")
                        st = (s * frames_per_seg * kwargs.get("segment_skip")) / fps
                        result = subprocess.run(['ffmpeg', '-y', '-ss', str(st), '-t', str(kwargs.get("video_segment_seconds")), '-i', file_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_wav], capture_output=True, text=True)
                        if result.returncode != 0:
                            print(f"⚠️ FFmpeg audio extraction failed: {result.stderr}")
                        mid_pil = Image.fromarray(cv2.cvtColor(seg_frames[len(seg_frames)//2], cv2.COLOR_BGR2RGB))
                        desc = self.generate_caption(device, mid_pil, final_instruction, kwargs.get("trigger_word"), token_limit)
                        if kwargs.get("append_speech_to_end") and os.path.exists(temp_wav):
                            speech = self.audio_model.transcribe(temp_wav)['text'].strip()
                            if speech: desc += f". Audio: \"{speech}\""
                        sv = os.path.join(output_path, "silent_temp.mp4")
                        h, w = seg_frames[0].shape[:2]
                        vw = cv2.VideoWriter(sv, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        for f in seg_frames: vw.write(f)
                        vw.release()
                        fv = os.path.join(output_path, f"{file_base}.mp4")
                        ffmpeg_cmd = ['ffmpeg', '-y', '-i', sv]
                        if kwargs.get("include_audio_in_video") and os.path.exists(temp_wav):
                            ffmpeg_cmd += ['-i', temp_wav, '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac']
                        ffmpeg_cmd += ['-filter:v', f'fps=fps={kwargs.get("target_fps")}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-shortest', fv]
                        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            print(f"⚠️ FFmpeg video encoding failed: {result.stderr}")
                        with open(os.path.join(output_path, f"{file_base}.txt"), "w", encoding="utf-8") as f: f.write(desc)
                        if os.path.exists(sv): os.remove(sv)
                        if os.path.exists(temp_wav): os.remove(temp_wav)
                        torch.cuda.empty_cache()   # Clear after each segment
                        gc.collect()
                    cap.release()
                torch.cuda.empty_cache()       # Clear after each file
                gc.collect()
            torch.cuda.empty_cache()
            return ("✅ Batch Folder Done – images & videos processed!",)
    
        # Single Video File Mode
        else:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                cap.release()
                return (f"❌ ERROR: Cannot open video: {input_path}",)
       
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            orig_name = os.path.splitext(os.path.basename(input_path))[0]
            frames_per_seg = int(fps * kwargs.get("video_segment_seconds"))
       
            for s in range(kwargs.get("video_max_segments")):
                if self.check_interrupt(): cap.release(); return (f"❌ STOPPED AT SEG {s}",)
                cap.set(cv2.CAP_PROP_POS_FRAMES, s * frames_per_seg * kwargs.get("segment_skip"))
                seg_frames = []
                for f_idx in range(frames_per_seg):
                    ret, frame = cap.read()
                    if not ret: break
                    seg_frames.append(self.smart_resize(frame, int(kwargs.get("target_resolution"))))
           
                if not seg_frames: break
           
                file_base = f"{orig_name}_seg_{s:04d}"
                temp_wav = os.path.join(output_path, "temp.wav")
                st = (s * frames_per_seg * kwargs.get("segment_skip")) / fps
           
                result = subprocess.run(['ffmpeg', '-y', '-ss', str(st), '-t', str(kwargs.get("video_segment_seconds")), '-i', input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_wav], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"⚠️ FFmpeg audio extraction failed: {result.stderr}")
                mid_pil = Image.fromarray(cv2.cvtColor(seg_frames[len(seg_frames)//2], cv2.COLOR_BGR2RGB))
                desc = self.generate_caption(device, mid_pil, final_instruction, kwargs.get("trigger_word"), token_limit)
                if kwargs.get("append_speech_to_end") and os.path.exists(temp_wav):
                    speech = self.audio_model.transcribe(temp_wav)['text'].strip()
                    if speech: desc += f". Audio: \"{speech}\""
                sv = os.path.join(output_path, "silent_temp.mp4")
                h, w = seg_frames[0].shape[:2]
                vw = cv2.VideoWriter(sv, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                for f in seg_frames: vw.write(f)
                vw.release()
                fv = os.path.join(output_path, f"{file_base}.mp4")
                ffmpeg_cmd = ['ffmpeg', '-y', '-i', sv]
                if kwargs.get("include_audio_in_video") and os.path.exists(temp_wav):
                    ffmpeg_cmd += ['-i', temp_wav, '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac']
                ffmpeg_cmd += ['-filter:v', f'fps=fps={kwargs.get("target_fps")}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-shortest', fv]
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"⚠️ FFmpeg video encoding failed: {result.stderr}")
            
                with open(os.path.join(output_path, f"{file_base}.txt"), "w", encoding="utf-8") as f: f.write(desc)
                if os.path.exists(sv): os.remove(sv)
                if os.path.exists(temp_wav): os.remove(temp_wav)
                torch.cuda.empty_cache()   # Clear after each segment
                gc.collect()
            cap.release()
            torch.cuda.empty_cache()
            gc.collect()
            return ("✅ Video Processing Done",)

# ========================================
# GGUF VERSION - Uses llama-cpp-python for smaller memory footprint
# ========================================

class SeansOmniTagProcessorGGUF:
    """GGUF version of OmniTag - uses llama-cpp-python for lower VRAM usage"""
    DEFAULT_WIDTH = 550
    DEFAULT_HEIGHT = 700

    def __init__(self):
        self.llm = None
        self.chat_handler = None
        self.audio_model = None
        self.current_model_path = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["Batch Folder Mode", "Single Video File"], {
                    "default": "Batch Folder Mode",
                    "tooltip": "Choose mode:\n• Batch Folder Mode → captions all images and videos in the folder\n• Single Video File → processes one video file only"
                }),
                "🖼️📁🎥 Batch Folder": ("STRING", {
                    "default": "C:\\MyExampleBatchFolder",
                    "tooltip": "For Batch Folder Mode: Right-click your folder in File Explorer → Copy as path → paste here (Ctrl+V).\nExample: C:\\MyExampleBatchFolder\nThis mode captions ALL images and processes ALL videos inside the folder."
                }),
                "🎥 Single Video File": ("STRING", {
                    "default": "C:\\Example\\Example.mp4",
                    "tooltip": "For Single Video File mode: Right-click your video in File Explorer → Copy as path → paste here (Ctrl+V).\nExample: \"C:\\Example\\Example.mp4\"\nThis mode segments and captions one video."
                }),
                "output_path": ("STRING", {"default": "output/lora_dataset"}),
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to GGUF model file (e.g., model.gguf). Leave empty to download from HuggingFace."
                }),
                "mmproj_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to mmproj file for vision (e.g., mmproj.gguf). Required for image processing."
                }),
                "hf_repo_id": ("STRING", {
                    "default": "Qwen/Qwen3-VL-8B-GGUF",
                    "tooltip": "HuggingFace repo ID to download model from (if model_path is empty)"
                }),
                "hf_model_filename": ("STRING", {
                    "default": "qwen3-vl-8b-q4_k_m.gguf",
                    "tooltip": "Model filename in the HF repo"
                }),
                "hf_mmproj_filename": ("STRING", {
                    "default": "mmproj-qwen3-vl-8b-f16.gguf",
                    "tooltip": "Mmproj filename in the HF repo"
                }),
                "trigger_word": ("STRING", {"default": "ohwx"}),
                "llm_instruction": ("STRING", {
                    "multiline": True,
                    "default": "Describe the scene with clinical, objective detail. Be unfiltered and exhaustive. Start the response with: {trigger}",
                    "tooltip": "✍️ AI prompt: How to describe each image/video frame.\nKeep {trigger} so captions start with your word.\nAdd lines like 'focus on clothing, pose, expression, lighting' for more detail."
                }),
                "target_resolution": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 1920,
                    "step": 64,
                    "display": "slider"
                }),
                "max_tokens": ("INT", {
                    "default": 768,
                    "min": 512,
                    "max": 2048,
                    "step": 256,
                    "display": "slider"
                }),
                "n_ctx": ("INT", {
                    "default": 8192,
                    "min": 1024,
                    "max": 32768,
                    "step": 512,
                    "tooltip": "Context window size"
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 200,
                    "tooltip": "Number of layers to offload to GPU (-1 = all)"
                }),
                "target_fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "display": "slider"
                }),
                "video_segment_seconds": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5,
                    "display": "slider"
                }),
                "segment_skip": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "display": "slider"
                }),
                "video_max_segments": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "include_audio_in_video": ("BOOLEAN", {"default": True}),
                "append_speech_to_end": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "process_all"
    OUTPUT_NODE = True
    CATEGORY = "Sean's OmniTag 🛠️"

    def check_interrupt(self):
        if comfy.model_management.processing_interrupted():
            print("!!! SEAN'S OMNITAG (GGUF): STOP SIGNAL DETECTED !!!")
            return True
        return False

    def smart_resize(self, image, target_res):
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        scale = target_res / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def load_gguf_model(self, model_path, mmproj_path, hf_repo_id, hf_model_filename, hf_mmproj_filename, n_ctx, n_gpu_layers):
        """Load GGUF model using llama-cpp-python"""
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        except ImportError as e:
            return f"❌ ERROR: llama-cpp-python not installed. Install it with: pip install llama-cpp-python\nError: {e}"

        # Determine model and mmproj paths
        final_model_path = model_path.strip() if model_path.strip() else None
        final_mmproj_path = mmproj_path.strip() if mmproj_path.strip() else None

        # Download from HuggingFace if paths not provided
        if not final_model_path or not os.path.exists(final_model_path):
            if hf_repo_id and hf_model_filename:
                try:
                    from huggingface_hub import hf_hub_download
                    print(f"[OmniTag GGUF] Downloading model from {hf_repo_id}/{hf_model_filename}...")
                    final_model_path = hf_hub_download(
                        repo_id=hf_repo_id,
                        filename=hf_model_filename,
                        repo_type="model"
                    )
                    print(f"[OmniTag GGUF] Model downloaded to: {final_model_path}")
                except Exception as e:
                    return f"❌ ERROR: Failed to download model: {e}"
            else:
                return "❌ ERROR: No model_path provided and no HuggingFace repo specified"

        if not final_mmproj_path or not os.path.exists(final_mmproj_path):
            if hf_repo_id and hf_mmproj_filename:
                try:
                    from huggingface_hub import hf_hub_download
                    print(f"[OmniTag GGUF] Downloading mmproj from {hf_repo_id}/{hf_mmproj_filename}...")
                    final_mmproj_path = hf_hub_download(
                        repo_id=hf_repo_id,
                        filename=hf_mmproj_filename,
                        repo_type="model"
                    )
                    print(f"[OmniTag GGUF] Mmproj downloaded to: {final_mmproj_path}")
                except Exception as e:
                    return f"❌ ERROR: Failed to download mmproj: {e}"
            else:
                return "❌ ERROR: No mmproj_path provided and no HuggingFace repo specified"

        # Check if model is already loaded
        if self.llm is not None and self.current_model_path == final_model_path:
            print(f"[OmniTag GGUF] Model already loaded: {final_model_path}")
            return None

        # Clear previous model
        if self.llm is not None:
            print("[OmniTag GGUF] Clearing previous model...")
            self.llm = None
            self.chat_handler = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Load the model
        try:
            print(f"[OmniTag GGUF] Loading model: {final_model_path}")
            print(f"[OmniTag GGUF] Loading mmproj: {final_mmproj_path}")
            print(f"[OmniTag GGUF] Context size: {n_ctx}, GPU layers: {n_gpu_layers}")

            # Initialize chat handler for vision
            self.chat_handler = Qwen3VLChatHandler(
                clip_model_path=final_mmproj_path,
                image_max_tokens=4096,
                verbose=False
            )

            # Load the LLM
            self.llm = Llama(
                model_path=final_model_path,
                chat_handler=self.chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_batch=512,
                verbose=False
            )

            self.current_model_path = final_model_path
            print("[OmniTag GGUF] Model loaded successfully!")
            return None

        except Exception as e:
            return f"❌ ERROR: Failed to load GGUF model: {e}"

    def generate_caption(self, pil_img, instruction, trigger, token_limit):
        """Generate caption using GGUF model"""
        if self.llm is None:
            return f"{trigger}, error: model not loaded"

        try:
            # Convert PIL image to base64 for llama-cpp-python
            import io
            import base64
            
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Create messages for chat completion
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful vision-language assistant. Answer directly with the final answer only."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ]
                }
            ]

            # Generate caption
            result = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=token_limit,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.12,
                stop=["<|im_end|>", "<|im_start|>"]
            )

            caption = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            # Fallback if caption is lazy
            if not caption or caption.lower() == trigger.lower() or len(caption) < 20:
                print(f"⚠️ Lazy caption detected. Retrying for {trigger}...")
                result = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max(512, token_limit // 2),
                    temperature=0.5,
                    repeat_penalty=1.25,
                    stop=["<|im_end|>", "<|im_start|>"]
                )
                caption = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            # Final fallback
            if not caption or caption.lower() == trigger.lower():
                caption = f"{trigger}, scene description unavailable"

            return caption

        except Exception as e:
            print(f"❌ ERROR generating caption: {e}")
            return f"{trigger}, error generating caption"

    def process_all(self, **kwargs):
        if not check_ffmpeg():
            return ("❌ ERROR: FFmpeg not found! Install + add to PATH for video/audio.",)

        mode = kwargs.get("mode")
        if mode == "Batch Folder Mode":
            input_path = kwargs.get("🖼️📁🎥 Batch Folder").strip().replace('"', '').replace("'", "").replace("\\", "/")
            if not input_path or not os.path.isdir(input_path):
                return ("❌ ERROR: Valid batch folder required in Batch Folder Mode.",)
        else:
            input_path = kwargs.get("🎥 Single Video File").strip().replace('"', '').replace("'", "").replace("\\", "/")
            if not input_path or not os.path.isfile(input_path):
                return ("❌ ERROR: Valid video file required in Single Video File mode.",)

        output_path = kwargs.get("output_path").strip().replace('"', '').replace("'", "").replace("\\", "/")
        token_limit = int(kwargs.get("max_tokens"))
        if not os.path.exists(input_path):
            return (f"❌ ERROR: Path not found: {input_path}",)

        # Clear VRAM before loading model
        torch.cuda.empty_cache()
        gc.collect()

        # Load GGUF model
        if self.llm is None:
            error = self.load_gguf_model(
                model_path=kwargs.get("model_path"),
                mmproj_path=kwargs.get("mmproj_path"),
                hf_repo_id=kwargs.get("hf_repo_id"),
                hf_model_filename=kwargs.get("hf_model_filename"),
                hf_mmproj_filename=kwargs.get("hf_mmproj_filename"),
                n_ctx=int(kwargs.get("n_ctx")),
                n_gpu_layers=int(kwargs.get("n_gpu_layers"))
            )
            if error:
                return (error,)

            # Clear again after loading
            torch.cuda.empty_cache()
            gc.collect()

        if kwargs.get("append_speech_to_end") and self.audio_model is None:
            self.audio_model = whisper.load_model("base")

        os.makedirs(output_path, exist_ok=True)
        final_instruction = kwargs.get("llm_instruction").replace("{trigger}", kwargs.get("trigger_word"))

        # Batch Folder Mode: process all images and videos in folder
        if mode == "Batch Folder Mode":
            files = []
            for fname in os.listdir(input_path):
                full_path = os.path.join(input_path, fname)
                if os.path.isfile(full_path):
                    ext = fname.lower()
                    if ext.endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                        files.append((full_path, "image"))
                    elif ext.endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
                        files.append((full_path, "video"))
            if not files:
                return ("❌ ERROR: No supported images or videos found in batch folder.",)

            for i, (file_path, file_type) in enumerate(files):
                if self.check_interrupt(): return (f"❌ STOPPED AT {i+1}/{len(files)}",)

                if file_type == "image":
                    img = cv2.imread(file_path)
                    if img is None: continue
                    proc_img = self.smart_resize(img, int(kwargs.get("target_resolution")))
                    pil_img = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
                    caption = self.generate_caption(pil_img, final_instruction, kwargs.get("trigger_word"), token_limit)
                    name = os.path.splitext(os.path.basename(file_path))[0]
                    cv2.imwrite(os.path.join(output_path, f"{name}.png"), proc_img)
                    with open(os.path.join(output_path, f"{name}.txt"), "w", encoding="utf-8") as f:
                        f.write(caption)
                else:  # video
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened(): continue
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    orig_name = os.path.splitext(os.path.basename(file_path))[0]
                    frames_per_seg = int(fps * kwargs.get("video_segment_seconds"))

                    for s in range(kwargs.get("video_max_segments")):
                        if self.check_interrupt(): cap.release(); break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, s * frames_per_seg * kwargs.get("segment_skip"))
                        seg_frames = []
                        for f_idx in range(frames_per_seg):
                            ret, frame = cap.read()
                            if not ret: break
                            seg_frames.append(self.smart_resize(frame, int(kwargs.get("target_resolution"))))
                        if not seg_frames: break

                        file_base = f"{orig_name}_seg_{s:04d}"
                        temp_wav = os.path.join(output_path, "temp.wav")
                        st = (s * frames_per_seg * kwargs.get("segment_skip")) / fps
                        subprocess.run(['ffmpeg', '-y', '-ss', str(st), '-t', str(kwargs.get("video_segment_seconds")), '-i', file_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_wav], capture_output=True)
                        mid_pil = Image.fromarray(cv2.cvtColor(seg_frames[len(seg_frames)//2], cv2.COLOR_BGR2RGB))
                        desc = self.generate_caption(mid_pil, final_instruction, kwargs.get("trigger_word"), token_limit)
                        if kwargs.get("append_speech_to_end") and os.path.exists(temp_wav):
                            speech = self.audio_model.transcribe(temp_wav)['text'].strip()
                            if speech: desc += f". Audio: \"{speech}\""
                        sv = os.path.join(output_path, "silent_temp.mp4")
                        h, w = seg_frames[0].shape[:2]
                        vw = cv2.VideoWriter(sv, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        for f in seg_frames: vw.write(f)
                        vw.release()
                        fv = os.path.join(output_path, f"{file_base}.mp4")
                        ffmpeg_cmd = ['ffmpeg', '-y', '-i', sv]
                        if kwargs.get("include_audio_in_video") and os.path.exists(temp_wav):
                            ffmpeg_cmd += ['-i', temp_wav, '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac']
                        ffmpeg_cmd += ['-filter:v', f'fps=fps={kwargs.get("target_fps")}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-shortest', fv]
                        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            print(f"⚠️ FFmpeg video encoding failed: {result.stderr}")
                        with open(os.path.join(output_path, f"{file_base}.txt"), "w", encoding="utf-8") as f: f.write(desc)
                        if os.path.exists(sv): os.remove(sv)
                        if os.path.exists(temp_wav): os.remove(temp_wav)
                        torch.cuda.empty_cache()
                        gc.collect()
                    cap.release()
                torch.cuda.empty_cache()
                gc.collect()
            torch.cuda.empty_cache()
            return ("✅ Batch Folder Done – images & videos processed!",)

        # Single Video File Mode
        else:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                cap.release()
                return (f"❌ ERROR: Cannot open video: {input_path}",)
       
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            orig_name = os.path.splitext(os.path.basename(input_path))[0]
            frames_per_seg = int(fps * kwargs.get("video_segment_seconds"))
       
            for s in range(kwargs.get("video_max_segments")):
                if self.check_interrupt(): cap.release(); return (f"❌ STOPPED AT SEG {s}",)
                cap.set(cv2.CAP_PROP_POS_FRAMES, s * frames_per_seg * kwargs.get("segment_skip"))
                seg_frames = []
                for f_idx in range(frames_per_seg):
                    ret, frame = cap.read()
                    if not ret: break
                    seg_frames.append(self.smart_resize(frame, int(kwargs.get("target_resolution"))))
           
                if not seg_frames: break
           
                file_base = f"{orig_name}_seg_{s:04d}"
                temp_wav = os.path.join(output_path, "temp.wav")
                st = (s * frames_per_seg * kwargs.get("segment_skip")) / fps
           
                subprocess.run(['ffmpeg', '-y', '-ss', str(st), '-t', str(kwargs.get("video_segment_seconds")), '-i', input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_wav], capture_output=True)
                mid_pil = Image.fromarray(cv2.cvtColor(seg_frames[len(seg_frames)//2], cv2.COLOR_BGR2RGB))
                desc = self.generate_caption(mid_pil, final_instruction, kwargs.get("trigger_word"), token_limit)
                if kwargs.get("append_speech_to_end") and os.path.exists(temp_wav):
                    speech = self.audio_model.transcribe(temp_wav)['text'].strip()
                    if speech: desc += f". Audio: \"{speech}\""
                sv = os.path.join(output_path, "silent_temp.mp4")
                h, w = seg_frames[0].shape[:2]
                vw = cv2.VideoWriter(sv, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                for f in seg_frames: vw.write(f)
                vw.release()
                fv = os.path.join(output_path, f"{file_base}.mp4")
                ffmpeg_cmd = ['ffmpeg', '-y', '-i', sv]
                if kwargs.get("include_audio_in_video") and os.path.exists(temp_wav):
                    ffmpeg_cmd += ['-i', temp_wav, '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac']
                ffmpeg_cmd += ['-filter:v', f'fps=fps={kwargs.get("target_fps")}', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-shortest', fv]
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"⚠️ FFmpeg video encoding failed: {result.stderr}")
            
                with open(os.path.join(output_path, f"{file_base}.txt"), "w", encoding="utf-8") as f: f.write(desc)
                if os.path.exists(sv): os.remove(sv)
                if os.path.exists(temp_wav): os.remove(temp_wav)
                torch.cuda.empty_cache()
                gc.collect()
            cap.release()
            torch.cuda.empty_cache()
            gc.collect()
            return ("✅ Video Processing Done",)


NODE_CLASS_MAPPINGS = {
    "SeansOmniTagProcessor": SeansOmniTagProcessor,
    "SeansOmniTagProcessorGGUF": SeansOmniTagProcessorGGUF
}