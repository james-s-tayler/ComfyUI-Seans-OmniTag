import os
import sys
import subprocess
import io
import base64
import torch
import gc
import cv2
import numpy as np
from PIL import Image
from urllib.parse import urlparse
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
        subprocess.check_call([
            current_python,
            "-m",
            "pip",
            "install",
            "transformers>=4.57.0",
            "accelerate",
            "qwen_vl_utils",
            "huggingface_hub",
            "torchvision",
            "opencv-python",
            "bitsandbytes",
            "openai-whisper",
            "llama-cpp-python",
        ])
        return True
    except:
        return False

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    import whisper
except ImportError:
    install_dependencies()
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    import whisper

class SeansOmniTagProcessor:
    DEFAULT_WIDTH = 550
    DEFAULT_HEIGHT = 700

    def __init__(self):
        self.model, self.processor = None, None
        self.audio_model = None
        self.backend = None

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
                "model_id": ("STRING", {"default": "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it-GGUF"}),
                "gguf_filename": ("STRING", {"default": "Qwen3-VL-8B-Abliterated-Caption-it.Q8_0.gguf"}),
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

    def _parse_hf_url(self, model_id, gguf_filename):
        if not model_id.startswith("http"):
            return model_id, gguf_filename

        parsed = urlparse(model_id)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            repo_id = f"{parts[0]}/{parts[1]}"
            if "blob" in parts and len(parts) > parts.index("blob") + 2:
                gguf_filename = parts[-1]
            return repo_id, gguf_filename
        return model_id, gguf_filename

    def _encode_pil_to_data_url(self, pil_img):
        png_bytes = io.BytesIO()
        pil_img.save(png_bytes, format="PNG")
        encoded = base64.b64encode(png_bytes.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _load_model(self, model_id, gguf_filename):
        parsed_model_id, parsed_gguf_filename = self._parse_hf_url(model_id, gguf_filename)

        if parsed_model_id.endswith(".gguf"):
            self.backend = "gguf"
            self.model = Llama(model_path=parsed_model_id, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            return

        if parsed_gguf_filename.lower().endswith(".gguf"):
            self.backend = "gguf"
            model_path = hf_hub_download(repo_id=parsed_model_id, filename=parsed_gguf_filename)
            self.model = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            return

        self.backend = "transformers"
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            parsed_model_id,
            quantization_config=q_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_buffers=True,
            torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(parsed_model_id, trust_remote_code=True)

    def generate_caption(self, device, pil_img, instruction, trigger, token_limit):
        if self.backend == "gguf":
            response = self.model.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": self._encode_pil_to_data_url(pil_img)}}
                    ]
                }],
                max_tokens=token_limit,
                temperature=0.7,
                top_p=0.9,
            )
            caption = response["choices"][0]["message"]["content"].strip()
            return caption if caption else trigger

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
            caption = f"{trigger}, a cinematic scene featuring a young woman in her mid-20s with long dark wavy hair, fair smooth skin, striking dark eyes, and a playful smile."
    
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
            self._load_model(kwargs.get("model_id"), kwargs.get("gguf_filename"))
    
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
                        subprocess.run(['ffmpeg', '-y', '-ss', str(st), '-t', str(kwargs.get("video_segment_seconds")), '-i', file_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_wav], capture_output=True)
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
                        subprocess.run(ffmpeg_cmd, capture_output=True)
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
            if not cap.isOpened(): return (f"❌ ERROR: Cannot open video: {input_path}",)
       
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
                subprocess.run(ffmpeg_cmd, capture_output=True)
            
                with open(os.path.join(output_path, f"{file_base}.txt"), "w", encoding="utf-8") as f: f.write(desc)
                if os.path.exists(sv): os.remove(sv)
                if os.path.exists(temp_wav): os.remove(temp_wav)
                torch.cuda.empty_cache()   # Clear after each segment
                gc.collect()
            cap.release()
            torch.cuda.empty_cache()
            gc.collect()
            return ("✅ Video Processing Done",)

NODE_CLASS_MAPPINGS = {"SeansOmniTagProcessor": SeansOmniTagProcessor}
