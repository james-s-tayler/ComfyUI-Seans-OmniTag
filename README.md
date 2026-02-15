🚨 BIG UPGRADE ALERT – Sean's OmniTag Processor just leveled up from Qwen 2.5 to **Qwen 3**! 🔥 
The old Qwen 2.5 was already savage.  
Now running **Qwen3-VL-8B-Abliterated-Caption-it** → captions are noticeably sharper, more exhaustive, better object counting, spatial reasoning, fewer hallucinations, and even less censored.  
Same VRAM footprint, way better "clinical / unfiltered / zero-BS" detail.

Tired of weak, lazy, censored captions for your LoRAs / datasets?  
Say hello to **uncensored clinical detail** powered by **Qwen3-VL-8B-Abliterated-Caption-it** in ComfyUI.

<img width="344" height="393" alt="image" src="https://github.com/user-attachments/assets/9d692f5e-5806-4db1-8078-628817072c44" />


✨ **What OmniTag does in one click:**

💾 **How to use (super easy on Windows):**

1. Right-click your folder/video in File Explorer  
2. Choose **Copy as path**  
3. Click the text field in OmniTag → Ctrl+V to paste  
4. Load your model with your preferred loader (for GGUF, use the existing **ComfyUI-GGUF** model loader node)
5. Connect that model output into **SeansOmniTagProcessor**
6. Set `processor_id` to the matching Hugging Face processor (default: `Qwen/Qwen2.5-VL-7B-Instruct`)
7. Press Queue Prompt → get PNGs/MP4s + perfect .txt captions ready for training!

🖼️📁 **Batch Folder Mode**  
→ Throw any folder at it (images + videos mixed)  
→ Captions EVERY .jpg/.png/.webp/.bmp  
→ Processes EVERY .mp4/.mov/.avi/.mkv/.webm as segmented clips  
→ Middle-frame Qwen3 captioning + optional Whisper audio transcript appended

🎥 **Single Video File Mode**  
→ Pick one video → splits into short segments  
→ Optional Whisper speech-to-text at the end of every caption

🎛️ **Everything is adjustable sliders**  
• Resolution (256–1920)  
• Max tokens (512–2048)  
• FPS output  
• Segment length (1–30s)  
• Skip frames between segments  
• Max segments (up to 100!)  

🔊 **Audio superpowers**  
• Include original audio in output clips? (Yes/No)  
• Append transcribed speech to caption end? (Yes/No)  

🧠 **Clinical / unfiltered / exhaustive mode by default**  
Starts every caption with your trigger word (default: ohwx)  
Anti-lazy retry + fallback if model tries to be boring

🖼️ How the smart resize works
The image is resized so the longest side (width or height) exactly matches your chosen target resolution (e.g. 768 px), while keeping the original aspect ratio perfectly intact — no stretching or squishing! 😎
The shorter side scales down proportionally, so a tall portrait stays tall, a wide landscape stays wide. Uses high-quality Lanczos interpolation for sharp, clean results.
Example: a 2000×1000 photo → resized to 768 on the long edge → becomes 768×384 (or 384×768 for portrait). Perfect for consistent LoRA training without weird distortions! 📏✨

Perfect for building high-quality LoRA datasets, especially when you want **raw, detailed, uncensored descriptions** without fighting refusal.



Works with 4-bit quantized Qwen3-VL-8B (≈10–14 GB VRAM)  
Model is now provided as an input, so you can reuse models loaded by other nodes (including GGUF workflows).

