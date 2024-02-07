pip install -r requirements.txt
python3 -m spacy download en_core_web_md
mkdir store
wget -c https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors?download=true -O store/v2-1_512-ema-pruned.safetensors
wget -c https://huggingface.co/luping-liu/Detector_Guidance/resolve/main/detector_guidance_yolo_sd2.ckpt?download=true -O store/detector_guidance_yolo_sd2.ckpt

