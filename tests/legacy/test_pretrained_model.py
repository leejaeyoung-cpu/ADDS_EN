import torch

model_path = "models/pretrained/swin_unetr_pretrained.pt"

print("[*] Loading pretrained model...")
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print(f"[OK] Model loaded successfully!")
print(f"Type: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print(f"Dict keys: {list(checkpoint.keys())}")
else:
    print(f"State dict with {len(checkpoint)} parameters")
    
print("\n[SUCCESS] Pretrained model is ready to use!")
