import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

from ldm.models.diffusion.ddim import DDIMSampler

from utils import seed_everything
from utils import preprocess_edge

from SD_model_wrapper import SDModelWrapper
from patched_unet import PatchedUNet

device = "cuda" if torch.cuda.is_available() else "cpu"

prompts_dict = {
    "B": "sks-b failure",
    "BJ": "sks-bj failure",
    "J": "sks-j failure"
}

class StructuralGenerator:
    def __init__(self, config, base_ckpt, adapter_ckpt):
        self.wrapper = SDModelWrapper(config, base_ckpt)
        
        print(f"[*] Loading Adapter: {adapter_ckpt}")
        ckpt = torch.load(adapter_ckpt, map_location="cpu")
        sd = {k.replace("adapter.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("adapter.")}
        self.wrapper.adapter.load_state_dict(sd)
        
        self.wrapper.to(device)
        original_unet = self.wrapper.model.model.diffusion_model
        self.wrapper.model.model.diffusion_model = PatchedUNet(original_unet)
        self.patched_unet = self.wrapper.model.model.diffusion_model
        
        self.sampler = DDIMSampler(self.wrapper.model)

    def generate(self, image_path, label, output_path, device, dilation, num_samples=1):
        tensor = preprocess_edge(image_path, device, dilation)
        
        base = prompts_dict[label]
        
        with torch.no_grad():
            features = self.wrapper.adapter(tensor)
            
            self.patched_unet.set_adapter_features(features)

            for i in range(num_samples):
                prompt = base
                
                c = self.wrapper.model.get_learned_conditioning([prompt])
                
                samples, _ = self.sampler.sample(S=50, conditioning=c, batch_size=1, shape=[4, 64, 64], 
                                               verbose=False, unconditional_guidance_scale=9.0)
                
                x_samples = self.wrapper.model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
                
                img_np = (255. * x_samples[0].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
                save_name = output_path.stem + f"_n{i}.png" if num_samples > 1 else output_path.name
                cv2.imwrite(str(output_path.parent / save_name), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                print(f"Saved: {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structural Failure Generation Inference")

    parser.add_argument("--config", required=True, help="path to v1-inference.yaml")
    parser.add_argument("--ckpt", required=True, help="path to base SD model checkpoint")
    parser.add_argument("--adapter", required=True, help="path to adapter checkpoint")
    parser.add_argument("--input", required=True, help="path to input edge image OR directory")
    parser.add_argument("--output", default="./results", help="output directory")
    parser.add_argument("--label", required=True, choices=["B", "BJ", "J"], help="failure mode")
    parser.add_argument("--device", choices=["cuda", "cpu"], default='cuda', help="device")
    parser.add_argument("--dilation", choices=[0, 10, 25, 50, 75, 85, 100], default=0, help="dilation: larger - thicker edges")
    parser.add_argument("--n", type=int, default=1, help="number of images to generate per input")
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    print(f"[*] Seed set to: {args.seed}")

    engine = StructuralGenerator(args.config, args.ckpt, args.adapter)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        save_path = out_dir / f"{args.label}_{input_path.stem}.png"
        engine.generate(input_path, args.label, save_path, args.device, args.dilation, args.n)
        
    elif input_path.is_dir():
        files = list(input_path.glob("*.png"))
        print(f"Found {len(files)} .PNG images in directory.")
        for f in files:
            if "edge" not in f.name: continue 
            save_path = out_dir / f"{args.label}_{f.stem}.png"
            engine.generate(f, args.label, save_path, args.device, args.dilation, args.n)