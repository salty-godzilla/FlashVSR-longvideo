#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, threading, argparse
from queue import Queue
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from einops import rearrange
import cv2
from vidgear.gears import WriteGear, VideoGear

from diffsynth import ModelManager, FlashVSRTinyLongPipeline
from utils.utils import Causal_LQ4x_Proj
from utils.TCDecoder import build_tcdecoder

def tensor2video(frames):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)  # HWC
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0                                              # CHW in [-1,1]
    return t.to(dtype)

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            f"Increase scale."
        )

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))


def init_pipeline():
    print(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([
        "./FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors",
    ])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    LQ_proj_in_path = "./FlashVSR-v1.1/LQ_proj_in.ckpt"
    if os.path.exists(LQ_proj_in_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(LQ_proj_in_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to('cuda')

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, new_latent_channels=16+768)
    mis = pipe.TCDecoder.load_state_dict(torch.load("./FlashVSR-v1.1/TCDecoder.ckpt"), strict=False)
    print(mis)


    pipe.to('cuda'); pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(); pipe.load_models_to_device(["dit","vae"])
    return pipe

class AsyncVideoLoader:
    def __init__(self, stream, w0, h0, scale, tW, tH, dtype):
        self.stream = stream
        self.w0 = w0
        self.h0 = h0
        self.scale = scale
        self.tW = tW
        self.tH = tH
        self.dtype = dtype
        
        self.buffer = {}
        self.read_head = 0
        self.target_head = 0
        self.last_frame = None
        
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = True
        self.error = None
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            while self.running:
                with self.lock:
                    if self.read_head >= self.target_head:
                        self.cond.wait()
                        if not self.running: break
                    
                    if self.read_head >= self.target_head:
                        continue
                        
                f = self.stream.read()
                
                if f is None:
                    if self.last_frame is not None:
                        img = self.last_frame
                    else:
                        img = Image.new('RGB', (self.w0, self.h0))
                else:
                    f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(f_rgb)
                
                current_last_frame = img
                
                img_out = upscale_then_center_crop(img, self.scale, self.tW, self.tH)
                tensor_f = pil_to_tensor_neg1_1(img_out, self.dtype, 'cpu')
                
                with self.lock:
                    self.buffer[self.read_head] = tensor_f
                    self.last_frame = current_last_frame
                    self.read_head += 1
                    self.cond.notify_all()
                    
        except Exception as e:
            with self.lock:
                self.error = e
                self.cond.notify_all()

    def stop(self):
        self.running = False
        with self.lock:
            self.cond.notify_all()
        self.thread.join()

    def set_target(self, end_idx):
        with self.lock:
            target = end_idx + 50
            if target > self.target_head:
                self.target_head = target
                self.cond.notify()

    def get_batch(self, start_idx, end_idx):
        needed = range(start_idx, end_idx)
        max_needed = end_idx - 1
        
        frames = []
        with self.lock:
            # Debug: Check how many frames ahead we are when called
            ahead = self.read_head - end_idx
            print(f"[Debug] get_batch({start_idx}:{end_idx}) | ReadHead: {self.read_head} | Ahead: {ahead}")
            
            while self.read_head <= max_needed:
                if self.error: raise self.error
                self.cond.wait()
            
            for i in needed:
                if i in self.buffer:
                    frames.append(self.buffer[i])
                else:
                    raise RuntimeError(f"Frame {i} missing from buffer")
                    
        return torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)

    def cleanup(self, limit):
        with self.lock:
            keys = list(self.buffer.keys())
            for k in keys:
                if k < limit:
                    del self.buffer[k]

class AsyncVideoWriter:
    def __init__(self, output_path, output_params):
        self.writer = WriteGear(output=output_path, compression_mode=True, logging=False, **output_params)
        self.queue = Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def _worker(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                self.queue.task_done()
                break
            self.writer.write(frame)
            self.queue.task_done()
            
    def write(self, frame):
        self.queue.put(frame)
        
    def close(self):
        self.queue.put(None)
        self.thread.join()
        self.writer.close()

def main():
    RESULT_ROOT = "./results"
    os.makedirs(RESULT_ROOT, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--sparse_ratio', type=float, default=2.0, help='Sparse ratio. Recommended: 1.5 or 2.0. 1.5 → faster; 2.0 → more stable.')
    parser.add_argument('--scale', type=float, required=True, help='Upscale factor')
    args = parser.parse_args()

    inputs = [args.video]
    seed, dtype, device = 0, torch.bfloat16, 'cuda'
    scale = args.scale
    sparse_ratio = args.sparse_ratio
    pipe = init_pipeline()

    for p in inputs:
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        name = os.path.basename(p.rstrip('/'))
        if name.startswith('.'):
            continue
        
        try:
            # 1. Initialize Video Reader
            print(f"Opening video: {p}")
            stream = VideoGear(source=p, backend=cv2.CAP_FFMPEG).start()
            
            # Get Metadata
            w0 = int(stream.stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            h0 = int(stream.stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = stream.stream.stream.get(cv2.CAP_PROP_FPS)
            total = int(stream.stream.stream.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[{name}] Original Resolution: {w0}x{h0} | Original Frames: {total} | FPS: {fps}")

            # 2. Compute Dimensions
            sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
            print(f"[{name}] Scaled (x{scale:.2f}): {sW}x{sH} -> Target (128-multiple): {tW}x{tH}")

            # 3. Compute Target Frames (Padding Logic)
            # Original logic: idx = list(range(total)) + [total-1]*4, then F = largest_8n1_leq(...)
            idx_len = total + 4
            F = largest_8n1_leq(idx_len)
            print(f"[{name}] Target Frames (8n-3): {F-4}")
            
            # 4. Initialize Video Writer
            save_path = os.path.join(RESULT_ROOT, f"FlashVSR_v1.1_Tiny_Long_{name.split('.')[0]}_seed{seed}.mp4")
            output_params = {
                "-input_framerate": fps, 
                "-vcodec": "libx264", 
                "-crf": 16, 
                "-movflags": "+faststart", 
                "-color_primaries": "bt709", 
                "-color_trc": "bt709", 
                "-colorspace": "bt709", 
                "-color_range": "tv"
            }
            writer = AsyncVideoWriter(save_path, output_params)
            
            loader = AsyncVideoLoader(stream, w0, h0, scale, tW, tH, dtype)

            # 5. Define Callbacks
            def get_input_chunk(start_idx, end_idx):
                loader.set_target(end_idx)
                loader.cleanup(start_idx - 100)
                return loader.get_batch(start_idx, end_idx)

            def save_output_chunk(video_tensor):
                # video_tensor: 1 C F_chunk H W
                pil_frames = tensor2video(video_tensor[0])
                
                for pf in pil_frames:
                    # Convert PIL RGB to CV2 BGR
                    frame_bgr = cv2.cvtColor(np.array(pf), cv2.COLOR_RGB2BGR)
                    writer.write(frame_bgr)

            # 6. Run Pipeline
            pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
                # LQ_video removed, use callbacks
                get_input_frames=get_input_chunk,
                put_output_frames=save_output_chunk,
                num_frames=F, 
                height=tH, width=tW, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(tH*tW), 
                kv_ratio=3.0,
                local_range=11,
                color_fix = True,
            )

            # 7. Cleanup
            loader.stop()
            stream.stop()
            writer.close()
            print(f"Saved: {save_path}")

        except Exception as e:
            print(f"[Error] {name}: {e}")
            try: loader.stop()
            except: pass
            try: stream.stop() 
            except: pass
            try: writer.close() 
            except: pass
            continue

    print("Done.")

if __name__ == "__main__":
    main()
