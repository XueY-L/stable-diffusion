import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    seed = 42
    config = 'configs/stable-diffusion/v1-inference.yaml'
    ckpt = 'ckpt/v1-5-pruned.ckpt'
    outdir = 'tmp'
    n_samples = batch_size = 1  # 3
    n_rows = batch_size
    n_iter = 1  # 2
    prompt = 'a photograph of a dog riding a horse'
    data = [batch_size * [prompt]]  # [[prompt, prompt, prompt]]
    scale = 7.5
    C = 4  # ddim生成图片的通道数，之后会被vae解码回三通道
    f = 8  # 下采样因子，ddim最后生成图像的分辨率是(H//f, W//f)
    H = W = 512
    ddim_steps = 50
    ddim_eta = 0.0  # 用于控制生成的随机性，0表示完全不随机，>0表示每次采样都添加随机噪声

    seed_everything(seed)

    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)  # ldm.models.diffusion.ddpm.LatentDiffusion (LDM model) 作为sampler的成员

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():  # ?
                all_samples = list()
                for n in trange(n_iter, desc="Sampling"):  # 生成n_iter个batch的图片
                    for prompts in tqdm(data, desc="data"):

                        uc = None  # CFG，值为1.0表明不使用引导采样，只用prompt
                        if scale != 1.0:  # scale表示执行CFG的程度
                            uc = model.get_learned_conditioning(batch_size * [""])  # CLIP编码，空字符串表示无约束
                        
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        # 用CLIP把prompts编码成张量
                        c = model.get_learned_conditioning(prompts)  # [3, 77, 768]
                        
                        shape = [C, H // f, W // f]  # ddim生成的tensor的形状
                        # 采样生成
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)  # None传参进去后会随机采样一个X_T
                        # VAE解码
                        x_samples_ddim = model.decode_first_stage(samples_ddim)  # [3, 3, 512, 512]
                        x_samples_ddim = torch.clamp(  # 将[-1,1]范围的值映射到[0,1]，后面会转换成像素值0~255
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        all_samples.append(x_samples_ddim)
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                img = Image.fromarray(grid.astype(np.uint8))
                img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()