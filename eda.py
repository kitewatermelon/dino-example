import hydra
from omegaconf import DictConfig, OmegaConf
# dino
import vision_transformer_dax as vits
from torchvision import transforms as pth_transforms
import requests
from io import BytesIO
from PIL import Image
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 화면 표시 없이 이미지 파일로만 저장
import matplotlib.pyplot as plt
import cv2
import utils
import matplotlib.colors as mcolors


def transform(img, args):
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    return img


def read_image(args):
    if args.image_path.startswith("http"):
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    return img


def load_model(model, checkpoint, device):
    checkpoint = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    teacher_checkpoint = checkpoint["teacher"]
    teacher_checkpoint = {
        k.replace("backbone.", ""): v 
        for k, v in teacher_checkpoint.items()
        if k.startswith("backbone.")
        }
    
    msg = model.load_state_dict(teacher_checkpoint, strict=False)
    model = model.to(device)
    model.eval()

    return model

def get_attn_maps(name, attn_maps):
    def hook(module, input, output):
        # output could be: tensor OR (out, attn) OR list/tuple containing attn
        attn = None
        if isinstance(output, (tuple, list)):
            # find a 4-D tensor (B, heads, N, N)
            for o in output:
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    attn = o
                    break
            # fallback
            if attn is None and len(output) > 1 and isinstance(output[1], torch.Tensor):
                attn = output[1]
        elif isinstance(output, torch.Tensor):
            # no attn returned; maybe module stored it as attribute (see 방법 B)
            if hasattr(module, "last_attn"):
                attn = module.last_attn

        if attn is None:
            return
        attn_maps[name] = attn.detach().cpu()
    return hook

def visualize_compare_all_heads(img_tensor, attn_original, attn_dax, patch_size, save_path="outputs/compare_attention_heads.png"):
    """
    img_tensor     : (B, C, H, W) torch.Tensor
    attn_original  : (B, heads, N, N) torch.Tensor
    attn_dax       : (B, heads, N, N) torch.Tensor
    patch_size     : int, 모델의 patch size
    """
    # 배치 1 가정
    img_H, img_W = img_tensor.shape[2], img_tensor.shape[3]
    print(f"Tensor image size: ({img_H}, {img_W}), patch size: {patch_size}")

    attn_original = attn_original[0]  # (heads, N, N)
    attn_dax = attn_dax[0]            # (heads, N, N)
    num_heads = attn_original.shape[0]

    fig, axes = plt.subplots(4, num_heads, figsize=(4 * num_heads, 16))

    w_featmap = img_H // patch_size
    h_featmap = img_W // patch_size

    def attn_to_map(attn_head):
        # CLS token 제외
        cls_attn = attn_head[0, 1:]  # (N-1,)
        assert cls_attn.shape[0] == w_featmap * h_featmap, \
            f"cls_attn {cls_attn.shape[0]} vs feature map {w_featmap*h_featmap}"

        cls_attn_map = cls_attn.reshape(w_featmap, h_featmap).detach().cpu().numpy()
        # Tensor 이미지를 PIL로 바꾸기 위해 transpose
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)  # 0~1 정규화

        # 원본 크기로 resize
        attn_map_resized = cv2.resize(cls_attn_map, (img_W, img_H))
        # 0~1 정규화
        attn_map_resized = (attn_map_resized - attn_map_resized.min()) / (
            attn_map_resized.max() - attn_map_resized.min() + 1e-6
        )
        return attn_map_resized, img_np

    for h in range(num_heads):
        # Original
        map_orig, img_np = attn_to_map(attn_original[h])
        axes[0, h].imshow(map_orig, cmap="viridis")
        axes[0, h].set_title(f"Orig Head {h}")
        axes[0, h].axis("off")

        axes[1, h].imshow(img_np)
        axes[1, h].imshow(map_orig, cmap="jet", alpha=0.5)
        axes[1, h].set_title(f"Orig Overlay {h}")
        axes[1, h].axis("off")

        # DAX
        map_dax, _ = attn_to_map(attn_dax[h])
        axes[2, h].imshow(map_dax, cmap="viridis")
        axes[2, h].set_title(f"DAX Head {h}")
        axes[2, h].axis("off")

        axes[3, h].imshow(img_np)
        axes[3, h].imshow(map_dax, cmap="jet", alpha=0.5)
        axes[3, h].set_title(f"DAX Overlay {h}")
        axes[3, h].axis("off")

    plt.suptitle("Attention Heads Comparison", fontsize=16)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()

def visualize_all_attn_maps(attn_maps, img_tensor, model_config, save_path="all_attn_maps.png"):
    """
    모든 attn_block(0~11)의 attention map을 행별로 저장하는 함수.
    각 행: 1개 원본 이미지 + num_heads 개의 head map (정규화 적용)

    Args:
        attn_maps (dict): attn_block_x 키를 포함한 attention map dict
        img_tensor (torch.Tensor): 입력 이미지 (1, C, H, W)
        model_config: patch_size 속성을 가진 config
        save_path (str): 저장할 파일 경로
    """
    num_blocks = 12  # attn_block_0 ~ attn_block_11
    patch_size = model_config.patch_size
    h_featmap = img_tensor.shape[2] // patch_size
    w_featmap = img_tensor.shape[3] // patch_size

    # 원본 이미지 변환
    orig_img = img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()

    # 정규화 해제 (imagenet 기준)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    orig_img = (orig_img * std + mean)
    orig_img = np.clip(orig_img, 0, 1)  # 0~1로 클리핑

    # 첫 번째 block에서 head 개수 확인
    sample_attn = attn_maps["attn_block_0"][0]
    num_heads = sample_attn.shape[0]

    # 전체 subplot 크기 계산 (행 = 블록, 열 = 1+head수)
    fig, axes = plt.subplots(num_blocks, num_heads + 1,
                             figsize=(3 * (num_heads + 1), 3 * num_blocks))

    if num_blocks == 1:
        axes = [axes]  # 1행일 경우에도 2차원 배열처럼 다루기

    for block_idx in range(num_blocks):
        attn = attn_maps[f"attn_block_{block_idx}"][0]  # (num_heads, N, N)
        cls_attn = attn[:, 0, 1:].reshape(num_heads, h_featmap, w_featmap)

        # (0) 원본 이미지
        ax = axes[block_idx, 0]
        ax.imshow(orig_img)
        ax.set_title(f"Block {block_idx}\nOriginal")
        ax.axis("off")

        # (1~num_heads) Attention Map (normalize 0~1)
        for h in range(num_heads):
            attn_map = cls_attn[h].detach().cpu().numpy()
            norm_attn = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            ax = axes[block_idx, h + 1]
            ax.imshow(norm_attn, cmap="grey", norm=mcolors.Normalize(vmin=0, vmax=1))
            ax.set_title(f"Head {h}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Attention maps saved to {save_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_config = cfg.model
    image_config = cfg.image.image
    patch_size = int(model_config.patch_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    img_pil = read_image(image_config)
    img_tensor = transform(img_pil, image_config).unsqueeze(0).to(device)

    original_attn_maps = {}
    original = vits.vit_small(patch_size=patch_size)
    original = load_model(original, model_config.original_checkpoint, device)
    for i, blk in enumerate(original.blocks):
        blk.attn.register_forward_hook(get_attn_maps(f"attn_block_{i}", original_attn_maps))
    _ = original(img_tensor)    
    visualize_all_attn_maps(original_attn_maps, img_tensor, model_config, save_path=f"{image_config.output_path}_original.png")

    dax_attn_maps = {}
    dax = vits.vit_small(patch_size=patch_size)
    dax = load_model(dax, model_config.dax_checkpoint, device)
    for i, blk in enumerate(dax.blocks):
        blk.attn.register_forward_hook(get_attn_maps(f"attn_block_{i}", dax_attn_maps))
    _ = dax(img_tensor)

    visualize_all_attn_maps(dax_attn_maps, img_tensor, model_config, save_path=f"{image_config.output_path}_dax.png")

    # original_attentions = original.get_last_selfattention(img_tensor)
    # dax_attentions = dax.get_last_selfattention(img_tensor)

    # visualize_compare_all_heads(img_tensor, original_attentions, dax_attentions, patch_size)



if __name__ == "__main__":
    main()
