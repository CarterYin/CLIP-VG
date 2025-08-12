import argparse
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F

from models import build_model
from utils.misc import NestedTensor
from utils.box_utils import xywh2xyxy
from models.clip import clip as clip_module


def _preprocess_image(
    image: Image.Image,
    imsize: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    # RGB
    image = image.convert("RGB")
    orig_w, orig_h = image.size

    # resize according to long side
    ratio = float(imsize / float(max(orig_h, orig_w)))
    new_w = int(round(orig_w * ratio))
    new_h = int(round(orig_h * ratio))
    resized = F.resize(image, (new_h, new_w))

    # to tensor + normalize (ImageNet mean/std as in datasets.NormalizeAndPad)
    tensor = F.to_tensor(resized)
    tensor = F.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # center pad to square (same policy as NormalizeAndPad without aug_translate)
    dh = imsize - new_h
    dw = imsize - new_w
    top = int(round(dh / 2.0 - 0.1))
    left = int(round(dw / 2.0 - 0.1))

    out_img = torch.zeros((3, imsize, imsize), dtype=torch.float32)
    out_mask = torch.ones((imsize, imsize), dtype=torch.int)
    out_img[:, top : top + new_h, left : left + new_w] = tensor
    out_mask[top : top + new_h, left : left + new_w] = 0

    meta = {
        "orig_size": (orig_w, orig_h),
        "resized_size": (new_w, new_h),
        "pad_offset": (left, top),
        "ratio": ratio,
        "imsize": imsize,
    }
    return out_img, out_mask, meta


def _tokenize_text(description: str) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = clip_module.tokenize(description)  # shape [1, 77]
    word_ids = tokens.int()  # keep as int tensor
    word_mask = (tokens.clone() > 0).int()  # 1 where token exists
    return word_ids, word_mask


def _build_args(device: str = "cuda", model_name: str = "ViT-B/16", imsize: int = 224, max_query_len: int = 77):
    return SimpleNamespace(
        device=device,
        model=model_name,
        imsize=imsize,
        max_query_len=max_query_len,
        vl_hidden_dim=512 if model_name != "ViT-L/14" and model_name != "ViT-L/14@336px" else 768,
        vl_dropout=0.1,
        vl_nheads=8,
        vl_dim_feedforward=2048,
        vl_enc_layers=6,
        prompt="{pseudo_query}",
    )


def load_pretrained_model(checkpoint_path: str = None, device: str = None, model_name: str = "ViT-B/16", imsize: int = 224):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    args = _build_args(device=device, model_name=model_name, imsize=imsize)
    model = build_model(args)
    model.to(device)
    model.eval()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])  # repo format
    return model, args


@torch.no_grad()
def forward(model: Any, image: Image.Image, description: str) -> Dict:
    device = next(model.parameters()).device
    # use model.imsize if present, else default 224
    imsize = getattr(model, "imsize", 224)
    try:
        imsize = int(imsize)
    except Exception:
        imsize = 224

    img_tensor, img_mask, meta = _preprocess_image(image, imsize)
    word_ids, word_mask = _tokenize_text(description)

    # batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    img_mask = img_mask.unsqueeze(0).to(device)
    word_ids = word_ids.to(device)
    word_mask = word_mask.to(device)

    img_data = NestedTensor(img_tensor, img_mask)
    text_data = NestedTensor(word_ids, word_mask)

    pred_box_cxcywh = model(img_data, text_data)  # shape [B, 4], normalized to [0,1] on padded square
    pred_box_cxcywh = pred_box_cxcywh[0]

    # to xyxy on padded square
    pred_box_xyxy = xywh2xyxy(pred_box_cxcywh)
    S = float(meta["imsize"])
    x0_p, y0_p, x1_p, y1_p = (pred_box_xyxy * torch.tensor([S, S, S, S], device=pred_box_xyxy.device)).tolist()

    # remove padding to resized coords
    left, top = meta["pad_offset"]
    x0_r, y0_r, x1_r, y1_r = x0_p - left, y0_p - top, x1_p - left, y1_p - top
    new_w, new_h = meta["resized_size"]
    x0_r = max(0.0, min(float(new_w), x0_r))
    y0_r = max(0.0, min(float(new_h), y0_r))
    x1_r = max(0.0, min(float(new_w), x1_r))
    y1_r = max(0.0, min(float(new_h), y1_r))

    # map back to original image
    ratio = meta["ratio"]
    x0_o, y0_o, x1_o, y1_o = x0_r / ratio, y0_r / ratio, x1_r / ratio, y1_r / ratio
    orig_w, orig_h = meta["orig_size"]
    x0_o = max(0.0, min(float(orig_w), x0_o))
    y0_o = max(0.0, min(float(orig_h), y0_o))
    x1_o = max(0.0, min(float(orig_w), x1_o))
    y1_o = max(0.0, min(float(orig_h), y1_o))

    result = {
        "box_xyxy": [x0_o, y0_o, x1_o, y1_o],
        "box_xyxy_int": [int(round(x0_o)), int(round(y0_o)), int(round(x1_o)), int(round(y1_o))],
        "box_cxcywh_normalized": pred_box_cxcywh.detach().cpu().tolist(),
        "original_size": [orig_w, orig_h],
        "resized_size": [new_w, new_h],
        "pad_offset": [left, top],
        "ratio": ratio,
        "description": description,
    }
    return result


def _visualize(image: Image.Image, result: Dict, save_path: str) -> None:
    x0, y0, x1, y1 = result["box_xyxy_int"]
    draw = ImageDraw.Draw(image)
    draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0), width=3)
    text = result.get("description", "")
    if text:
        try:
            # Attempt to load a default font; fallback to basic text if not available
            font = ImageFont.load_default()
            draw.text((x0 + 2, max(0, y0 - 12)), text, fill=(255, 0, 0), font=font)
        except Exception:
            draw.text((x0 + 2, max(0, y0 - 12)), text, fill=(255, 0, 0))
    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    image.save(save_path)


def main():
    parser = argparse.ArgumentParser("CLIP-VG demo")
    parser.add_argument("--input_image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_image_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to pretrained fine-tuned checkpoint (.pth)")
    parser.add_argument("--model", type=str, default="ViT-B/16", help="CLIP backbone: ViT-B/16, ViT-B/32, ViT-L/14, ...")
    parser.add_argument("--imsize", type=int, default=224, help="Input size used by the model (must match training)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_args = load_pretrained_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        model_name=args.model,
        imsize=args.imsize,
    )

    image = Image.open(args.input_image_path).convert("RGB")
    result = forward(model, image.copy(), args.prompt)
    _visualize(image, result, args.output_image_path)


if __name__ == "__main__":
    main()


