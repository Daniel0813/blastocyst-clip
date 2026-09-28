"""Load a CLIP-style model with our preprocessing.

Preprocessing: pad the image to a square with its own background colour (the mean
of its left/right border strips), then the model's own transform (for BiomedCLIP:
bicubic resize to 224 + normalisation). The default centre crop would cut ~25% of
the width of the 512x384 blastocyst images, and embryos are not always centred.
"""
import numpy as np
import open_clip
import torch
from PIL import Image, ImageFile

# 106 of the 1,100 caption-dataset JPEGs lack their last 0-6 bytes (end marker);
# the pixel data is intact, so load them instead of failing.
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODELS = {
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
}
CONTEXT_LENGTH = {"biomedclip": 256}


def pad_to_square(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w == h:
        return img
    a = np.asarray(img)
    strips = np.concatenate([a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3),
                             a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3)])
    fill = tuple(int(v) for v in strips.mean(0))
    side = max(w, h)
    out = Image.new("RGB", (side, side), fill)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


def load(name: str = "biomedclip", device: str | None = None, weights=None):
    """Return (model, image_transform, tokenize_fn, device).

    weights: optional adapter.pt from src/finetune.py (the fine-tuned parameters only).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = open_clip.create_model_from_pretrained(MODELS[name])
    tokenizer = open_clip.get_tokenizer(MODELS[name])
    if weights is not None:
        state = torch.load(weights, map_location="cpu")
        missing = set(state) - set(dict(model.named_parameters()))
        assert not missing, f"adapter has unknown parameters: {sorted(missing)[:3]}"
        model.load_state_dict(state, strict=False)
        print(f"loaded {len(state)} fine-tuned tensors from {weights}")
    model = model.to(device).eval()

    def transform(img: Image.Image) -> torch.Tensor:
        return preprocess(pad_to_square(img))

    def tokenize(texts: list[str]) -> torch.Tensor:
        return tokenizer(texts, context_length=CONTEXT_LENGTH[name])

    return model, transform, tokenize, device
