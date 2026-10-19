"""Load the selected pretrained towers and run the new CDM input/state path.

The CDM projections and decision core are randomly initialized here. This is
an integration example, not a trained policy or a task-performance evaluation.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from cdm.core import CoreConfig
from cdm.multimodal import MultimodalCore, MultimodalEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--instruction", default="Measure the temperature before changing the heater.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    spec = json.loads((Path(__file__).resolve().parents[1] / "configs/cdm_multimodal_v1.json").read_text())
    e = spec["encoder"]
    processor = AutoProcessor.from_pretrained(e["model_id"], revision=e["revision"])
    inputs = processor(text=[args.instruction], images=None if args.image is None else [Image.open(args.image).convert("RGB")],
                       padding="max_length", max_length=e["text_max_length"], truncation=False, return_tensors="pt")
    if inputs["input_ids"].shape[-1] > e["text_max_length"]:
        raise ValueError("Split the instruction into events of at most 64 tokens before encoding")
    encoder = MultimodalEncoder.from_pretrained(numeric_dim=6, width=spec["core"]["width"],
                                                image_grid=e["image_grid"], model_id=e["model_id"], revision=e["revision"])
    config = CoreConfig(**spec["core"], record_dim=6, actions=3)
    model = MultimodalCore(config, torch.tensor([-1., 0., 1.]), encoder).to(args.device).eval()
    inputs = {k: v.to(args.device) for k, v in inputs.items() if k in ("pixel_values", "input_ids", "attention_mask")}
    numeric, timing = torch.zeros(1, 6, device=args.device), torch.zeros(1, 3, device=args.device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(args.device).startswith("cuda") else nullcontext()
    with torch.no_grad(), amp:
        event = model.encoder(numeric, timing, **inputs)
        state = model.observe(model.initial(1, args.device), event)
        scalars, goal = torch.zeros(1, config.scalar_dim, device=args.device), torch.zeros(1, device=args.device)
        immediate = model.readout(state, scalars, goal)
        state = model.refine(state, event, torch.zeros(1, config.records, 6, device=args.device),
                             torch.ones(1, dtype=torch.long, device=args.device))
        refined = model.readout(state, scalars, goal)
    print(json.dumps({"encoder": e["model_id"], "revision": e["revision"],
                      "cdm_weights": "random_initialization", "event_tokens": list(event.tokens.shape),
                      "immediate_action_logits": list(immediate[0].shape),
                      "refined_action_logits": list(refined[0].shape), "workspace": list(state["z"].shape)}, indent=2))


if __name__ == "__main__":
    main()
