"""Run pretrained perception, optional native vision-language context and CDM.

Pass a released lifetime checkpoint to reuse its trained core. New input
projections and the semantic reader still need task training. Numeric values,
records and goals below are placeholders for demonstrating the interface.
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
from cdm.multimodal import MultimodalCore, MultimodalEncoder, join_events
from cdm.semantic import SemanticExpert, SemanticReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--instruction", default="Measure the temperature before changing the heater.")
    parser.add_argument("--core-checkpoint", type=Path)
    parser.add_argument("--semantic", action="store_true", help="Use the selected native vision-language model.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    spec = json.loads((Path(__file__).resolve().parents[1] / "configs/cdm_multimodal_v1.json").read_text())
    saved = None if args.core_checkpoint is None else torch.load(
        args.core_checkpoint, map_location="cpu", weights_only=True)
    config = CoreConfig(**saved["config"]) if saved is not None else CoreConfig(
        **spec["core"], record_dim=6, actions=3)
    e = spec["encoder"]
    processor = AutoProcessor.from_pretrained(e["model_id"], revision=e["revision"])
    picture = None
    if args.image is not None:
        with Image.open(args.image) as source:
            picture = source.convert("RGB")
    if args.semantic:
        inputs = {} if picture is None else processor(images=[picture], return_tensors="pt")
    else:
        inputs = processor(text=[args.instruction], images=None if picture is None else [picture],
                           padding="max_length", max_length=e["text_max_length"],
                           truncation=False, return_tensors="pt")
        if inputs["input_ids"].shape[-1] > e["text_max_length"]:
            raise ValueError("Use --semantic for instructions longer than the SigLIP 64-token context.")
    encoder = MultimodalEncoder.from_pretrained(
        numeric_dim=6, width=config.width, image_grid=e["image_grid"],
        model_id=e["model_id"], revision=e["revision"])
    model = MultimodalCore.from_lifetime_checkpoint(saved, encoder) if saved is not None \
        else MultimodalCore(config, torch.tensor([-1., 0., 1.]), encoder)
    model = model.to(args.device).eval()
    inputs = {k: v.to(args.device) for k, v in inputs.items()
              if k in ("pixel_values", "input_ids", "attention_mask")}
    numeric = torch.zeros(1, 6, device=args.device)
    timing = torch.zeros(1, 3, device=args.device)
    state = model.initial(1, args.device)
    semantic_memory = None
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(args.device).startswith("cuda") else nullcontext()
    with torch.no_grad(), amp:
        event = model.encoder(numeric, timing, **inputs)
        # Real integration supplies the original task's observed numeric fields.
        structured = None if saved is None else torch.zeros(1, config.event_dim, device=args.device)
        state = model.observe(state, event, structured)
        scalars = torch.zeros(1, config.scalar_dim, device=args.device)
        goal = torch.zeros(1, device=args.device)
        immediate = model.readout(state, scalars, goal)
        if args.semantic:
            s = spec["semantic_model"]
            semantic_processor = AutoProcessor.from_pretrained(s["model_id"], revision=s["revision"])
            content = ([] if picture is None else [{"type": "image"}]) + [
                {"type": "text", "text": args.instruction}]
            prompt = semantic_processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            semantic_inputs = semantic_processor(
                text=[prompt], images=None if picture is None else [picture],
                return_tensors="pt").to(args.device)
            if semantic_inputs["input_ids"].shape[-1] > s["initial_max_input_tokens"]:
                raise ValueError("This example uses at most 4096 input tokens; supply a shorter event.")
            dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
            expert = SemanticExpert.from_pretrained(
                s["model_id"], s["revision"], dtype=dtype).to(args.device)
            reader = SemanticReader(expert.hidden_size, config.width,
                                    s["context_tokens"], config.heads).to(args.device).eval()
            features = expert.encode(**semantic_inputs)
            semantic_memory = reader.compress(features, semantic_inputs["attention_mask"], state["h"])
            # Zero age here; a runtime uses now minus the source capture time.
            context = reader.at_age(semantic_memory, torch.zeros(1, device=args.device))
            event = join_events(event, context)
        state = model.refine(
            state, event, torch.zeros(1, config.records, config.record_dim, device=args.device),
            torch.ones(1, dtype=torch.long, device=args.device))
        refined = model.readout(state, scalars, goal)
    print(json.dumps({
        "encoder": e["model_id"],
        "encoder_revision": e["revision"],
        "core_initialization": str(args.core_checkpoint) if saved is not None else "random",
        "core_width": config.width,
        "new_connections": "untrained",
        "semantic_model": spec["semantic_model"]["model_id"] if args.semantic else None,
        "semantic_tokens": None if semantic_memory is None else list(semantic_memory.tokens.shape),
        "event_tokens": list(event.tokens.shape),
        "immediate_action_logits": list(immediate[0].shape),
        "refined_action_logits": list(refined[0].shape),
        "workspace": list(state["z"].shape),
    }, indent=2))


if __name__ == "__main__":
    main()
