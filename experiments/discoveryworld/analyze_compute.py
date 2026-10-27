"""Measure conditional compute choices on reference replay, not task return.

All depths start from the same assimilated event. Only the allocator's chosen
state continues. The minimum-loss depth uses the known reference action for
analysis; it is never supplied to an acting policy.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .adaptive_compute import DEPTHS, BELIEF_DIM, from_reader
from .features import load_features, batch_features
from .train_joint_policy import select_state


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train", "validation"), default="validation")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, _, _ = from_reader(saved["base_checkpoint"], args.device)
    model.load_state_dict(saved["model"]); model.eval()
    data = torch.load(args.manifest, map_location="cpu", weights_only=True)[args.split]
    cost = saved["training"]["compute_cost"]
    count = 0
    ce_total = torch.zeros(4, device=args.device)
    correct = torch.zeros(4, device=args.device)
    optimal = torch.zeros(4, device=args.device)
    selected = torch.zeros(4, device=args.device)
    gap = 0.
    for episode in data:
        state = model.digital.initial(1, args.device)
        for row in episode["rows"]:
            obs, actions = batch_features([load_features(row["feature_path"])], args.device)
            evidence = row["belief"].to(args.device).float()[None]
            allocation, logits, states = model.proposals(obs, actions, evidence, state)
            target = torch.tensor([row["target"]], device=args.device)
            ce = torch.stack([F.cross_entropy(logits[:, i], target) for i in range(4)])
            losses = ce + cost * ce.new_tensor(DEPTHS)
            chosen = allocation.argmax(-1)
            state = select_state(states, chosen)
            count += 1; ce_total += ce
            correct += logits[0].argmax(-1).eq(target).float()
            optimal[losses.argmin()] += 1; selected[chosen[0]] += 1
            gap += float(losses[chosen[0]] - losses.min())
    result = dict(scope="teacher_forced_reference_replay; conditional_depths; not_native_task_returns",
                  split=args.split, actions=count, depths=list(DEPTHS), compute_cost=cost,
                  mean_action_ce=(ce_total / count).tolist(), action_accuracy=(correct / count).tolist(),
                  minimum_reference_loss_depth_counts=optimal.int().tolist(),
                  selected_depth_counts=selected.int().tolist(), mean_selection_loss_gap=gap/count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
