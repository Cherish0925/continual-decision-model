#!/usr/bin/env python3
"""Train CDM ablations on sequential contextual recall, using observed cues only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cdm import CDMConfig, ContinualDecisionModel  # noqa: E402

METHODS = ("restart_workspace", "persistent_workspace", "computational_memory")


def episodes(batch_size: int, generator: torch.Generator, device: torch.device,
             query_steps: int) -> dict[str, torch.Tensor]:
    """Every requested action is determined by a cue shown in this same episode."""
    assignments = torch.randint(4, (batch_size, 4), generator=generator)
    reveal_order = torch.rand(batch_size, 4, generator=generator).argsort(dim=1)
    queries = torch.randint(4, (batch_size, query_steps), generator=generator)
    contexts = torch.cat((reveal_order, queries), dim=1)
    goals = torch.randint(2, contexts.shape, generator=generator)
    cue_actions = assignments.gather(1, contexts)
    visible = torch.arange(contexts.shape[1])[None, :] < 4
    visible = visible.expand(batch_size, -1)
    observations = torch.cat((
        nn.functional.one_hot(cue_actions, 4).float() * visible[..., None],
        visible[..., None].float(),
    ), dim=-1)
    return {
        "observations": observations.to(device),
        "environments": nn.functional.one_hot(contexts, 4).float().to(device),
        "goals": nn.functional.one_hot(goals, 2).float().to(device),
        "labels": ((cue_actions + goals) % 4).to(device),
        "visible": visible.to(device),
    }


def run_sequence(model: ContinualDecisionModel, data: dict[str, torch.Tensor],
                 method: str, refinements: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, length = data["labels"].shape
    state = model.initialize(batch_size, device=data["labels"].device)
    losses, final_logits = [], []
    for t in range(length):
        state = model.assimilate(
            state, data["observations"][:, t], data["goals"][:, t],
            data["environments"][:, t],
        )
        if method == "restart_workspace":
            state = model.reset_workspace(state)
        elif method == "computational_memory":
            state, _ = model.read_memory(state)
        for _ in range(refinements):
            state = model.refine(state, steps=1)
            logits, _, _ = model.readout(state)
            losses.append(nn.functional.cross_entropy(logits, data["labels"][:, t]))
        final_logits.append(logits)
        if method == "computational_memory" and t < 4:
            state = model.write_memory(state)
    # Gradients span the complete episode, including cue encoding and later recall.
    return torch.stack(losses).mean(), torch.stack(final_logits, dim=1)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model: ContinualDecisionModel, method: str, args: argparse.Namespace,
             device: torch.device) -> dict[str, float | int]:
    model.eval()
    # Disjoint random stream, fixed across methods and training seeds.
    generator = torch.Generator().manual_seed(args.eval_seed)
    totals = {"all": [0, 0], "visible": [0, 0], "query": [0, 0]}
    weighted_loss = 0.0
    synchronize(device)
    start = time.perf_counter()
    for offset in range(0, args.eval_episodes, args.batch_size):
        count = min(args.batch_size, args.eval_episodes - offset)
        data = episodes(count, generator, device, args.query_steps)
        loss, logits = run_sequence(model, data, method, args.refinements)
        correct = logits.argmax(-1).eq(data["labels"])
        masks = {"all": torch.ones_like(correct), "visible": data["visible"],
                 "query": ~data["visible"]}
        for name, mask in masks.items():
            totals[name][0] += int(correct[mask].sum())
            totals[name][1] += int(mask.sum())
        weighted_loss += float(loss) * count
    synchronize(device)
    seconds = time.perf_counter() - start
    result = {f"{name}_accuracy": correct / count
              for name, (correct, count) in totals.items()}
    result.update({"cross_entropy": weighted_loss / args.eval_episodes,
                   "episodes": args.eval_episodes,
                   "query_decisions": totals["query"][1],
                   "evaluation_seconds": seconds,
                   "batched_event_ms": 1000 * seconds / totals["all"][1]})
    return result


def train_one(seed: int, method: str, args: argparse.Namespace,
              device: torch.device, output: Path) -> dict:
    torch.manual_seed(seed)
    config = CDMConfig(observation_dim=5, action_dim=4, goal_dim=2,
                       environment_dim=4, hidden_dim=args.hidden_dim, slots=4,
                       memory_size=8, heads=4)
    model = ContinualDecisionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=0.0)
    # Each method sees exactly the same training episodes for this seed.
    generator = torch.Generator().manual_seed(seed + 100_000)
    synchronize(device)
    start = time.perf_counter()
    curve = []
    model.train()
    for update in range(args.updates):
        data = episodes(args.batch_size, generator, device, args.query_steps)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = run_sequence(model, data, method, args.refinements)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if update == 0 or (update + 1) % 20 == 0 or update + 1 == args.updates:
            curve.append({"update": update + 1, "cross_entropy": float(loss.detach())})
    synchronize(device)
    train_seconds = time.perf_counter() - start
    metrics = evaluate(model, method, args, device)
    result = {
        "seed": seed, "method": method, "model_config": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_updates": args.updates,
        "training_episodes": args.updates * args.batch_size,
        "training_seconds": train_seconds, "learning_curve": curve,
        "held_out": metrics,
        "trained_outputs": ["action_logits"],
        "untrained_outputs": ["activity_logits", "value"],
    }
    if args.save_checkpoints:
        path = output / f"{method}-seed-{seed}.pt"
        torch.save({"model_config": asdict(config), "state_dict": model.state_dict(),
                    "seed": seed, "method": method}, path)
    (output / f"{method}-seed-{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"method": method, "seed": seed,
                      "query_accuracy": metrics["query_accuracy"],
                      "visible_accuracy": metrics["visible_accuracy"],
                      "train_seconds": round(train_seconds, 3)}), flush=True)
    return result


def summarize(results: list[dict]) -> dict:
    summary = {}
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        if not rows:
            continue
        entry = {"seeds": len(rows)}
        for metric in ("all_accuracy", "visible_accuracy", "query_accuracy", "cross_entropy"):
            values = [r["held_out"][metric] for r in rows]
            entry[metric] = {"mean": statistics.mean(values),
                             "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0}
        entry["mean_training_seconds"] = statistics.mean(r["training_seconds"] for r in rows)
        entry["parameters"] = rows[0]["parameters"]
        summary[method] = entry
    if "computational_memory" in summary and "persistent_workspace" in summary:
        by_key = {(r["method"], r["seed"]): r for r in results}
        seeds = sorted({r["seed"] for r in results})
        differences = [by_key[("computational_memory", s)]["held_out"]["query_accuracy"]
                       - by_key[("persistent_workspace", s)]["held_out"]["query_accuracy"]
                       for s in seeds
                       if ("computational_memory", s) in by_key
                       and ("persistent_workspace", s) in by_key]
        summary["memory_minus_persistent_query_accuracy"] = {
            "paired_differences": differences, "mean": statistics.mean(differences),
            "sample_sd": statistics.stdev(differences) if len(differences) > 1 else 0.0,
        }
    return summary


def write_report(results: list[dict], args: argparse.Namespace, output: Path) -> None:
    summary = summarize(results)
    lines = [
        "# Sequential contextual recall: trained neural pilot", "",
        f"{len(args.seeds)} initialization/data seeds; {args.updates} AdamW updates per run; "
        f"{args.batch_size} episodes per update; {args.eval_episodes} held-out episodes per run. "
        f"Each episode has four cue events and {args.query_steps} query events. "
        f"Each event uses {args.refinements} shared Transformer refinements.", "",
        "Values are mean ± sample standard deviation across training seeds. Accuracy is "
        "measured after the final refinement; cross-entropy averages both refinement readouts.", "",
        "| Method | Visible-cue accuracy | Query accuracy | Cross-entropy | Train seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in args.methods:
        row = summary[method]
        visible, query, loss = (row[k] for k in ("visible_accuracy", "query_accuracy", "cross_entropy"))
        lines.append(f"| {method} | {100*visible['mean']:.2f} ± {100*visible['sample_sd']:.2f}% | "
                     f"{100*query['mean']:.2f} ± {100*query['sample_sd']:.2f}% | "
                     f"{loss['mean']:.4f} ± {loss['sample_sd']:.4f} | "
                     f"{row['mean_training_seconds']:.2f} |")
    paired = summary.get("memory_minus_persistent_query_accuracy")
    if paired:
        lines.extend(["", "Memory minus persistent-workspace query accuracy: "
                      f"**{100*paired['mean']:+.2f} percentage points** "
                      f"(paired sample SD {100*paired['sample_sd']:.2f} points)."])
    lines.extend([
        "", f"Each method instantiates the same {results[0]['parameters']:,}-parameter model. "
        "Bypassed memory parameters receive no action-loss gradients in the two memory-free ablations. "
        "The action head is trained; activity and value heads are untrained. "
        "All methods have a recurrent context and receive identical current observations and goals.", "",
        "The comparison matches training episodes, optimizer updates and refinement count. "
        "Memory reading and writing add computation; wall time is reported rather than matched. "
        f"Elapsed times are measurements on `{args.device}` from this run, not real-time environment performance.", "",
        "This pilot measures trainability and cue-conditioned recall. It does not measure "
        "moving-environment control, learned activity allocation, continued parameter adaptation, "
        "or a reduction in computation needed to resume an interrupted plan.", "",
        "Raw measurements and learning curves are in `results.json` and the per-run JSON files. "
        "The exact arguments and software versions are in `config.json`.", "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--refinements", type=int, default=2)
    parser.add_argument("--query-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--eval-seed", type=int, default=900001)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/neural-pilot")
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.refinements, args.query_steps,
           args.eval_episodes, args.threads) < 1:
        parser.error("counts must be positive")
    if args.hidden_dim % 4:
        parser.error("hidden dimension must be divisible by four attention heads")
    if args.eval_seed in {seed + 100_000 for seed in args.seeds}:
        parser.error("evaluation seed must differ from every training data seed")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output_dir"] = str(output)
    config.update({"torch_version": torch.__version__, "python_version": platform.python_version(),
                   "device": str(device), "task": "sequential_contextual_recall",
                   "chance_query_accuracy": 0.25,
                   "memory_schedule": "read every event; write only the first four cue events",
                   "gradient_window": "entire episode; state resets between episodes",
                   "eval_random_stream": "fixed and disjoint from all training seeds"})
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = []
    for seed in args.seeds:
        for method in args.methods:
            results.append(train_one(seed, method, args, device, output))
            (output / "results.json").write_text(json.dumps({"runs": results,
                "summary": summarize(results)}, indent=2) + "\n")
    write_report(results, args, output)
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
