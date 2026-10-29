"""Train physical predictive beliefs through changing and returning experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from cdm.experimental_belief import ExperimentalBelief, save_experimental_belief_checkpoint
from experiments.discoveryworld.calibration_mixture import load_mixture_checkpoint
from .data import RecordedWorlds, episode, target_order


def stream_loss(model, stream, *, mode="retain", return_rows=False):
    """Score before revealing each working setting, then preserve its evidence.

    Reset, discarded updates and last-context comparisons still compute the
    same public posterior. No experimental identity is supplied to the model.
    """
    batch, length = stream["values"].shape[:2]
    memory = model.initial(batch)
    numerator, denominator = stream["values"].new_zeros(()), 0
    rows = []
    for index in range(length):
        if mode == "reset": memory = model.initial(batch)
        current = episode(stream, index)
        values, seen, settings = (current[k] for k in ("values", "seen", "settings"))
        known = current["support"].clone()
        order = target_order(current["targets"])
        row_indices = torch.arange(batch, device=values.device)
        for target_number in range(order.shape[1]):
            query = order[:, target_number]
            valid = current["targets"][row_indices, query]
            # Unreceived labels are zeroed at the interface as well as masked.
            posterior = model.posterior(values, seen, torch.where(known, settings, 0.), known, memory,
                                        retrieve=mode != "no_retrieval")
            prediction = model.predict(values[row_indices, query], seen[row_indices, query], posterior)
            target = settings[row_indices, query]
            nll = model.nll(prediction, target)
            numerator = numerator + torch.where(valid, nll, 0.).sum()
            denominator += int(valid.sum())
            if return_rows:
                selected = prediction["weights"].argmax(-1)
                map_mean = prediction["means"].gather(1, selected[:, None])[:, 0]
                rows.append({"episode": index, "target_number": target_number,
                             "valid": valid, "nll": nll, "target_hz": target,
                             "mixture_mean_hz": prediction["mixture_mean"],
                             "map_mean_hz": map_mean, "map_quantity": selected,
                             "weights": prediction["weights"],
                             "means_hz": prediction["means"], "std_hz": prediction["std"]})
            known = known.scatter(1, query[:, None], valid[:, None] | known.gather(1, query[:, None]))
        final = model.posterior(values, seen, torch.where(known, settings, 0.), known, memory,
                                retrieve=mode != "no_retrieval")
        computed = model.remember(memory, final)
        if mode == "last_context":
            # The implementation's memory is chronological. Clearing before
            # writing keeps the last completed posterior with the same API.
            memory = model.remember(model.initial(batch), final)
        elif mode != "discard":
            memory = computed
    return numerator / max(denominator, 1), rows


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    mixture, _ = load_mixture_checkpoint(args.mixture_checkpoint, device)
    model = ExperimentalBelief(mixture.normalization, memory_slots=args.memory_slots,
                              noise_floor_hz=mixture.noise_floor_hz).to(device)
    model.initialize_from_mixture(mixture)
    # Independent starts affect learnable retrieval/retention parameters as
    # well as data order, without overwriting the inherited numerical prior.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in ("prior_mean", "log_coefficient_scale", "log_noise_extra_hz", "mixture_logits"):
                parameter.add_(.01 * torch.randn_like(parameter))
    worlds = RecordedWorlds(args.data / "train.npz", device, args.radiation_readings)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "training.jsonl").exists():
        raise ValueError("Use a fresh output directory")
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    configuration.update(parameters=sum(p.numel() for p in model.parameters()),
                         training_worlds=worlds.size,
                         native_training_quantities=sorted(worlds.native_pools),
                         trained_core="physical predictive belief; inherited action core not updated",
                         checkpoint_selection="last fixed-budget update",
                         observation="recorded readings; public settings revealed after prediction",
                         generated_tasks="affine alternative through original two support settings",
                         memory_gradient="through all episodes in each sampled task stream")
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    save_experimental_belief_checkpoint(model, args.output / "initial.pt", training=configuration)
    start = perf_counter()
    with (args.output / "training.jsonl").open("w", buffering=1) as log:
        for update in range(1, args.updates + 1):
            stream = worlds.sample_stream(args.batch, generator=generator,
                                          segment_length=args.segment_length,
                                          generated_fraction=args.generated_fraction)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = stream_loss(model, stream, mode=args.mode)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite predictive loss at update {update}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"Nonfinite gradient at update {update}")
            optimizer.step()
            row = {"update": update, "nll_hz": float(loss.detach()), "gradient_norm": float(grad_norm),
                   "elapsed_seconds": perf_counter() - start,
                   "generated_streams": int(stream["generated"][:, 0].sum()),
                   "episodes": args.batch * 3 * args.segment_length}
            log.write(json.dumps(row) + "\n")
            if update == 1 or update % args.log_every == 0:
                print(json.dumps(row), flush=True)
            if update % args.save_every == 0 or update == args.updates:
                save_experimental_belief_checkpoint(model, args.output / "last.pt",
                                                    training={**configuration, **row})
    (args.output / "completed.json").write_text(json.dumps(row, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mixture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--segment-length", type=int, default=2)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--radiation-readings", type=int, default=32)
    parser.add_argument("--generated-fraction", type=float, default=.5)
    parser.add_argument("--mode", choices=("retain", "reset", "discard", "last_context"), default="retain")
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--grad-clip", type=float, default=10.)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
