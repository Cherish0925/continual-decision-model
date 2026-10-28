"""Train experimental decisions across unannounced changes and returning relations.

Each task uses recorded instruments and a generated affine calibration through
two original public support settings. The causal quantity persists within a
block; its coefficients are recalibrated for each new set of objects. Only the
environment knows the relation, block, or task index. Task ledgers reset while
the continuing model may retain state. This is state learning, not deployment
time parameter updating.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from cdm.active_experiment import load_active_checkpoint
from .calibration_mixture import load_mixture_checkpoint
from .train_active_core import ActiveTableEnvironment, counterfactual_calibration


class CalibrationTaskPool:
    """Actual recorded readings, with all admissible two-support task variants."""

    def __init__(self, path, mixture, device):
        size = len(np.load(path)["x"])
        reference = ActiveTableEnvironment(path, mixture, batch=size, device=device, permute=False)
        reference.reset(torch.arange(size, device=device))
        self.data, self.size, self.device = reference.data, size, torch.device(device)
        self.labels = torch.empty(size, 9, 4, device=device)
        valid = torch.zeros(size, 9, dtype=torch.bool, device=device)
        for quantity in range(9):
            generated, accepted, _ = counterfactual_calibration(
                reference.full_values, reference.full_present, reference.targets,
                reference.initial_support, reference.target_mask,
                torch.full((size,), quantity, device=device, dtype=torch.long))
            self.labels[:, quantity] = generated
            valid[:, quantity] = accepted
        self.eligible = [torch.where(valid[:, quantity])[0].cpu().numpy() for quantity in range(9)]
        if any(len(indices) < 11 for indices in self.eligible):
            raise ValueError("Each relation needs at least eleven distinct admissible worlds")

    def description(self):
        return {"worlds": self.size, "eligible_worlds_by_quantity": [len(x) for x in self.eligible],
                "labels": "generated queries; original public support settings unchanged",
                "measurements": "unchanged recorded instrument returns"}


class TaskStreamPlan:
    """Policy-independent A->B->A streams, with randomized block durations."""

    def __init__(self, pool, *, streams, lifetimes, seed, device):
        rng = np.random.default_rng(seed)
        shape = (streams, lifetimes)
        first = rng.integers(9, size=shape)
        middle = (first + rng.integers(1, 9, size=shape)) % 9
        durations = np.stack((rng.integers(2, 5, size=shape), rng.integers(1, 4, size=shape),
                              rng.integers(2, 5, size=shape)), -1)
        position = np.arange(11)[None, None]
        phase = (position >= durations[..., 0, None]).astype(np.uint8)
        phase += (position >= durations[..., :2].sum(-1)[..., None]).astype(np.uint8)
        quantities = np.where(phase == 1, middle[..., None], first[..., None]).astype(np.uint8)
        worlds = np.empty((*shape, 11), dtype=np.int32)
        for index in range(11):
            q = quantities[..., index]
            for quantity, eligible in enumerate(pool.eligible):
                selected = q == quantity
                proposed = rng.choice(eligible, size=int(selected.sum()))
                if index:
                    previous = worlds[..., :index][selected]
                    duplicate = (proposed[:, None] == previous).any(-1)
                    while duplicate.any():
                        proposed[duplicate] = rng.choice(eligible, size=int(duplicate.sum()))
                        duplicate = (proposed[:, None] == previous).any(-1)
                worlds[..., index][selected] = proposed
        permutation = rng.random((*shape, 11, 4)).argsort(-1).astype(np.uint8)
        self.worlds = torch.as_tensor(worlds, device=device)
        self.quantities = torch.as_tensor(quantities, device=device)
        self.phase = torch.as_tensor(phase, device=device)
        self.permutation = torch.as_tensor(permutation, device=device)
        self.lengths = torch.as_tensor(durations.sum(-1), device=device)
        self.streams, self.lifetimes, self.seed = streams, lifetimes, seed

    def description(self):
        return {"seed": self.seed, "streams": self.streams, "reserved_lifetimes": self.lifetimes,
                "pattern": "A->B->A", "block_task_ranges": [[2, 4], [1, 3], [2, 4]],
                "relation_sampling": "uniform among nine quantities; B differs from A",
                "objects": "distinct source worlds within each lifetime; independently permuted slots",
                "actor_metadata": "none", "world_order": "precomputed independently of actions and policy RNG"}


_TASK_FIELDS = ("indices", "permutation", "samples", "sample_present", "sample_count", "targets",
                "initial_support", "target_mask", "full_values", "full_present", "values", "counts",
                "group_counts", "known", "known_hz", "query", "failed", "event", "steps", "alive",
                "successes", "acquisitions", "tests", "best_quality", "best_error")


class ActiveLifetimeEnvironment:
    def __init__(self, pool, plan, mixture, *, max_steps=160, tolerance_hz=8.):
        self.pool, self.plan, self.mixture = pool, plan, mixture
        self.batch, self.device = plan.streams, pool.device
        self.lifetime = torch.zeros(self.batch, dtype=torch.long, device=self.device)
        self.task = torch.zeros_like(self.lifetime)
        self.completed = torch.zeros_like(self.lifetime)
        self.task_rewards = torch.zeros(self.batch, device=self.device)
        self.task_rows = []
        self.table = ActiveTableEnvironment.__new__(ActiveTableEnvironment)
        self.table.__dict__.update(data=pool.data, size=pool.size, batch=self.batch, device=self.device,
                                   mixture=mixture, max_steps=max_steps, tolerance=tolerance_hz,
                                   measurement_cost=.005, test_cost=.025, test_retry_rule="changed_evidence")
        self._begin_tasks(torch.arange(self.batch, device=self.device), initial=True)

    @torch.no_grad()
    def _begin_tasks(self, rows, initial=False):
        n = len(rows)
        lifetime, task = self.lifetime[rows], self.task[rows]
        indices = self.plan.worlds[rows, lifetime, task].long()
        quantities = self.plan.quantities[rows, lifetime, task].long()
        permutation = self.plan.permutation[rows, lifetime, task].long()
        fresh = ActiveTableEnvironment.__new__(ActiveTableEnvironment)
        fresh.__dict__ = {**self.table.__dict__, "batch": n, "indices": indices, "permutation": permutation}
        fresh._load_data()
        fresh.targets = self.pool.labels[indices, quantities].gather(1, permutation)
        fresh.values = torch.zeros(n, 4, 9, device=self.device)
        fresh.counts = torch.zeros(n, 4, 9, dtype=torch.long, device=self.device)
        fresh.group_counts = torch.zeros(n, 4, 5, dtype=torch.long, device=self.device)
        fresh.known = fresh.initial_support.clone()
        fresh.known_hz = torch.where(fresh.known, fresh.targets, 0.)
        fresh.query = fresh.target_mask.float().argmax(-1)
        fresh.failed = torch.zeros(n, 9, dtype=torch.bool, device=self.device)
        fresh.event = torch.zeros(n, 36, device=self.device)
        fresh.event[torch.arange(n, device=self.device), 32 + fresh.query] = 1
        fresh.steps = torch.zeros(n, dtype=torch.long, device=self.device)
        fresh.alive = torch.ones(n, dtype=torch.bool, device=self.device)
        fresh.successes = torch.zeros(n, device=self.device)
        fresh.acquisitions = torch.zeros(n, device=self.device)
        fresh.tests = torch.zeros(n, device=self.device)
        fresh.best_quality = torch.zeros(n, 4, device=self.device)
        fresh.best_error = torch.full((n, 4), torch.inf, device=self.device)
        for name in _TASK_FIELDS:
            if initial:
                setattr(self.table, name, getattr(fresh, name))
            else:
                getattr(self.table, name)[rows] = getattr(fresh, name)
        self.task_rewards[rows] = 0

    def observation(self):
        return self.table.observation()

    def oracle_values(self):
        return self.table.oracle_values()

    def step(self, actions):
        rewards, task_done = self.table.step(actions)
        self.task_rewards += rewards
        lengths = self.plan.lengths[torch.arange(self.batch, device=self.device), self.lifetime]
        lifetime_done = task_done & (self.task + 1 == lengths)
        return rewards, task_done, lifetime_done

    @torch.no_grad()
    def next_tasks(self, task_done, *, restart_lifetimes=True):
        rows = torch.where(task_done)[0]
        if not len(rows):
            return
        lifetime, task = self.lifetime[rows], self.task[rows]
        data = torch.stack((rows, lifetime, task,
                            self.plan.phase[rows, lifetime, task],
                            self.plan.quantities[rows, lifetime, task],
                            self.plan.worlds[rows, lifetime, task],
                            self.table.successes[rows], self.table.target_mask[rows].sum(-1),
                            self.table.acquisitions[rows], self.table.tests[rows],
                            self.task_rewards[rows]), -1).cpu().tolist()
        keys = ("stream", "lifetime", "task", "phase", "quantity", "source_world", "successes", "queries",
                "measurements", "tests", "reward")
        self.task_rows.extend({key: float(value) if key == "reward" else int(value)
                               for key, value in zip(keys, row)} for row in data)
        end = self.task[rows] + 1 == self.plan.lengths[rows, lifetime]
        self.task[rows] += 1
        self.completed[rows[end]] += 1
        self.lifetime[rows[end]] += 1
        self.task[rows[end]] = 0
        if not restart_lifetimes:
            rows = rows[~end]
        if len(rows):
            self._begin_tasks(rows)

    def summary(self):
        phases = {}
        for phase, label in enumerate(("initial_A", "interference_B", "return_A")):
            selected = [row for row in self.task_rows if row["phase"] == phase]
            if selected:
                phases[label] = {"tasks": len(selected),
                    "complete_tasks": sum(row["successes"] == row["queries"] for row in selected),
                    "successful_queries": sum(row["successes"] for row in selected),
                    "queries": sum(row["queries"] for row in selected),
                    "mean_measurements": float(np.mean([row["measurements"] for row in selected])),
                    "mean_tests": float(np.mean([row["tests"] for row in selected])),
                    "mean_reward": float(np.mean([row["reward"] for row in selected]))}
        return {"completed_tasks": len(self.task_rows), "completed_lifetimes": int(self.completed.sum()),
                "phases": phases}


def flush_terminal_state(model, state, observation, done, refinements=1):
    """Consume the final received outcome once, with no action or world step."""
    rows = torch.where(done)[0]
    if not len(rows):
        return state
    terminal = model.advance(observation["event"][rows], observation["evidence"][rows],
                             observation["evidence_mask"][rows], {key: value[rows] for key, value in state.items()},
                             refinements)
    return {key: value.index_copy(0, rows, terminal[key]) for key, value in state.items()}


@torch.no_grad()
def evaluate(model, pool, mixture, *, state_mode, streams, seed, max_steps, refinements):
    plan = TaskStreamPlan(pool, streams=streams, lifetimes=2, seed=seed, device=pool.device)
    env = ActiveLifetimeEnvironment(pool, plan, mixture, max_steps=max_steps)
    state = model.initial(streams, pool.device)
    finished = torch.zeros(streams, dtype=torch.bool, device=pool.device)
    model.eval()
    for _ in range(11 * max_steps):
        out = model(**env.observation(), state=state, refinements=refinements)
        _, task_done, lifetime_done = env.step(out["logits"].argmax(-1))
        task_done &= ~finished
        lifetime_done &= ~finished
        state = flush_terminal_state(model, out["state"], env.observation(), task_done, refinements)
        state = model.reset_rows(state, task_done if state_mode == "task_reset" else lifetime_done)
        finished |= lifetime_done
        env.next_tasks(task_done, restart_lifetimes=False)
        if finished.all():
            break
    return {"summary": env.summary(), "instances": env.task_rows, "plan": plan.description()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixture-checkpoint", type=Path,
                        default=Path("runs/discoveryworld/calibration/mixture/shared_train_only_seed31/last.pt"))
    parser.add_argument("--state-mode", choices=("carry", "task_reset"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=191)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--unroll", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--validation-streams", type=int, default=64)
    parser.add_argument("--refinements", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    mixture, _ = load_mixture_checkpoint(args.mixture_checkpoint, args.device)
    mixture.requires_grad_(False)
    pool = CalibrationTaskPool(args.data / "train.npz", mixture, args.device)
    validation_pool = CalibrationTaskPool(args.data / "validation.npz", mixture, args.device)
    # Every task needs at least three measurements and one test. A lifetime has
    # at least five tasks, so this reserves more than the maximum needed plans.
    lifetimes = (args.updates * args.unroll + 19) // 20 + 2
    plan = TaskStreamPlan(pool, streams=args.batch, lifetimes=lifetimes, seed=args.seed, device=args.device)
    env = ActiveLifetimeEnvironment(pool, plan, mixture, max_steps=args.max_steps)
    model, parent = load_active_checkpoint(args.checkpoint, args.device)
    if model.config.mode != "persistent":
        raise ValueError("Both arms must start from the same persistent policy")
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in parameters if name.startswith("core.")], "lr": args.lr},
        {"params": [p for name, p in parameters if not name.startswith("core.")], "lr": args.head_lr}], weight_decay=.01)
    optimizer.load_state_dict(parent["optimizer"])
    optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"] = args.lr, args.head_lr
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(training_pool=pool.description(), validation_pool=validation_pool.description(),
                    stream_plan=plan.description(), scope="cross-task recurrent state learning; parameters frozen in evaluation",
                    task_boundary="reset ledger and numerical fit; no phase/relation/index actor inputs",
                    terminal_feedback="one state update without an action before each task boundary",
                    return_boundary="bootstrap across tasks; discount stops at lifetime end only",
                    checkpoint_selection="fixed final update", bptt="connected across task boundaries within unroll; detached only between updates")
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    state = model.initial(args.batch, args.device)
    history, started, boundary_windows = [], time.monotonic(), 0
    terminal_updates = 0
    task_log = gzip.open(args.output / "training_tasks.jsonl.gz", "wt")
    logged_tasks = 0
    for update in range(args.updates):
        model.train()
        state = {key: value.detach() for key, value in state.items()}
        rewards, ends, values, log_probs, entropies, auxiliaries, imitation = [], [], [], [], [], [], []
        task_crossings = 0
        for _ in range(args.unroll):
            obs = env.observation()
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.bf16):
                out = model(**obs, state=state, refinements=args.refinements)
            oracle = env.oracle_values()
            distribution = torch.distributions.Categorical(logits=out["logits"].float())
            actions = distribution.sample()
            reward, task_done, lifetime_done = env.step(actions)
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.bf16):
                state = flush_terminal_state(model, out["state"], env.observation(), task_done, args.refinements)
            state = model.reset_rows(state, task_done if args.state_mode == "task_reset" else lifetime_done)
            env.next_tasks(task_done)
            crossings = int(task_done.sum())
            task_crossings += crossings
            terminal_updates += crossings
            rewards.append(reward); ends.append(lifetime_done); values.append(out["value"].float())
            log_probs.append(distribution.log_prob(actions)); entropies.append(distribution.entropy())
            auxiliaries.append(F.smooth_l1_loss(out["outcomes"][..., 0].float()[obs["action_mask"]],
                                                oracle[obs["action_mask"]]))
            teacher = (oracle / .1).masked_fill(~obs["action_mask"], -torch.inf).softmax(-1)
            logp = out["logits"].float().log_softmax(-1).masked_fill(~obs["action_mask"], 0.)
            imitation.append(-(teacher * logp).sum(-1).mean())
        boundary_windows += int(task_crossings > 0)
        with torch.no_grad(), torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.bf16):
            target = model(**env.observation(), state=state, refinements=args.refinements)["value"].float()
        returns = []
        for reward, lifetime_done in zip(reversed(rewards), reversed(ends)):
            target = reward + .99 * target * (~lifetime_done)
            returns.append(target)
        returns = torch.stack(list(reversed(returns)))
        predicted = torch.stack(values)
        advantage = returns - predicted
        value_loss = F.smooth_l1_loss(predicted, returns)
        policy_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        auxiliary = torch.stack(auxiliaries).mean()
        imitation_loss = torch.stack(imitation).mean()
        entropy = torch.stack(entropies).mean()
        loss = policy_loss + .5 * value_loss + .1 * auxiliary + .02 * imitation_loss - .01 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.)
        optimizer.step()
        for completed_task in env.task_rows[logged_tasks:]:
            task_log.write(json.dumps(completed_task) + "\n")
        logged_tasks = len(env.task_rows)
        if update == 0 or (update + 1) % 10 == 0:
            row = {"update": update + 1, "loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                   "value_loss": float(value_loss.detach()), "outcome_loss": float(auxiliary.detach()),
                   "entropy": float(entropy.detach()), "mean_step_reward": float(torch.stack(rewards).mean()),
                   "gradient_norm": float(norm), "task_boundaries_this_update": task_crossings,
                   "windows_with_task_boundaries": boundary_windows, "terminal_feedback_updates": terminal_updates,
                   "elapsed_seconds": time.monotonic() - started}
            print(json.dumps(row), flush=True); history.append(row)
        if (update + 1) % args.eval_every == 0 or update + 1 == args.updates:
            result = evaluate(model, validation_pool, mixture, state_mode=args.state_mode,
                              streams=args.validation_streams, seed=args.seed + 100000,
                              max_steps=args.max_steps, refinements=args.refinements)
            row = {"update": update + 1, "validation": result["summary"], "elapsed_seconds": time.monotonic() - started}
            print(json.dumps(row), flush=True); history.append(row)
            saved = {**model.configuration(), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "training": settings, "update": update + 1, "validation": result["summary"]}
            torch.save(saved, args.output / "last.pt")
            if update + 1 == args.updates:
                torch.save(saved, args.output / f"update{update + 1}.pt")
            (args.output / f"validation{update + 1}.json").write_text(json.dumps(result, indent=2) + "\n")
            (args.output / "training.json").write_text(json.dumps({"config": settings, "history": history,
                                                                  "training_tasks": env.summary()}, indent=2) + "\n")
            task_log.flush()
    task_log.close()
    print(json.dumps({"finished": True, "updates": args.updates, "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
