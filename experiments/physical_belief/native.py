"""Public-evidence bridge between physical learning and native experimental control."""
from __future__ import annotations

import torch


class PhysicalBelief:
    """One learned physical belief per ordered stream, without task-family labels.

    Current coefficients are recomputed from the current public ledger. A task
    contributes to cross-task memory exactly once, after its last observation.
    ``discard`` computes that same memory update and then discards it; ``reset``
    clears memory at task boundaries. Neither changes the shared model weights.
    """

    def __init__(self, checkpoint, device, *, mode="carry"):
        from cdm.experimental_belief import load_experimental_belief_checkpoint
        if mode not in ("carry", "reset", "discard"):
            raise ValueError("Physical belief mode must be carry, reset or discard")
        self.model, _ = load_experimental_belief_checkpoint(checkpoint, device)
        self.model.eval()
        self.device, self.mode = torch.device(device), mode
        self.memory = self.model.initial(1, device=self.device)
        self.episodes = 0
        self.last_update_norm = 0.

    def begin_episode(self):
        if self.mode == "reset":
            self.memory = self.model.initial(1, device=self.device)
        self.last_update_norm = 0.

    def __call__(self, tensors):
        predictions, _ = self.model.observe_predict(
            tensors["values"], tensors["seen"], tensors["known_hz"], tensors["known"],
            tensors["query"], self.memory)
        return predictions

    def end_episode(self, tensors):
        _, posterior = self.model.observe_predict(
            tensors["values"], tensors["seen"], tensors["known_hz"], tensors["known"],
            tensors["query"], self.memory)
        updated = self.model.remember(self.memory, posterior)
        self.last_update_norm = sum(float((updated[key].float() - value.float()).norm())
                                    for key, value in self.memory.items()
                                    if torch.is_tensor(value) and value.is_floating_point())
        if self.mode != "discard":
            self.memory = updated
        self.episodes += 1

    def summary(self):
        return {"episodes_received": self.episodes, "computed_update_norm": self.last_update_norm,
                "state_norms": {key: float(value.float().norm())
                                for key, value in self.memory.items() if torch.is_tensor(value)},
                "retained": self.mode != "discard"}
