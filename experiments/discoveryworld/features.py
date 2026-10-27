"""Frozen public-observation features shared by trajectory training and acting.

SigLIP sees every text chunk, including late instrument results. Frozen tower
outputs are cached independently of the trainable policy projections. The
lexical mode is an explicit image-masked bag-of-words ablation, not a substitute
pretrained encoder and not random synthetic input.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import re

import torch
import torch.nn.functional as F

from cdm.digital import observation_numeric, observation_text
from cdm.multimodal import ENCODER_ID, ENCODER_REVISION

FEATURE_SCHEMA = 2
MEASUREMENT_RELATION_COLUMNS = (
    "arg1_is_instrument", "arg1_is_target", "arg1_is_contained",
    "arg2_is_instrument", "arg2_is_target", "arg2_is_contained",
    "dialog_owner_is_target", "dialog_currently_contains_record_target",
)


@dataclass(frozen=True)
class FeatureConfig:
    schema: int = FEATURE_SCHEMA
    time_basis: str = "world_steps"
    mode: str = "siglip"
    encoder_id: str = ENCODER_ID
    encoder_revision: str = ENCODER_REVISION
    lexical_dim: int = 256
    image_grid: int = 4
    measurement_memory: bool = False
    object_relations: bool = False
    mask_object_relations: bool = False

    def __post_init__(self):
        if self.object_relations and not self.measurement_memory:
            raise ValueError("object_relations requires measurement_memory")
        if self.mask_object_relations and not self.object_relations:
            raise ValueError("mask_object_relations requires object_relations")


def canonical_action(action: dict) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def lexical_features(text: str, width: int = 256) -> torch.Tensor:
    """Stable signed feature hashing of received words and adjacent word pairs."""
    words = re.findall(r"[\w.-]+|[^\w\s]", text.lower())
    terms = words + [a + " " + b for a, b in zip(words, words[1:])]
    result = torch.zeros(width)
    for term in terms:
        digest = sha256(term.encode()).digest()
        result[int.from_bytes(digest[:4], "little") % width] += 1 if digest[4] & 1 else -1
    return F.normalize(result, dim=0)


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _measurement_relations(observation: dict, candidates: list, rows: list[dict]) -> torch.Tensor:
    """Equality links only; handles never become numerical model features.

    The last column uses contents explicitly observed in the current UI. Past
    object locations or earlier measurements cannot establish current contents.
    """
    owner, ui = observation.get("measurement_dialog_owner"), observation["ui"]
    objects = {obj["uuid"]: obj for group in ui.get("nearbyObjects", {}).get("objects", {}).values()
               for obj in group}
    objects.update((obj["uuid"], obj) for obj in ui.get("accessibleEnvironmentObjects", []) + ui.get("inventoryObjects", []))
    contents = set()
    if owner is not None:
        for handle, obj in objects.items():
            parent = re.search(r"\(in [^\n]*?\[uuid:\s*(\d+)\]", obj.get("description", ""))
            if parent is not None and parent.group(1) == str(owner):
                contents.add(handle)

    def equal(first, second):
        return first is not None and second is not None and first == second

    values = []
    for candidate in candidates:
        relations = []
        for row in rows:
            entry = []
            for handle in candidate.object_arguments:
                entry.extend((equal(handle, row["instrument_handle"]), equal(handle, row["target_handle"]),
                              handle is not None and handle in row.get("contained_objects", ())))
            entry.extend((equal(owner, row["target_handle"]), row["target_handle"] in contents))
            relations.append(entry)
        values.append(relations)
    return torch.tensor(values, dtype=torch.float32).reshape(len(candidates), len(rows), 8)


def _image_bytes(observation: dict, root: Path | None) -> bytes | None:
    vision = observation.get("vision", {})
    image = vision.get("base64_no_grid") or vision.get("no_grid")
    if image is None and vision:
        image = next(iter(vision.values()))
    if image is None:
        return None
    if isinstance(image, str):
        header, encoded = image.split(",", 1)
        if header != "data:image/png;base64":
            raise ValueError("Expected a PNG data URL")
        return base64.b64decode(encoded, validate=True)
    if root is None:
        raise ValueError("A trajectory root is required for an image path")
    return (root / image["path"]).read_bytes()


def _save(value, path: str | Path):
    """Concurrent collectors/readers see a complete cache entry."""
    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, prefix="." + path.name, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=128)
def load_features(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


@lru_cache(maxsize=4096)
def _load_text(path: str) -> torch.Tensor:
    # Action descriptions recur across successive observations. Content-addressed
    # text tensors are immutable; reuse them instead of reopening hundreds of
    # identical files at every decision. Event caches have a separate budget.
    return torch.load(path, map_location="cpu", weights_only=True)


class FeatureCache:
    def __init__(self, directory: str | Path, config: FeatureConfig,
                 device: str = "cpu", encoder_path: str | Path | None = None,
                 encode_batch_size: int = 32):
        if config.mode not in ("siglip", "lexical"):
            raise ValueError("Feature mode must be siglip or lexical")
        if config.time_basis not in ("world_steps", "wall_clock"):
            raise ValueError("time_basis must be world_steps or wall_clock")
        self.config, self.device = config, torch.device(device)
        self.encoder_path = str(encoder_path) if encoder_path else None
        self.encode_batch_size = encode_batch_size
        identity = {"schema": FEATURE_SCHEMA, **asdict(config)}
        # Preserve the existing tower/event cache when the new input is off.
        if not config.measurement_memory:
            identity.pop("measurement_memory")
        if not config.object_relations:
            identity.pop("object_relations")
        if not config.mask_object_relations:
            identity.pop("mask_object_relations")
        tower_identity = {key: value for key, value in identity.items()
                          if key not in ("measurement_memory", "object_relations", "mask_object_relations")}
        self.tower_directory = Path(directory) / _digest(tower_identity)[:20]
        self.tower_directory.mkdir(parents=True, exist_ok=True)
        self.directory = Path(directory) / _digest(identity)[:20]
        self.directory.mkdir(parents=True, exist_ok=True)
        metadata = asdict(config)
        if not config.mask_object_relations:
            metadata.pop("mask_object_relations")
        if not config.object_relations:
            metadata.pop("object_relations")
        else:
            metadata["measurement_relation_columns"] = MEASUREMENT_RELATION_COLUMNS
        (self.directory / "metadata.json").write_text(json.dumps(
            {**metadata, "text_truncation": False,
             "timing_clock": config.time_basis}, indent=2) + "\n")
        self.model = self.processor = None
        self.text_dim = self.image_dim = 768 if config.mode == "siglip" else config.lexical_dim

    def _load_encoder(self):
        if self.model is not None:
            return
        from transformers import AutoModel, AutoProcessor
        source = self.encoder_path or self.config.encoder_id
        options = {"local_files_only": True} if self.encoder_path else {"revision": self.config.encoder_revision}
        self.processor = AutoProcessor.from_pretrained(source, **options)
        self.model = AutoModel.from_pretrained(source, **options).to(self.device).requires_grad_(False).eval()
        self.text_dim = self.model.config.text_config.hidden_size
        self.image_dim = self.model.config.vision_config.hidden_size
        if (self.text_dim, self.image_dim) != (768, 768):
            raise ValueError("This cache schema expects the pinned SigLIP2 base dimensions")

    def _path(self, kind: str, key: str) -> Path:
        folder = (self.tower_directory if kind in ("text", "image") else self.directory) / kind
        folder.mkdir(exist_ok=True)
        return folder / (key + ".pt")

    @torch.no_grad()
    def texts(self, texts: list[str]) -> list[torch.Tensor]:
        paths = [self._path("text", _digest(text)) for text in texts]
        missing = {str(path): text for path, text in zip(paths, texts) if not path.exists()}
        if missing and self.config.mode == "lexical":
            for name, text in missing.items():
                # One token per public line retains separately visible objects/results.
                lines = [line for line in text.splitlines() if line.strip()] or [""]
                _save(torch.stack([lexical_features(line, self.text_dim) for line in lines]), name)
        elif missing:
            self._load_encoder()
            tokenizer = self.processor.tokenizer
            context = int(self.model.config.text_config.max_position_embeddings)
            capacity = context - tokenizer.num_special_tokens_to_add(pair=False)
            if capacity <= 0:
                raise ValueError("Tokenizer special tokens exceed text context")
            chunks, owners = [], []
            backend = tokenizer.backend_tokenizer
            backend.no_truncation()
            backend.no_padding()
            for name, text in missing.items():
                # Transformers 5 uses TokenizersBackend, which intentionally
                # has no prepare_for_model. Chunk complete native encodings,
                # then apply the checkpoint's own special-token postprocessor.
                encoded_text = backend.encode(text, add_special_tokens=False)
                encoded_text.truncate(capacity)
                for chunk in [encoded_text, *encoded_text.overflowing]:
                    chunk = backend.post_process(chunk, add_special_tokens=True)
                    chunk.pad(context, direction=tokenizer.padding_side,
                              pad_id=tokenizer.pad_token_id, pad_token=tokenizer.pad_token)
                    chunks.append({"input_ids": chunk.ids, "attention_mask": chunk.attention_mask})
                    owners.append(name)
            encoded = {name: [] for name in missing}
            for start in range(0, len(chunks), self.encode_batch_size):
                batch = chunks[start:start + self.encode_batch_size]
                ids = torch.tensor([row["input_ids"] for row in batch], device=self.device)
                mask = torch.tensor([row["attention_mask"] for row in batch], device=self.device)
                result = self.model.text_model(input_ids=ids, attention_mask=mask,
                                               return_dict=True).last_hidden_state
                for offset, features in enumerate(result):
                    encoded[owners[start + offset]].append(features[mask[offset].bool()].cpu().half())
            for name, tensors in encoded.items():
                _save(torch.cat(tensors), name)
        return [_load_text(str(path)) for path in paths]

    @torch.no_grad()
    def image(self, payload: bytes | None) -> torch.Tensor | None:
        if self.config.mode == "lexical" or payload is None:
            return None
        path = self._path("image", sha256(payload).hexdigest())
        if not path.exists():
            self._load_encoder()
            from PIL import Image
            with Image.open(BytesIO(payload)) as source:
                inputs = self.processor(images=source.convert("RGB"), return_tensors="pt")
            features = self.model.vision_model(
                pixel_values=inputs["pixel_values"].to(self.device), return_dict=True).last_hidden_state
            side = int(features.shape[1] ** .5)
            if side * side != features.shape[1]:
                raise ValueError("Expected the pinned SigLIP square patch grid")
            features = features.transpose(1, 2).reshape(1, -1, side, side)
            features = F.adaptive_avg_pool2d(features, self.config.image_grid).flatten(2).transpose(1, 2)[0]
            _save(features.cpu().half(), path)
        return torch.load(path, map_location="cpu", weights_only=True)

    def encode(self, observation: dict, candidates: list, root: Path | None = None) -> Path:
        text = observation_text(observation)
        numeric = observation_numeric(observation)
        payload = _image_bytes(observation, root) if self.config.mode == "siglip" else None
        capture = max(float(observation.get("capture_seconds", 0)), 0)
        arrival = max(float(observation.get("arrival_seconds", capture)), capture)
        # The official simulator advances one native tick per action; renderer
        # and teacher-inference wall time do not advance its scientific state.
        timing = [float(observation["ui"].get("world_steps", 0)), 0.0, 0.0] if self.config.time_basis == "world_steps" else [capture, arrival - capture, 0.0]
        measurements, rows = [], []
        if self.config.measurement_memory:
            rows = [asdict(row) if is_dataclass(row) else row
                    for row in observation.get("measurement_records", [])]
            rows = sorted(rows, key=lambda row: row["last_sequence"])
            last_sequence = max((row["last_sequence"] for row in rows), default=-1)
            for row in rows:
                location = row["conditions"].get("agent_location", {})
                inventory = sorted(obj.get("name", "object") for obj in row["conditions"].get("inventory", []))
                label = (f"{row['instrument_name']} measuring {row['target_name']}; "
                         f"quantity {row['quantity']}; unit {row['unit'] or 'unspecified'}; "
                         f"location {location.get('x', '?')}, {location.get('y', '?')}; "
                         f"inventory {', '.join(inventory)}")
                values = [row['count'], row['minimum'], row['maximum'], row['mean'], row['latest'],
                          row['maximum'] - row['minimum'], row['last_change'],
                          row['last_sequence'] - row['first_sequence']]
                measurements.append((label, values,
                                     (last_sequence - row['last_sequence']) / max(last_sequence + 1, 1)))
        identity = {"text": text, "numeric": numeric, "timing": timing,
                    "image": sha256(payload).hexdigest() if payload else None,
                    "candidates": [(c.text, c.numeric, canonical_action(c.action)) for c in candidates]}
        if self.config.measurement_memory:
            identity["measurements"] = measurements
        relations = None
        if self.config.object_relations:
            relations = _measurement_relations(observation, candidates, rows)
            if self.config.mask_object_relations:
                relations.zero_()
            identity["measurement_relations"] = relations.tolist()
        path = self._path("event", _digest(identity))
        if not path.exists():
            encoded_text = self.texts([text] + [candidate.text for candidate in candidates])
            observation_features = {"numeric": torch.tensor(numeric, dtype=torch.float32),
                                    "timing": torch.tensor(timing, dtype=torch.float32),
                                    "text": encoded_text[0]}
            image = self.image(payload)
            if image is not None:
                observation_features["image"] = image
            if measurements:
                labels = self.texts([row[0] for row in measurements])
                observation_features["measurement_text"] = torch.stack([part.float().mean(0) for part in labels])
                values = torch.tensor([row[1] for row in measurements], dtype=torch.float32)
                observation_features["measurement_numeric"] = values.sign() * values.abs().log1p() / 10
                observation_features["measurement_age"] = torch.tensor([row[2] for row in measurements], dtype=torch.float32)
            candidate_features = {"text": torch.stack([part.float().mean(0) for part in encoded_text[1:]]),
                                  "numeric": torch.tensor([c.numeric for c in candidates], dtype=torch.float32)}
            if relations is not None:
                candidate_features["measurement_relations"] = relations
            _save({"observation": observation_features, "candidates": candidate_features}, path)
        return path

    def close(self):
        self.model = self.processor = None
        _load_text.cache_clear()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()


def batch_features(features: list[dict], device: torch.device | str) -> tuple[dict, dict]:
    """Pad variable public text/images and dynamic action sets without exposing padding."""
    device = torch.device(device)
    observations = [row["observation"] for row in features]
    candidates = [row["candidates"] for row in features]
    obs = {key: torch.stack([row[key] for row in observations]).to(device)
           for key in ("numeric", "timing")}
    for modality in ("text", "image"):
        available = [row[modality] for row in observations if modality in row]
        if not available:
            continue
        length, width = max(part.shape[0] for part in available), available[0].shape[-1]
        tensor = torch.zeros(len(features), length, width)
        present = torch.zeros(len(features), length, dtype=torch.bool)
        for index, row in enumerate(observations):
            if modality in row:
                value = row[modality]
                tensor[index, :len(value)] = value
                present[index, :len(value)] = True
        obs[modality], obs[modality + "_present"] = tensor.to(device), present.to(device)
    length = max((len(row.get("measurement_numeric", [])) for row in observations), default=0)
    if length:
        width = next(row["measurement_text"].shape[-1] for row in observations if "measurement_text" in row)
        for key, shape in (("measurement_text", (len(features), length, width)),
                           ("measurement_numeric", (len(features), length, 8)),
                           ("measurement_age", (len(features), length))):
            value = torch.zeros(shape)
            for index, row in enumerate(observations):
                if key in row:
                    value[index, :len(row[key])] = row[key]
            obs[key] = value.to(device)
        obs["measurement_present"] = torch.tensor(
            [[j < len(row.get("measurement_numeric", [])) for j in range(length)] for row in observations],
            dtype=torch.bool, device=device)
    count = max(len(row["text"]) for row in candidates)
    actions = {key: torch.zeros(len(features), count, candidates[0][key].shape[-1], device=device)
               for key in ("text", "numeric")}
    actions["present"] = torch.zeros(len(features), count, dtype=torch.bool, device=device)
    if any("measurement_relations" in row for row in candidates):
        actions["measurement_relations"] = torch.zeros(len(features), count, length, 8, device=device)
    for index, row in enumerate(candidates):
        n = len(row["text"])
        for key in ("text", "numeric"):
            actions[key][index, :n] = row[key].to(device)
        actions["present"][index, :n] = True
        if "measurement_relations" in row:
            relations = row["measurement_relations"]
            actions["measurement_relations"][index, :n, :relations.shape[1]] = relations.to(device)
    return obs, actions
