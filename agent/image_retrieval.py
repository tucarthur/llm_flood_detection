"""Image-exemplar RAG tool: retrieves visually similar labeled reference images from
the season-masked, class-balanced DINOv2 embedding bank built in
knowledge_base/build_image_bank.ipynb (see knowledge_base/image_bank_index/).

Unlike the text retriever (agent/retrieval.py), the query isn't a string the model
chooses -- it's the image already being classified, embedded here each time the tool
is called. Masking of the current example's own season (the leave-one-season-out test
fold) happens automatically inside dispatch(), not something the model controls, since
letting the model pick which season to exclude would defeat the point.
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import chromadb

INDEX_DIR = Path(__file__).parent.parent / "knowledge_base" / "image_bank_index"
COLLECTION_NAME = "flood_image_examples"
# The bank's own images (built by knowledge_base/build_image_bank.ipynb; download
# locally with `python -m scripts.download_bank_images`) -- NOT data/raw/test_images
# or sample_images, which only happen to overlap on the bank's "rare" rows and are
# missing all 796 FPS-selected "low_diverse" exemplars.
DEFAULT_IMAGES_DIR = Path(__file__).parent.parent / "knowledge_base" / "image_bank_images"

RETRIEVE_SIMILAR_EXAMPLES_TOOL_SCHEMA = {
    "name": "retrieve_similar_examples",
    "description": (
        "Retrieve labeled reference images from past classifications that are visually "
        "similar to the current camera frame (found via DINOv2 embedding similarity over "
        "a season-masked, class-balanced exemplar bank -- images from the current "
        "example's own season are automatically excluded to avoid leaking the answer). "
        "Use this to compare the current frame's waterline against confirmed examples of "
        "each category, especially to distinguish borderline medium/high/flood cases."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "n_results": {"type": "integer", "description": "Number of similar examples to return (default 3)."},
        },
        "required": [],
    },
}

def _build_preprocess():
    # Deferred import: torch/torchvision are an optional dependency (`pip install
    # -e ".[image-rag]"`) -- importing this module (e.g. just for the tool schema
    # constant) must not require them, only actually using the retriever does.
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ImageBankRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR, images_dir: Path = DEFAULT_IMAGES_DIR):
        import torch

        if not index_dir.exists():
            raise FileNotFoundError(
                f"No image bank at {index_dir} -- build it with "
                "knowledge_base/build_image_bank.ipynb (on Kaggle) and unzip the output here."
            )
        client = chromadb.PersistentClient(path=str(index_dir))
        self.collection = client.get_collection(COLLECTION_NAME)
        self.images_dir = images_dir
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._preprocess = _build_preprocess()
        self._model = None  # lazy: don't pay the DINOv2 download/load cost for runs
        # that never actually call this tool (e.g. a text-RAG-only ablation cell).

    def _dino(self):
        import torch

        if self._model is None:
            self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self._model.eval().to(self._device)
        return self._model

    def _embed_image(self, image_path: str) -> list[float]:
        import torch
        from PIL import Image

        with torch.no_grad():
            with Image.open(image_path) as img:
                tensor = self._preprocess(img.convert("RGB")).unsqueeze(0).to(self._device)
            return self._dino()(tensor)[0].cpu().tolist()

    def _local_image_data(self, bank_path: str) -> dict | None:
        """bank_path is the CSV-style relative path stored in the bank's metadata
        (e.g. '2018/11/10/x-SHOP.jpg'); resolves it to a local file using the same
        '/' -> '_' convention as data/enoe_images.py:local_image_path. Returns None if
        the exemplar's actual bytes aren't available locally (e.g. running against the
        dev sample rather than the full downloaded dataset). Returns a neutral
        {media_type, data} pair rather than an API-specific content block -- the
        calling classifier formats it for whichever LLM API it talks to."""
        local_path = self.images_dir / bank_path.replace("/", "_")
        if not local_path.exists():
            return None
        media_type = mimetypes.guess_type(local_path.name)[0] or "image/jpeg"
        data = base64.standard_b64encode(local_path.read_bytes()).decode("utf-8")
        return {"media_type": media_type, "data": data}

    def retrieve_similar_examples(self, image_path: str, exclude_season: str, n_results: int = 3) -> dict:
        query_embedding = self._embed_image(image_path)
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"season": {"$ne": exclude_season}},
        )
        examples = [
            {
                "label": meta["label"],
                "season": meta["season"],
                "place": meta["place"],
                "is_night": meta["is_night"],
                "path": meta["path"],
            }
            for meta in res["metadatas"][0]
        ]
        return {"examples": examples}

    def dispatch(self, tool_name: str, tool_input: dict, *, image_path: str, exclude_season: str) -> tuple[dict, list[dict]]:
        """Returns (json-serializable result, extra {media_type, data} image payloads
        for exemplars whose bytes are resolvable locally). Each payload is neutral,
        not tied to any one LLM API's content-block schema -- that exemplar's result
        entry is flagged 'image_available_locally': False when unavailable, rather
        than silently dropping it."""
        if tool_name != "retrieve_similar_examples":
            return {"error": f"unknown tool {tool_name}"}, []

        result = self.retrieve_similar_examples(image_path, exclude_season, **tool_input)
        image_payloads = []
        for ex in result["examples"]:
            payload = self._local_image_data(ex["path"])
            if payload:
                image_payloads.append(payload)
            ex["image_available_locally"] = payload is not None
        return result, image_payloads


# Two hand-picked calibration sets (ablation: does a fixed, curated few-shot set that
# clearly illustrates the low/medium/high/flood *boundaries* beat similarity-based
# retrieval, which just finds visually-similar frames without teaching the threshold?
# See ImageBankRetriever above for the similarity-based baseline this is compared
# against). Each set is one rising/receding water event at a single camera, so the
# only thing that changes between images is the water level itself -- the clearest
# possible illustration of "this is the difference between medium/high/flood".
# Two sets (different seasons) exist so exclude_season can always drop one and keep
# the other, preserving the same leave-one-season-out discipline as similarity search.
_FEWSHOT_SETS = {
    "2018-2019": [
        {"label": "low", "path": "2019/01/19/20190119_082529-SHOP.jpg", "season": "2018-2019", "place": "SHOP", "is_night": False},
        {"label": "medium", "path": "2018/11/10/20181110_171815-SHOP.jpg", "season": "2018-2019", "place": "SHOP", "is_night": False},
        {"label": "high", "path": "2018/11/10/20181110_161626-SHOP.jpg", "season": "2018-2019", "place": "SHOP", "is_night": False},
        {"label": "flood", "path": "2018/11/10/20181110_163153-SHOP.jpg", "season": "2018-2019", "place": "SHOP", "is_night": False},
    ],
    "2021-2022": [
        {"label": "low", "path": "2022/02/11/20220211_080702-SHOP2.jpg", "season": "2021-2022", "place": "SHOP2", "is_night": False},
        {"label": "medium", "path": "2021/11/10/20211110_163628-SHOP2.jpg", "season": "2021-2022", "place": "SHOP2", "is_night": False},
        {"label": "high", "path": "2021/11/10/20211110_164220-SHOP2.jpg", "season": "2021-2022", "place": "SHOP2", "is_night": False},
        {"label": "flood", "path": "2021/11/10/20211110_165407-SHOP2.jpg", "season": "2021-2022", "place": "SHOP2", "is_night": False},
    ],
}
_FEWSHOT_SET_ORDER = ["2018-2019", "2021-2022"]  # deterministic tie-break when neither matches exclude_season


class EpisodeExemplarProvider:
    """Serves one episode's support set from a manifest built by data/episodes.py.

    This is the provider for the K-shot arm. Unlike ImageBankRetriever it does no
    embedding and no similarity search: the support set is a random, event-disjoint draw
    fixed in advance, which is what makes the K curve an episodic average rather than a
    property of one hand-picked set.

    `exclude_season` selects which fold's support set to use -- it is the query's own
    season, so the support set returned is always drawn from the other three. Exemplars
    arrive pre-sorted ascending by severity from the manifest.
    """

    def __init__(self, manifest_path: str | Path, episode_index: int = 0, images_dir: Path = DEFAULT_IMAGES_DIR):
        import json

        self.manifest = json.load(open(manifest_path))
        self.manifest_path = str(manifest_path)
        if not 0 <= episode_index < len(self.manifest["episodes"]):
            raise ValueError(
                f"episode {episode_index} out of range: manifest has "
                f"{len(self.manifest['episodes'])} episodes"
            )
        self.episode_index = episode_index
        self.supports = self.manifest["episodes"][episode_index]["supports"]
        self.k = self.manifest["k"]
        self.images_dir = images_dir

    def _local_image_data(self, bank_path: str) -> dict | None:
        local_path = self.images_dir / bank_path.replace("/", "_")
        if not local_path.exists():
            return None
        media_type = mimetypes.guess_type(local_path.name)[0] or "image/jpeg"
        data = base64.standard_b64encode(local_path.read_bytes()).decode("utf-8")
        return {"media_type": media_type, "data": data}

    def support_paths(self, season: str) -> list[str]:
        """The exact bank paths used for a fold -- recorded in the run's meta.json so a
        cell can be traced to the images it actually saw."""
        return [ex["path"] for ex in self.supports[season]]

    def dispatch(self, tool_name: str, tool_input: dict, *, image_path: str, exclude_season: str) -> tuple[dict, list[dict]]:
        support = self.supports.get(exclude_season)
        if support is None:
            raise KeyError(
                f"no support set for held-out season {exclude_season!r} in {self.manifest_path}"
            )
        examples, payloads = [], []
        for ex in support:
            entry = dict(ex)
            payload = self._local_image_data(entry["path"])
            if payload:
                payloads.append(payload)
            entry["image_available_locally"] = payload is not None
            examples.append(entry)
        return {"examples": examples}, payloads


class FixedFewShotExemplarProvider:
    """Drop-in alternative to ImageBankRetriever: instead of embedding the query image
    and finding similar frames, always returns the same curated low/medium/high/flood
    calibration set (see _FEWSHOT_SETS), picking whichever of the two prepared sets
    doesn't match the current example's season. Exposes the same dispatch() signature
    so agent/classifier.py doesn't need to know which strategy is in play."""

    def __init__(self, images_dir: Path = DEFAULT_IMAGES_DIR):
        self.images_dir = images_dir

    def _local_image_data(self, bank_path: str) -> dict | None:
        local_path = self.images_dir / bank_path.replace("/", "_")
        if not local_path.exists():
            return None
        media_type = mimetypes.guess_type(local_path.name)[0] or "image/jpeg"
        data = base64.standard_b64encode(local_path.read_bytes()).decode("utf-8")
        return {"media_type": media_type, "data": data}

    def retrieve_fewshot_examples(self, exclude_season: str) -> dict:
        chosen_season = next(
            (s for s in _FEWSHOT_SET_ORDER if s != exclude_season), _FEWSHOT_SET_ORDER[0]
        )
        examples = [dict(ex) for ex in _FEWSHOT_SETS[chosen_season]]
        return {"examples": examples}

    def dispatch(self, tool_name: str, tool_input: dict, *, image_path: str, exclude_season: str) -> tuple[dict, list[dict]]:
        result = self.retrieve_fewshot_examples(exclude_season)
        image_payloads = []
        for ex in result["examples"]:
            payload = self._local_image_data(ex["path"])
            if payload:
                image_payloads.append(payload)
            ex["image_available_locally"] = payload is not None
        return result, image_payloads
