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
import torch
from PIL import Image
from torchvision import transforms

INDEX_DIR = Path(__file__).parent.parent / "knowledge_base" / "image_bank_index"
COLLECTION_NAME = "flood_image_examples"
DEFAULT_IMAGES_DIR = Path(__file__).parent.parent / "data" / "raw" / "sample_images"

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

_preprocess = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ImageBankRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR, images_dir: Path = DEFAULT_IMAGES_DIR):
        if not index_dir.exists():
            raise FileNotFoundError(
                f"No image bank at {index_dir} -- build it with "
                "knowledge_base/build_image_bank.ipynb (on Kaggle) and unzip the output here."
            )
        client = chromadb.PersistentClient(path=str(index_dir))
        self.collection = client.get_collection(COLLECTION_NAME)
        self.images_dir = images_dir
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None  # lazy: don't pay the DINOv2 download/load cost for runs
        # that never actually call this tool (e.g. a text-RAG-only ablation cell).

    def _dino(self):
        if self._model is None:
            self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self._model.eval().to(self._device)
        return self._model

    @torch.no_grad()
    def _embed_image(self, image_path: str) -> list[float]:
        with Image.open(image_path) as img:
            tensor = _preprocess(img.convert("RGB")).unsqueeze(0).to(self._device)
        return self._dino()(tensor)[0].cpu().tolist()

    def _local_image_block(self, bank_path: str) -> dict | None:
        """bank_path is the CSV-style relative path stored in the bank's metadata
        (e.g. '2018/11/10/x-SHOP.jpg'); resolves it to a local file using the same
        '/' -> '_' convention as data/enoe_images.py:local_image_path. Returns None if
        the exemplar's actual bytes aren't available locally (e.g. running against the
        dev sample rather than the full downloaded dataset)."""
        local_path = self.images_dir / bank_path.replace("/", "_")
        if not local_path.exists():
            return None
        media_type = mimetypes.guess_type(local_path.name)[0] or "image/jpeg"
        data = base64.standard_b64encode(local_path.read_bytes()).decode("utf-8")
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}

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
        """Returns (json-serializable result, extra image content blocks to append to
        the tool_result). The image list is empty for any exemplar whose bytes aren't
        resolvable locally -- that exemplar's result entry is flagged
        'image_available_locally': False instead of silently dropping it."""
        if tool_name != "retrieve_similar_examples":
            return {"error": f"unknown tool {tool_name}"}, []

        result = self.retrieve_similar_examples(image_path, exclude_season, **tool_input)
        image_blocks = []
        for ex in result["examples"]:
            block = self._local_image_block(ex["path"])
            if block:
                image_blocks.append(block)
            ex["image_available_locally"] = block is not None
        return result, image_blocks
