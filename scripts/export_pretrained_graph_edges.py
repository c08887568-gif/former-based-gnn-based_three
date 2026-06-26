import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.threading_config import (
    apply_torch_thread_config,
    configure_default_threads,
    get_runtime_thread_count,
)

configure_default_threads()

import torch
apply_torch_thread_config(torch)
import torch.nn.functional as F

from dataset import CachedGraphDataset
from models.Encoder import VIT_GIN_Parallel
from utils.utils import get_default_device, to_edge_index


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def parse_args():
    parser = argparse.ArgumentParser(description="Export extra graph edges discovered from a pretrained encoder.")
    parser.add_argument("--pretrained_path", required=True)
    parser.add_argument("--pretrain_mode", choices=["current", "edge_weight", "edge_type_weight"], required=True)
    parser.add_argument("--output_cache", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--split", default="train,valid")
    parser.add_argument("--dry_run", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--cache_dir", default="cache/wheat_non_iid")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--num_edge_types", type=int, default=4)
    parser.add_argument("--include_original_edge_weight", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--negative_edge_weight", choices=["zero", "filter"], default="zero")
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return get_default_device()
    return torch.device(name)


def normalize_trace_id(trace_id):
    if isinstance(trace_id, torch.Tensor):
        if trace_id.numel() == 1:
            return str(trace_id.item())
        return str(trace_id.detach().cpu().tolist())
    if isinstance(trace_id, (list, tuple)):
        if len(trace_id) == 1:
            return normalize_trace_id(trace_id[0])
        return str(tuple(normalize_trace_id(value) for value in trace_id))
    return str(trace_id)


def tensor_content_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha1()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parse_splits(value):
    splits = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"train", "valid", "test"}
    unknown = [split for split in splits if split not in valid]
    if unknown:
        raise SystemExit(f"UNSUPPORTED_SPLIT: {','.join(unknown)}")
    return splits


def build_model(args, device):
    return VIT_GIN_Parallel(
        img_size=43,
        patch_size=1,
        in_chans=1,
        num_classes=2,
        embed_dim=108,
        depth=1,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.3,
        drop_path_rate=0.1,
        pretrain_mode=args.pretrain_mode,
        num_edge_types=args.num_edge_types,
    ).to(device)


def load_pretrained_encoder(model, pretrained_path):
    checkpoint_path = Path(pretrained_path)
    if not checkpoint_path.exists():
        raise SystemExit("PRETRAIN_CHECKPOINT_NOT_FOUND")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise SystemExit("INVALID_PRETRAIN_CHECKPOINT")

    model_state = model.state_dict()
    loaded = {}
    for key, value in checkpoint.items():
        if not key.startswith("encoder."):
            continue
        model_key = key.replace("encoder.", "", 1)
        if model_key not in model_state:
            continue
        if hasattr(value, "shape") and tuple(value.shape) != tuple(model_state[model_key].shape):
            continue
        loaded[model_key] = value

    if not loaded:
        raise SystemExit("NO_ENCODER_KEYS_LOADED")

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=False)
    return sorted(loaded.keys())


def load_cached_split(cache_dir, split):
    cache_path = Path(cache_dir) / f"{split}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"CACHE_SPLIT_NOT_FOUND: {cache_path}")
    return CachedGraphDataset(cache_path, mode=split)


def trim_sample(points, edge_index, max_len):
    if max_len is None or max_len <= 0 or points.shape[0] <= max_len:
        return points, edge_index
    points = points[:max_len]
    keep = (edge_index[0] < max_len) & (edge_index[1] < max_len)
    return points, edge_index[:, keep]


def extract_graph_embeddings(model, points, edge_index, device):
    model.eval()
    with torch.no_grad():
        points = points.to(torch.float32).to(device)
        edge_index = edge_index.to(torch.long).to(device)
        x = points.view(-1, 1, 43, 1)
        _, _, _, graph_features, _, _ = model.forward_features(
            x,
            edge_index,
            mask_ratio_image=0,
            mask_ratio_graph=0,
        )
    return graph_features.detach()


def generate_extra_edges(embeddings, original_edge_index, topk):
    num_nodes = int(embeddings.shape[0])
    if topk <= 0 or num_nodes <= 1:
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        empty_weight = torch.empty((0,), dtype=torch.float32)
        return empty_edges, empty_weight

    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity = embeddings @ embeddings.t()
    similarity.fill_diagonal_(float("-inf"))

    original_edge_index = original_edge_index.to(torch.long).to(similarity.device)
    if original_edge_index.numel() > 0:
        valid = (
            (original_edge_index[0] >= 0)
            & (original_edge_index[1] >= 0)
            & (original_edge_index[0] < num_nodes)
            & (original_edge_index[1] < num_nodes)
        )
        original_edge_index = original_edge_index[:, valid]
        similarity[original_edge_index[0], original_edge_index[1]] = float("-inf")

    k = min(topk, num_nodes - 1)
    values, indices = torch.topk(similarity, k=k, dim=1)
    sources = torch.arange(num_nodes, device=similarity.device).view(-1, 1).expand_as(indices)
    keep = values > 0
    if not torch.any(keep):
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        empty_weight = torch.empty((0,), dtype=torch.float32)
        return empty_edges, empty_weight

    extra_edge_index = torch.stack([sources[keep], indices[keep]], dim=0).detach().cpu().to(torch.long)
    extra_edge_weight = values[keep].detach().cpu().to(torch.float32)
    return extra_edge_index, extra_edge_weight


def cosine_edge_weight(embeddings, edge_index, negative_action="zero", preserve_edges=True):
    if edge_index.numel() == 0:
        return torch.empty((0,), dtype=torch.float32), edge_index.detach().cpu().to(torch.long)

    embeddings = F.normalize(embeddings, p=2, dim=1)
    edge_index = edge_index.to(torch.long).to(embeddings.device)
    src, dst = edge_index[0], edge_index[1]
    weights = (embeddings[src] * embeddings[dst]).sum(dim=1)
    if negative_action == "filter" and not preserve_edges:
        keep = weights >= 0
        edge_index = edge_index[:, keep]
        weights = weights[keep]
    else:
        weights = weights.clamp_min(0)
    weights = weights.clamp(0, 1)
    return weights.detach().cpu().to(torch.float32), edge_index.detach().cpu().to(torch.long)


def infer_edge_type_prob(model, points, extra_edge_index, device):
    if extra_edge_index.numel() == 0:
        return None
    learner = getattr(getattr(model.graph, "conv1", None), "edge_type_weight_learner", None)
    if learner is None:
        return None
    with torch.no_grad():
        _, edge_type_prob = learner(
            points.to(torch.float32).to(device),
            extra_edge_index.to(torch.long).to(device),
        )
    return edge_type_prob.detach().cpu().to(torch.float32)


def build_cache_item(model, dataset, index, args, device):
    points, _labels, edge_index, trace_id = dataset[index]
    points = points.clone().detach().to(torch.float32)
    edge_index = to_edge_index(edge_index, torch.device("cpu")).clone().detach().to(torch.long)
    points, edge_index = trim_sample(points, edge_index, args.max_len)
    embeddings = extract_graph_embeddings(model, points, edge_index, device)
    extra_edge_index, extra_edge_weight = generate_extra_edges(embeddings, edge_index, args.topk)
    original_edge_weight = None
    if args.include_original_edge_weight:
        original_edge_weight, edge_index = cosine_edge_weight(
            embeddings,
            edge_index,
            negative_action=args.negative_edge_weight,
            preserve_edges=True,
        )
    extra_edge_type_prob = infer_edge_type_prob(model, points, extra_edge_index, device)

    item = dict(
        trace_id=normalize_trace_id(trace_id),
        sample_index=index,
        points_hash=tensor_content_hash(points),
        extra_edge_index=extra_edge_index,
        extra_edge_weight=extra_edge_weight,
        original_num_edges=int(edge_index.shape[1]),
        extra_num_edges=int(extra_edge_index.shape[1]),
    )
    if original_edge_weight is not None:
        item["original_edge_weight"] = original_edge_weight
        item["edge_weight_source"] = "pretrained_encoder_cosine_clamped"
    if extra_edge_type_prob is not None:
        item["extra_edge_type_prob"] = extra_edge_type_prob
    return item


def export_split(model, split, args, device):
    dataset = load_cached_split(args.cache_dir, split)
    if len(dataset) == 0:
        raise RuntimeError(f"EMPTY_SPLIT: {split}")

    if args.dry_run:
        item = build_cache_item(model, dataset, 0, args, device)
        print(
            "DRY_RUN "
            f"split={split} trace_id={item['trace_id']} "
            f"original_num_edges={item['original_num_edges']} "
            f"extra_num_edges={item['extra_num_edges']}"
        )
        return None

    items = []
    for index in range(len(dataset)):
        item = build_cache_item(model, dataset, index, args, device)
        items.append(item)
        if (index + 1) % 20 == 0:
            print(f"split={split} exported={index + 1}/{len(dataset)}")

    output_cache = Path(args.output_cache)
    output_cache.mkdir(parents=True, exist_ok=True)
    output_path = output_cache / f"{split}.pt"
    torch.save(items, output_path)
    print(f"WROTE {output_path} samples={len(items)}")
    return output_path


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = build_model(args, device)
    loaded_keys = load_pretrained_encoder(model, args.pretrained_path)

    output_cache = Path(args.output_cache)
    if not args.dry_run:
        output_cache.mkdir(parents=True, exist_ok=True)
        manifest = dict(
            command=" ".join(sys.argv),
            pretrained_path=args.pretrained_path,
            pretrain_mode=args.pretrain_mode,
            topk=args.topk,
            max_len=args.max_len,
            cache_dir=args.cache_dir,
            loaded_encoder_keys=len(loaded_keys),
            include_original_edge_weight=args.include_original_edge_weight,
            negative_edge_weight=args.negative_edge_weight,
            runtime_num_threads=get_runtime_thread_count(),
        )
        (output_cache / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    for split in parse_splits(args.split):
        export_split(model, split, args, device)
        if args.dry_run:
            break


if __name__ == "__main__":
    main()
