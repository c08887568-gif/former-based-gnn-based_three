import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from utils.threading_config import (
    apply_torch_thread_config,
    configure_default_threads,
    get_runtime_thread_count,
)

configure_default_threads()

import matplotlib.pyplot as plt
import numpy as np
import torch
apply_torch_thread_config(torch)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Data
from sklearn.metrics import classification_report

from dataset import CachedGraphDataset, GraphDataset
from fieldroaddatapipeline.dataloader import FieldRoadDataLoader
from models.Encoder import VIT_GIN_Parallel
from utils.utils import WarmupCosineLR, get_default_device, to_edge_index


TOTAL_EPOCHS = 100


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def normalize_optional_path(value):
    if value is None:
        return None
    if value in ("", "None", "none", "null", "NULL"):
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the baseline VIT+GIN model.")
    parser.add_argument("--use_pretrain", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--dry_run", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--skip_test", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--cache_dir", default="cache/wheat_non_iid")
    parser.add_argument("--graph_cache_path", default=None)
    parser.add_argument(
        "--pretrain_mode",
        choices=["current", "edge_weight", "edge_type_weight"],
        default="current",
    )
    parser.add_argument(
        "--segment_context_mode",
        choices=["none", "msc"],
        default="none",
    )
    return parser.parse_args()


def auto_run_name():
    return datetime.now().strftime("fine_tune_%Y%m%d_%H%M%S")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_config(args, run_name, run_dir):
    pretrained_path = normalize_optional_path(args.pretrained_path)
    effective_pretrained_path = pretrained_path if args.use_pretrain else None
    graph_cache_path = normalize_optional_path(args.graph_cache_path)
    return dict(
        use_pretrain=args.use_pretrain,
        pretrained_path=pretrained_path,
        effective_pretrained_path=effective_pretrained_path,
        run_name=run_name,
        run_dir=str(run_dir),
        dry_run=args.dry_run,
        skip_test=args.skip_test,
        total_epochs=args.epochs,
        cache_dir=args.cache_dir,
        graph_cache_path=graph_cache_path,
        pretrain_mode=args.pretrain_mode,
        segment_context_mode=args.segment_context_mode,
        runtime_num_threads=get_runtime_thread_count(),
    )


def write_run_metadata(run_dir, config):
    write_json(run_dir / "config_resolved.json", config)
    command = " ".join(sys.argv)
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


GRAPH_DEDUP_AUDIT_FIELDS = [
    "run_name",
    "split",
    "epoch",
    "batch_id",
    "trace_id",
    "original_edges",
    "extra_edges_before_dedup",
    "duplicate_edges_removed",
    "merged_edges_after_dedup",
]

SEGMENT_CONTEXT_AUDIT_FIELDS = [
    "epoch",
    "segment_scale",
    "train_loss",
    "valid_loss",
    "valid_accuracy",
    "valid_macro_f1",
    "valid_road_f1",
    "valid_field_f1",
    "fused_norm_mean_before_msc",
    "fused_norm_mean_after_msc",
    "context_norm_mean",
    "context_to_fused_ratio",
]


def initialize_graph_dedup_audit(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRAPH_DEDUP_AUDIT_FIELDS)
        writer.writeheader()


def append_graph_dedup_audit(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRAPH_DEDUP_AUDIT_FIELDS)
        writer.writerow(row)


def initialize_segment_context_audit(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENT_CONTEXT_AUDIT_FIELDS)
        writer.writeheader()


def append_segment_context_audit(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENT_CONTEXT_AUDIT_FIELDS)
        writer.writerow(row)


def valid_metric_summary(predictions, labels, epoch, valid_loss, valid_acc):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions)
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    predictions = predictions.reshape(-1)
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["road", "field"],
        output_dict=True,
        zero_division=0,
    )
    return dict(
        best_epoch=epoch,
        valid_loss=float(valid_loss),
        valid_accuracy=float(valid_acc),
        valid_macro_f1=float(report["macro avg"]["f1-score"]),
        valid_road_f1=float(report["road"]["f1-score"]),
        valid_field_f1=float(report["field"]["f1-score"]),
        pred_road_rate=float((predictions == 0).mean()) if len(predictions) else 0.0,
    )


def test_metric_summary(predictions, labels, epoch):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions)
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    predictions = predictions.reshape(-1)
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["road", "field"],
        output_dict=True,
        zero_division=0,
    )
    pred_road_rate = float((predictions == 0).mean()) if len(predictions) else 0.0
    pred_field_rate = float((predictions == 1).mean()) if len(predictions) else 0.0
    return dict(
        test_epoch=epoch,
        test_accuracy=float((predictions == labels).mean()) if len(predictions) else 0.0,
        test_macro_f1=float(report["macro avg"]["f1-score"]),
        test_road_f1=float(report["road"]["f1-score"]),
        test_field_f1=float(report["field"]["f1-score"]),
        pred_road_rate=pred_road_rate,
        pred_field_rate=pred_field_rate,
        collapse_flag=bool(pred_road_rate < 0.05 or pred_field_rate < 0.05),
    )


def disabled_pretrain_audit(config):
    return dict(
        use_pretrain=False,
        pretrained_path=config["pretrained_path"],
        checkpoint_exists=False,
        encoder_keys_found=[],
        encoder_keys_found_count=0,
        keys_loaded=[],
        keys_loaded_count=0,
        missing_keys=[],
        missing_keys_count=0,
        unexpected_keys=[],
        unexpected_keys_count=0,
        shape_mismatch_keys=[],
        shape_mismatch_keys_count=0,
        load_success=False,
        reason="pretraining disabled",
    )


def missing_pretrain_audit(config):
    return dict(
        use_pretrain=True,
        pretrained_path=config["pretrained_path"],
        checkpoint_exists=False,
        encoder_keys_found=[],
        encoder_keys_found_count=0,
        keys_loaded=[],
        keys_loaded_count=0,
        missing_keys=[],
        missing_keys_count=0,
        unexpected_keys=[],
        unexpected_keys_count=0,
        shape_mismatch_keys=[],
        shape_mismatch_keys_count=0,
        load_success=False,
        reason="PRETRAIN_CHECKPOINT_NOT_FOUND",
    )


def inspect_pretrain_load(config, model, load_error=None):
    if not config["use_pretrain"]:
        return disabled_pretrain_audit(config)

    checkpoint_path = Path(config["pretrained_path"])
    audit = dict(
        use_pretrain=True,
        pretrained_path=config["pretrained_path"],
        checkpoint_exists=checkpoint_path.exists(),
        encoder_keys_found=[],
        encoder_keys_found_count=0,
        keys_loaded=[],
        keys_loaded_count=0,
        missing_keys=[],
        missing_keys_count=0,
        unexpected_keys=[],
        unexpected_keys_count=0,
        shape_mismatch_keys=[],
        shape_mismatch_keys_count=0,
        load_success=False,
    )
    if load_error is not None:
        audit["load_error"] = str(load_error)
        return audit
    if not checkpoint_path.exists():
        audit["reason"] = "PRETRAIN_CHECKPOINT_NOT_FOUND"
        return audit

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        audit["reason"] = "checkpoint is not a state_dict"
        return audit

    encoder_items = {k: v for k, v in checkpoint.items() if k.startswith("encoder.")}
    stripped_items = {k.replace("encoder.", "", 1): v for k, v in encoder_items.items()}
    model_state = model.state_dict()

    keys_loaded = []
    unexpected_keys = []
    shape_mismatch_keys = []
    for key, value in stripped_items.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if hasattr(value, "shape") and hasattr(model_state[key], "shape") and tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch_keys.append(key)
            continue
        keys_loaded.append(key)

    missing_keys = [key for key in model_state.keys() if key not in keys_loaded]
    audit.update(
        encoder_keys_found=sorted(encoder_items.keys()),
        encoder_keys_found_count=len(encoder_items),
        keys_loaded=sorted(keys_loaded),
        keys_loaded_count=len(keys_loaded),
        missing_keys=sorted(missing_keys),
        missing_keys_count=len(missing_keys),
        unexpected_keys=sorted(unexpected_keys),
        unexpected_keys_count=len(unexpected_keys),
        shape_mismatch_keys=sorted(shape_mismatch_keys),
        shape_mismatch_keys_count=len(shape_mismatch_keys),
        load_success=len(keys_loaded) > 0 and load_error is None,
    )
    return audit


def build_model(config, device):
    pretrained_path = config["effective_pretrained_path"]
    model = VIT_GIN_Parallel(
        img_size=43,
        patch_size=1,
        in_chans=1,
        num_classes=2,
        embed_dim=108,
        depth=1,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.3,
        drop_path_rate=0.1,
        pretrained_path=pretrained_path,
        pretrain_mode=config["pretrain_mode"],
        segment_context_mode=config["segment_context_mode"],
    ).to(device)
    return model


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


def load_graph_cache_split(graph_cache_path, split, required=True):
    if graph_cache_path is None:
        return None
    cache_file = Path(graph_cache_path) / f"{split}.pt"
    if not cache_file.exists():
        if required:
            raise FileNotFoundError(f"GRAPH_CACHE_SPLIT_NOT_FOUND: {cache_file}")
        return None

    raw_cache = torch.load(cache_file, map_location="cpu")
    if isinstance(raw_cache, dict) and "items" in raw_cache:
        raw_cache = raw_cache["items"]

    grouped = {}
    if isinstance(raw_cache, dict):
        iterable = raw_cache.values()
    elif isinstance(raw_cache, list):
        iterable = raw_cache
    else:
        raise TypeError(f"UNSUPPORTED_GRAPH_CACHE_FORMAT: {cache_file}")

    for item in iterable:
        if not isinstance(item, dict) or "trace_id" not in item:
            raise TypeError(f"INVALID_GRAPH_CACHE_ITEM: {cache_file}")
        key = normalize_trace_id(item["trace_id"])
        grouped.setdefault(key, []).append(item)
    return grouped


def find_graph_cache_item(graph_cache, trace_id, points):
    if graph_cache is None:
        return None
    items = graph_cache.get(normalize_trace_id(trace_id))
    if not items:
        return None
    if len(items) == 1:
        return items[0]

    points_hash = tensor_content_hash(points)
    for item in items:
        if item.get("points_hash") == points_hash:
            return item
    return items[0]


def deduplicate_directed_edges(
    original_edge_index,
    extra_edge_index,
    original_edge_weight=None,
    extra_edge_weight=None,
):
    original_edges = int(original_edge_index.shape[1])
    extra_edges_before_dedup = int(extra_edge_index.shape[1])
    has_edge_weight = original_edge_weight is not None or extra_edge_weight is not None
    if has_edge_weight:
        if original_edge_weight is None:
            original_edge_weight = torch.ones(
                original_edges,
                dtype=torch.float32,
                device=original_edge_index.device,
            )
        else:
            original_edge_weight = original_edge_weight.to(torch.float32).to(original_edge_index.device)
        if extra_edge_weight is None:
            extra_edge_weight = torch.ones(
                extra_edges_before_dedup,
                dtype=torch.float32,
                device=extra_edge_index.device,
            )
        else:
            extra_edge_weight = extra_edge_weight.to(torch.float32).to(extra_edge_index.device)

    if extra_edges_before_dedup == 0:
        return original_edge_index, original_edge_weight if has_edge_weight else None, dict(
            original_edges=original_edges,
            extra_edges_before_dedup=0,
            duplicate_edges_removed=0,
            merged_edges_after_dedup=original_edges,
        )

    original_cpu = original_edge_index.detach().cpu()
    extra_cpu = extra_edge_index.detach().cpu()
    seen = set()
    keep_extra_indices = []

    for src, dst in zip(original_cpu[0].tolist(), original_cpu[1].tolist()):
        seen.add((int(src), int(dst)))

    for idx, (src, dst) in enumerate(zip(extra_cpu[0].tolist(), extra_cpu[1].tolist())):
        edge = (int(src), int(dst))
        if edge in seen:
            continue
        seen.add(edge)
        keep_extra_indices.append(idx)

    if keep_extra_indices:
        index = torch.tensor(keep_extra_indices, dtype=torch.long, device=extra_edge_index.device)
        unique_extra_edge_index = extra_edge_index.index_select(1, index)
        merged_edge_index = torch.cat([original_edge_index, unique_extra_edge_index], dim=1)
        if has_edge_weight:
            unique_extra_edge_weight = extra_edge_weight.index_select(0, index)
            merged_edge_weight = torch.cat([original_edge_weight, unique_extra_edge_weight], dim=0)
        else:
            merged_edge_weight = None
    else:
        merged_edge_index = original_edge_index
        merged_edge_weight = original_edge_weight if has_edge_weight else None

    duplicate_edges_removed = extra_edges_before_dedup - len(keep_extra_indices)
    return merged_edge_index, merged_edge_weight, dict(
        original_edges=original_edges,
        extra_edges_before_dedup=extra_edges_before_dedup,
        duplicate_edges_removed=duplicate_edges_removed,
        merged_edges_after_dedup=int(merged_edge_index.shape[1]),
    )


def merge_graph_cache_edges(
    adjs,
    graph_cache,
    trace_id,
    points,
    device,
    audit_path=None,
    run_name=None,
    split=None,
    epoch=None,
    batch_id=None,
):
    edge_index = to_edge_index(adjs, device)
    cache_item = find_graph_cache_item(graph_cache, trace_id, points)
    if cache_item is None:
        return edge_index, None

    original_edge_weight = cache_item.get("original_edge_weight")
    if original_edge_weight is not None:
        original_edge_weight = torch.as_tensor(original_edge_weight).clone().detach()
        if original_edge_weight.numel() != edge_index.shape[1]:
            original_edge_weight = None

    extra_edge_index = cache_item.get("extra_edge_index")
    if extra_edge_index is None:
        return edge_index, original_edge_weight.to(torch.float32).to(device) if original_edge_weight is not None else None
    extra_edge_index = torch.as_tensor(extra_edge_index)
    if extra_edge_index.numel() == 0:
        return edge_index, original_edge_weight.to(torch.float32).to(device) if original_edge_weight is not None else None

    extra_edge_weight = cache_item.get("extra_edge_weight")
    if extra_edge_weight is not None:
        extra_edge_weight = torch.as_tensor(extra_edge_weight).clone().detach()
        if extra_edge_weight.numel() != extra_edge_index.shape[1]:
            extra_edge_weight = None

    extra_edge_index = extra_edge_index.clone().detach().to(torch.long).to(device)
    num_nodes = int(points.shape[0])
    valid_mask = (
        (extra_edge_index[0] >= 0)
        & (extra_edge_index[1] >= 0)
        & (extra_edge_index[0] < num_nodes)
        & (extra_edge_index[1] < num_nodes)
    )
    extra_edge_index = extra_edge_index[:, valid_mask]
    if extra_edge_weight is not None:
        extra_edge_weight = extra_edge_weight[valid_mask.detach().cpu()]
    if extra_edge_index.numel() == 0:
        return edge_index, original_edge_weight.to(torch.float32).to(device) if original_edge_weight is not None else None
    merged_edge_index, merged_edge_weight, stats = deduplicate_directed_edges(
        edge_index,
        extra_edge_index,
        original_edge_weight=original_edge_weight.to(device) if original_edge_weight is not None else None,
        extra_edge_weight=extra_edge_weight.to(device) if extra_edge_weight is not None else None,
    )
    if audit_path is not None:
        append_graph_dedup_audit(
            audit_path,
            dict(
                run_name=run_name,
                split=split,
                epoch=epoch,
                batch_id=batch_id,
                trace_id=normalize_trace_id(trace_id),
                **stats,
            ),
        )
    return merged_edge_index, merged_edge_weight


def build_data(points, edge_index, labels, edge_weight=None):
    data = Data(x=points, edge_index=edge_index, y=labels)
    if edge_weight is not None:
        data.edge_weight = edge_weight.to(torch.float32).to(points.device)
    return data


def evaluate_test_once(
    model,
    test_loader,
    test_graph_cache,
    device,
    audit_path=None,
    run_name=None,
    epoch=None,
):
    model.eval()
    all_predictions = []
    all_labels = []
    test_num_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_id, (points, labels, adjs, trace_id) in enumerate(test_loader()):
            points = points.clone().detach().to(torch.float32).squeeze(0).to(device)
            labels = labels.clone().detach().to(torch.int64).squeeze().to(device)
            edge_index, edge_weight = merge_graph_cache_edges(
                adjs,
                test_graph_cache,
                trace_id,
                points,
                device,
                audit_path=audit_path,
                run_name=run_name,
                split="test",
                epoch=epoch,
                batch_id=batch_id,
            )
            data = build_data(points, edge_index, labels, edge_weight=edge_weight)
            pred = model.test_step(data)
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            test_num_samples += points.shape[0]
    test_time = time.time() - start_time
    test_fps = test_num_samples / max(test_time, 1e-12)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    class_result = model.calculate_classification_metrics(all_predictions, all_labels)
    return class_result, test_metric_summary(all_predictions, all_labels, epoch), test_time, test_fps


def main():
    args = parse_args()
    run_name = args.run_name or auto_run_name()
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = resolve_config(args, run_name, run_dir)
    write_run_metadata(run_dir, config)

    if config["use_pretrain"] and (
        config["pretrained_path"] is None or not Path(config["pretrained_path"]).exists()
    ):
        write_json(run_dir / "pretrain_load_audit.json", missing_pretrain_audit(config))
        raise SystemExit("PRETRAIN_CHECKPOINT_NOT_FOUND")

    device = get_default_device()
    try:
        model = build_model(config, device)
    except Exception as exc:
        write_json(run_dir / "pretrain_load_audit.json", inspect_pretrain_load(config, None, load_error=exc))
        raise
    pretrain_audit = inspect_pretrain_load(config, model)
    write_json(run_dir / "pretrain_load_audit.json", pretrain_audit)

    if config["dry_run"]:
        print(f"DRY_RUN_COMPLETE run_dir={run_dir}")
        return

    from utils.logger import Logger

    logger = Logger(model_name="VIT_GIN_Parallel", dataset_kind="paddy_small")
    logger.log_environment_info()
    cache_dir = Path(config["cache_dir"])
    if (cache_dir / "train.pt").exists() and (cache_dir / "valid.pt").exists():
        train_dataset = CachedGraphDataset(cache_dir / "train.pt", mode='train')
        valid_dataset = CachedGraphDataset(cache_dir / "valid.pt", mode='valid')
    else:
        train_path = dict(
            gnss="../dataset_5/dataset_high/paddy/sampled_paddy_43",
            adj="../dataset_5/dataset_high/paddy/sampled_paddy_adj",
            json="../dataset_5/dataset_high/paddy/Non-Identically_Distributed_Coco/sampled_paddy_43_train.json"
        )
        train_dataset = GraphDataset(train_path, mode='train', num_workers=2, max_len=1000, drop_rate=0)
        valid_path = dict(
            gnss="../dataset_5/dataset_high/paddy/sampled_paddy_43",
            adj="../dataset_5/dataset_high/paddy/sampled_paddy_adj",
            json="../dataset_5/dataset_high/paddy/Non-Identically_Distributed_Coco/sampled_paddy_43_valid.json"
        )
        valid_dataset = GraphDataset(valid_path, mode='valid', num_workers=2, max_len=1000, drop_rate=0)
    test_dataset = None
    test_loader = None
    if not config["skip_test"]:
        if (cache_dir / "test.pt").exists():
            test_dataset = CachedGraphDataset(cache_dir / "test.pt", mode='test')
        else:
            test_path = dict(
                gnss="../dataset_5/dataset_high/paddy/sampled_paddy_43",
                adj="../dataset_5/dataset_high/paddy/sampled_paddy_adj",
                json="../dataset_5/dataset_high/paddy/Non-Identically_Distributed_Coco/sampled_paddy_43_test.json"
            )
            test_dataset = GraphDataset(test_path, mode='test', num_workers=2, max_len=1000, drop_rate=0)
    train_graph_cache = load_graph_cache_split(config["graph_cache_path"], "train", required=True)
    valid_graph_cache = load_graph_cache_split(config["graph_cache_path"], "valid", required=True)
    test_graph_cache = load_graph_cache_split(config["graph_cache_path"], "test", required=False)
    graph_dedup_audit_path = None
    if config["graph_cache_path"] is not None:
        graph_dedup_audit_path = Path("diagnostics") / "graph_cache_dedup_audit.csv"
        initialize_graph_dedup_audit(graph_dedup_audit_path)
    segment_context_audit_path = None
    if config["segment_context_mode"] == "msc":
        segment_context_audit_path = Path("diagnostics") / f"{run_name}_segment_context_audit.csv"
        initialize_segment_context_audit(segment_context_audit_path)
    logger.log_dataset_info(train_dataset, valid_dataset, test_dataset)
    # Create data loaders using PyTorch DataLoader
    train_loader = FieldRoadDataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    valid_loader = FieldRoadDataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=True)
    if test_dataset is not None:
        test_loader = FieldRoadDataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    ####################################超参数
    #torch.autograd.set_detect_anomaly(True)
    total_epochs = config["total_epochs"]
    # Set the random seed if needed
    #torch.manual_seed(2023)
    # Initialize your model, optimizer, and LR scheduler
    logger.log_model_info(model)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_start_lr=0.0001,
        end_lr=0.00005,
        warmup_epochs=5,
        total_epochs=total_epochs
    )
    loss_config = dict(
        label_smoothing=dict(
            epsilon=0.1,
        ),
        collaborative_training=dict(
            tau=0.5,
            alpha=0.7,
        ),
        orthogonal=dict(
            reg=1e-5,
        )
    )
    logger.log_training_config_info(
        optimizer,
        scheduler,
        dict(
            total_epochs=total_epochs,
            device=device
        ),
        train_loader,
        valid_loader,
        test_loader
    )
    ##########################################训练
    best_train_acc = 0.0
    best_valid_acc = -1.0
    total_train_time = 0
    total_valid_time = 0
    total_test_time = 0
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    class_result_train = None
    class_result_valid = None
    class_result_test_valid = None
    train_log_rows = []
    best_valid_metrics = None
    best_test_metrics = None
    for pass_num in range(total_epochs):
        if config["segment_context_mode"] == "msc":
            model.reset_segment_context_statistics()
        model.train()
        epoch_start_time = time.time()
        train_loss_total = 0.0
        train_acc_total = 0.0
        num_samples = 0
        all_predictions = []
        all_labels = []
        for batch_id, (points, labels, adjs, trace_id) in enumerate(train_loader()):
            points = points.clone().detach().to(torch.float32).squeeze(0).to(device)
            labels = labels.clone().detach().to(torch.int64).squeeze().to(device)
            edge_index, edge_weight = merge_graph_cache_edges(
                adjs,
                train_graph_cache,
                trace_id,
                points,
                device,
                audit_path=graph_dedup_audit_path,
                run_name=run_name,
                split="train",
                epoch=pass_num + 1,
                batch_id=batch_id,
            )
            data = build_data(points, edge_index, labels, edge_weight=edge_weight)
            pred, loss, acc = model.train_step(data, labels, optimizer, loss_config)
            trajectory_length = points.shape[0]
            train_loss_total += loss * trajectory_length
            train_acc_total += acc * trajectory_length
            num_samples += trajectory_length
            all_predictions.extend(pred.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_train_loss = train_loss_total / num_samples
        train_losses.append(avg_train_loss)
        avg_train_acc = train_acc_total / num_samples
        train_accuracies.append(avg_train_acc)
        train_end_time = time.time()
        epoch_train_time = train_end_time - epoch_start_time
        total_train_time += epoch_train_time
        train_fps = num_samples / epoch_train_time
        if avg_train_acc > best_train_acc:
            best_train_acc = avg_train_acc
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            class_result_train = model.calculate_classification_metrics(all_predictions, all_labels)
        scheduler.step()
        with torch.no_grad():
            model.eval()
            valid_loss_total = 0.0
            valid_acc_total = 0.0
            num_samples = 0
            all_predictions = []
            all_labels = []
            for batch_id, (points, labels, adjs, trace_id) in enumerate(valid_loader()):
                points = points.clone().detach().to(torch.float32).squeeze(0).to(device)
                labels = labels.clone().detach().to(torch.int64).squeeze().to(device)
                edge_index, edge_weight = merge_graph_cache_edges(
                    adjs,
                    valid_graph_cache,
                    trace_id,
                    points,
                    device,
                    audit_path=graph_dedup_audit_path,
                    run_name=run_name,
                    split="valid",
                    epoch=pass_num + 1,
                    batch_id=batch_id,
                )
                data = build_data(points, edge_index, labels, edge_weight=edge_weight)
                pred, loss, acc = model.valid_step(data, labels, None)
                trajectory_length = points.shape[0]
                valid_loss_total += loss * trajectory_length
                valid_acc_total += acc * trajectory_length
                num_samples += trajectory_length
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            avg_valid_loss = valid_loss_total / num_samples
            valid_losses.append(avg_valid_loss)
            avg_valid_acc = valid_acc_total / num_samples
            valid_accuracies.append(avg_valid_acc)
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            epoch_valid_metrics = valid_metric_summary(
                all_predictions,
                all_labels,
                pass_num + 1,
                avg_valid_loss,
                avg_valid_acc,
            )
            if avg_valid_acc > best_valid_acc:
                best_valid_acc = avg_valid_acc
                class_result_valid = model.calculate_classification_metrics(all_predictions, all_labels)
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                best_valid_metrics = epoch_valid_metrics
            valid_end_time = time.time()
            epoch_valid_time = valid_end_time - train_end_time
            total_valid_time += epoch_valid_time
            valid_fps = num_samples / epoch_valid_time
            logger.log_training_info(
                pass_num,
                optimizer.param_groups[0]['lr'],
                avg_train_loss,
                avg_train_acc,
                avg_valid_loss,
                avg_valid_acc,
                best_valid_acc,
                epoch_train_time,
                train_fps,
                epoch_valid_time,
                valid_fps
            )
            train_log_rows.append(
                dict(
                    epoch=pass_num + 1,
                    train_loss=float(avg_train_loss),
                    train_accuracy=float(avg_train_acc),
                    valid_loss=float(avg_valid_loss),
                    valid_accuracy=float(avg_valid_acc),
                    best_valid_accuracy=float(best_valid_acc),
                    lr=float(optimizer.param_groups[0]['lr']),
                )
            )
            if segment_context_audit_path is not None:
                segment_stats = model.get_segment_context_statistics()
                append_segment_context_audit(
                    segment_context_audit_path,
                    dict(
                        epoch=pass_num + 1,
                        segment_scale=float(segment_stats.get("segment_scale", 0.0)),
                        train_loss=float(avg_train_loss),
                        valid_loss=float(avg_valid_loss),
                        valid_accuracy=float(avg_valid_acc),
                        valid_macro_f1=float(epoch_valid_metrics["valid_macro_f1"]),
                        valid_road_f1=float(epoch_valid_metrics["valid_road_f1"]),
                        valid_field_f1=float(epoch_valid_metrics["valid_field_f1"]),
                        fused_norm_mean_before_msc=float(segment_stats.get("fused_norm_mean_before_msc", 0.0)),
                        fused_norm_mean_after_msc=float(segment_stats.get("fused_norm_mean_after_msc", 0.0)),
                        context_norm_mean=float(segment_stats.get("context_norm_mean", 0.0)),
                        context_to_fused_ratio=float(segment_stats.get("context_to_fused_ratio", 0.0)),
                    ),
                )
    if not config["skip_test"] and test_loader is not None:
        best_model_path = run_dir / "best_model.pt"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_epoch = best_valid_metrics["best_epoch"] if best_valid_metrics is not None else total_epochs
        class_result_test_valid, best_test_metrics, test_time, test_fps = evaluate_test_once(
            model,
            test_loader,
            test_graph_cache,
            device,
            audit_path=graph_dedup_audit_path,
            run_name=run_name,
            epoch=test_epoch,
        )
        total_test_time += test_time
        logger.log_test_info(test_time, test_fps, class_result_test_valid)
    # Calculate average training and evaluation times
    avg_train_time = total_train_time / total_epochs
    avg_valid_time = total_valid_time / total_epochs
    avg_test_time = total_test_time if not config["skip_test"] else 0
    logger.log_end_of_training(avg_train_time, avg_valid_time, avg_test_time, class_result_train, class_result_valid,
                               None, class_result_test_valid)
    logger.log_start_of_outputs()
    write_csv(
        run_dir / "training_metrics.csv",
        ["epoch", "train_loss", "train_accuracy", "valid_loss", "valid_accuracy", "best_valid_accuracy", "lr"],
        train_log_rows,
    )
    if best_valid_metrics is not None:
        write_json(run_dir / "metrics_summary.json", best_valid_metrics)
    if best_test_metrics is not None:
        write_json(run_dir / "test_metrics_summary.json", best_test_metrics)
    # 训练完成后，绘制损失曲线
    logger.plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, is_save=True)
    logger.clean_up_logger()


if __name__ == "__main__":
    main()
