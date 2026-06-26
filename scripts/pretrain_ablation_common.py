import argparse
import csv
import json
import shutil
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
import torch.optim as optim
from torch_geometric.data import Data

from dataset import CachedGraphDataset
from models.Pretrain import Pretrain_Parallel
from utils.utils import WarmupCosineLR, get_default_device


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def build_parser(default_run_name, default_weight_path, pretrain_mode):
    parser = argparse.ArgumentParser(description=f"Run {default_run_name}.")
    parser.add_argument("--pretrain_epochs", "--epochs", dest="pretrain_epochs", type=int, default=40)
    parser.add_argument("--mask_ratio_image", type=float, default=0.75)
    parser.add_argument("--mask_ratio_graph", type=float, default=0.5)
    parser.add_argument("--num_edge_types", type=int, default=4)
    parser.add_argument("--run_name", default=default_run_name)
    parser.add_argument("--output_weight", default=default_weight_path)
    parser.add_argument("--cache_dir", default="cache/wheat_non_iid")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dry_run", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--limit_train_samples", type=int, default=None)
    parser.add_argument("--limit_valid_samples", type=int, default=None)
    parser.set_defaults(pretrain_mode=pretrain_mode)
    return parser


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_device(name):
    if name == "auto":
        return get_default_device()
    return torch.device(name)


def build_model(pretrain_mode, num_edge_types, device):
    return Pretrain_Parallel(
        img_size=43,
        patch_size=1,
        in_chans=1,
        num_classes=2,
        embed_dim=108,
        num_heads=6,
        depth=1,
        decoder_embed_dim=108,
        decoder_depth=2,
        decoder_num_heads=6,
        norm_pix_loss=False,
        pretrain_mode=pretrain_mode,
        num_edge_types=num_edge_types,
    ).to(device)


def load_cached_split(cache_dir, split):
    cache_path = Path(cache_dir) / f"{split}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"CACHE_SPLIT_NOT_FOUND: {cache_path}")
    return CachedGraphDataset(cache_path, mode=split)


def item_to_data(item, device):
    points, labels, edge_index, trace_id = item
    return Data(
        x=points.clone().detach().to(torch.float32).to(device),
        edge_index=edge_index.clone().detach().to(torch.long).to(device),
        y=labels.clone().detach().to(torch.int64).squeeze().to(device),
    )


def iter_indices(dataset, shuffle):
    if shuffle:
        return torch.randperm(len(dataset)).tolist()
    return list(range(len(dataset)))


def collect_edge_stats(model):
    if hasattr(model, "get_pretrain_edge_statistics"):
        return model.get_pretrain_edge_statistics()
    return {}


def empty_weight_stats(epoch):
    return dict(
        epoch=epoch,
        edge_weight_mean="",
        edge_weight_std="",
        edge_weight_min="",
        edge_weight_q25="",
        edge_weight_q50="",
        edge_weight_q75="",
        edge_weight_max="",
    )


def empty_type_stats(epoch, num_edge_types):
    row = {f"type_{idx}_mean_prob": "" for idx in range(num_edge_types)}
    row.update(
        epoch=epoch,
        type_entropy_mean="",
        dominant_type_ratio="",
        type_collapse_flag="",
    )
    return row


def average_stats(stats_list, key, fields):
    values = [stats[key] for stats in stats_list if key in stats]
    if not values:
        return None
    return {field: sum(item[field] for item in values) / len(values) for field in fields}


def average_type_stats(stats_list, num_edge_types):
    values = [stats["edge_type"] for stats in stats_list if "edge_type" in stats]
    if not values:
        return None
    mean_prob = []
    for idx in range(num_edge_types):
        mean_prob.append(sum(item["mean_prob"][idx] for item in values) / len(values))
    dominant_type_ratio = sum(item["dominant_type_ratio"] for item in values) / len(values)
    return dict(
        mean_prob=mean_prob,
        entropy_mean=sum(item["entropy_mean"] for item in values) / len(values),
        dominant_type_ratio=dominant_type_ratio,
        type_collapse_flag=dominant_type_ratio > 0.95,
    )


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_epoch(model, dataset, args, optimizer, loss_config, device, train):
    model.train(train)
    indices = iter_indices(dataset, shuffle=train)
    limit = args.limit_train_samples if train else args.limit_valid_samples
    if limit is not None:
        indices = indices[:limit]
    total_loss = 0.0
    total_points = 0
    stats_list = []
    for idx in indices:
        data = item_to_data(dataset[idx], device)
        trajectory_length = int(data.x.shape[0])
        if train:
            loss = model.train_step(
                data,
                mask_ratio_image=args.mask_ratio_image,
                mask_ratio_graph=args.mask_ratio_graph,
                optimizer=optimizer,
                loss_config=loss_config,
            )
        else:
            with torch.no_grad():
                loss = model.valid_step(
                    data,
                    mask_ratio_image=args.mask_ratio_image,
                    mask_ratio_graph=args.mask_ratio_graph,
                    loss_config=loss_config,
                )
        total_loss += loss * trajectory_length
        total_points += trajectory_length
        stats_list.append(collect_edge_stats(model))
    return total_loss / max(total_points, 1), stats_list


def run_dry_run(model, train_dataset, args, run_dir, device):
    data = item_to_data(train_dataset[0], device)
    model.eval()
    with torch.no_grad():
        loss = model.forward(
            data,
            mask_ratio_image=args.mask_ratio_image,
            mask_ratio_graph=args.mask_ratio_graph,
            loss_config=None,
        )
    summary = dict(
        dry_run=True,
        pretrain_mode=args.pretrain_mode,
        loss=float(loss.item()),
        num_nodes=int(data.x.shape[0]),
        num_edges=int(data.edge_index.shape[1]),
        edge_statistics=collect_edge_stats(model),
    )
    write_json(run_dir / "dry_run_summary.json", summary)
    print(f"DRY_RUN_COMPLETE run_dir={run_dir} loss={summary['loss']:.6f}")


def run_pretrain(default_run_name, default_weight_path, pretrain_mode):
    parser = build_parser(default_run_name, default_weight_path, pretrain_mode)
    args = parser.parse_args()
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    weight_path = Path(args.output_weight)
    config = vars(args).copy()
    config.update(
        run_dir=str(run_dir),
        output_weight=str(weight_path),
        command=" ".join(sys.argv),
        runtime_num_threads=get_runtime_thread_count(),
    )
    write_json(run_dir / "config_resolved.json", config)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    device = resolve_device(args.device)
    train_dataset = load_cached_split(args.cache_dir, "train")
    valid_dataset = load_cached_split(args.cache_dir, "valid")
    model = build_model(args.pretrain_mode, args.num_edge_types, device)

    if args.dry_run:
        run_dry_run(model, train_dataset, args, run_dir, device)
        return

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0005)
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_start_lr=0.0001,
        end_lr=0.00005,
        warmup_epochs=5,
        total_epochs=args.pretrain_epochs,
    )
    loss_config = dict(orthogonal=dict(reg=1e-5))
    train_log_rows = []
    valid_curve_rows = []
    edge_weight_rows = []
    edge_type_rows = []
    best_valid_loss = float("inf")

    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss, train_stats = run_epoch(model, train_dataset, args, optimizer, loss_config, device, train=True)
        scheduler.step()
        valid_loss, valid_stats = run_epoch(model, valid_dataset, args, optimizer, None, device, train=False)
        train_log_rows.append(dict(epoch=epoch, train_loss=train_loss, valid_loss=valid_loss, lr=optimizer.param_groups[0]["lr"]))
        valid_curve_rows.append(dict(epoch=epoch, valid_loss=valid_loss))

        stats = valid_stats if valid_stats else train_stats
        weight_stats = average_stats(
            stats,
            "edge_weight",
            ["mean", "std", "min", "q25", "q50", "q75", "max"],
        )
        if weight_stats is None:
            edge_weight_rows.append(empty_weight_stats(epoch))
        else:
            edge_weight_rows.append(
                dict(
                    epoch=epoch,
                    edge_weight_mean=weight_stats["mean"],
                    edge_weight_std=weight_stats["std"],
                    edge_weight_min=weight_stats["min"],
                    edge_weight_q25=weight_stats["q25"],
                    edge_weight_q50=weight_stats["q50"],
                    edge_weight_q75=weight_stats["q75"],
                    edge_weight_max=weight_stats["max"],
                )
            )

        type_stats = average_type_stats(stats, args.num_edge_types)
        if type_stats is None:
            edge_type_rows.append(empty_type_stats(epoch, args.num_edge_types))
        else:
            row = {f"type_{idx}_mean_prob": type_stats["mean_prob"][idx] for idx in range(args.num_edge_types)}
            row.update(
                epoch=epoch,
                type_entropy_mean=type_stats["entropy_mean"],
                dominant_type_ratio=type_stats["dominant_type_ratio"],
                type_collapse_flag=type_stats["type_collapse_flag"],
            )
            edge_type_rows.append(row)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), run_dir / "best_pretrain_model.pt")
            weight_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / "best_pretrain_model.pt", weight_path)

        print(f"epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")

    write_csv(run_dir / "train_log.csv", ["epoch", "train_loss", "valid_loss", "lr"], train_log_rows)
    write_csv(run_dir / "valid_loss_curve.csv", ["epoch", "valid_loss"], valid_curve_rows)
    write_csv(
        run_dir / "edge_weight_statistics.csv",
        ["epoch", "edge_weight_mean", "edge_weight_std", "edge_weight_min", "edge_weight_q25", "edge_weight_q50", "edge_weight_q75", "edge_weight_max"],
        edge_weight_rows,
    )
    type_fields = ["epoch"] + [f"type_{idx}_mean_prob" for idx in range(args.num_edge_types)] + [
        "type_entropy_mean",
        "dominant_type_ratio",
        "type_collapse_flag",
    ]
    write_csv(run_dir / "edge_type_statistics.csv", type_fields, edge_type_rows)
    write_json(
        run_dir / "pretrain_summary.json",
        dict(
            pretrain_mode=args.pretrain_mode,
            best_valid_loss=best_valid_loss,
            best_pretrain_model=str(run_dir / "best_pretrain_model.pt"),
            reusable_weight=str(weight_path),
            epochs=args.pretrain_epochs,
        ),
    )
