import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Data

from dataset import GraphDataset
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
    return parser.parse_args()


def auto_run_name():
    return datetime.now().strftime("fine_tune_%Y%m%d_%H%M%S")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_config(args, run_name, run_dir):
    pretrained_path = normalize_optional_path(args.pretrained_path)
    effective_pretrained_path = pretrained_path if args.use_pretrain else None
    return dict(
        use_pretrain=args.use_pretrain,
        pretrained_path=pretrained_path,
        effective_pretrained_path=effective_pretrained_path,
        run_name=run_name,
        run_dir=str(run_dir),
        dry_run=args.dry_run,
        skip_test=args.skip_test,
        total_epochs=TOTAL_EPOCHS,
    )


def write_run_metadata(run_dir, config):
    write_json(run_dir / "config_resolved.json", config)
    command = " ".join(sys.argv)
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")


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
    ).to(device)
    return model


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
        test_path = dict(
            gnss="../dataset_5/dataset_high/paddy/sampled_paddy_43",
            adj="../dataset_5/dataset_high/paddy/sampled_paddy_adj",
            json="../dataset_5/dataset_high/paddy/Non-Identically_Distributed_Coco/sampled_paddy_43_test.json"
        )
        test_dataset = GraphDataset(test_path, mode='test', num_workers=2, max_len=1000, drop_rate=0)
    logger.log_dataset_info(train_dataset, valid_dataset, test_dataset)
    # Create data loaders using PyTorch DataLoader
    train_loader = FieldRoadDataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    valid_loader = FieldRoadDataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=True)
    if test_dataset is not None:
        test_loader = FieldRoadDataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    ####################################超参数
    #torch.autograd.set_detect_anomaly(True)
    total_epochs = TOTAL_EPOCHS
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
    best_valid_acc = 0.0
    total_train_time = 0
    total_valid_time = 0
    total_test_time = 0
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    class_result_train = None
    class_result_valid = None
    class_result_test_train = None
    class_result_test_valid = None
    final_test_train = False
    final_test_valid = False
    for pass_num in range(total_epochs):
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
            edge_index = to_edge_index(adjs, device)
            data = Data(x=points, edge_index=edge_index, y=labels)
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
            final_test_train = True
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
                edge_index = to_edge_index(adjs, device)
                data = Data(x=points, edge_index=edge_index, y=labels)
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
            if avg_valid_acc > best_valid_acc:
                best_valid_acc = avg_valid_acc
                all_predictions = np.array(all_predictions)
                all_labels = np.array(all_labels)
                class_result_valid = model.calculate_classification_metrics(all_predictions, all_labels)
                torch.save(model.state_dict(), './weights/model.pt')
                final_test_valid = True
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
            if not config["skip_test"]:
                all_predictions = []
                all_labels = []
                test_num_samples = 0
                for batch_id, (points, labels, adjs, trace_id) in enumerate(test_loader()):
                    points = points.clone().detach().to(torch.float32).squeeze(0).to(device)
                    labels = labels.clone().detach().to(torch.int64).squeeze().to(device)
                    edge_index = to_edge_index(adjs, device)
                    data = Data(x=points, edge_index=edge_index, y=labels)
                    pred = model.test_step(data)
                    all_predictions.extend(pred.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    test_num_samples += points.shape[0]
                all_predictions = np.array(all_predictions)
                all_labels = np.array(all_labels)
                class_result = model.calculate_classification_metrics(all_predictions, all_labels)
                end_time = time.time()
                epoch_test_time = end_time - valid_end_time
                total_test_time += epoch_test_time
                test_fps = test_num_samples / epoch_test_time
                logger.log_test_info(epoch_test_time, test_fps, class_result)
                if final_test_train == True:
                    class_result_test_train = class_result
                    final_test_train = False
                if final_test_valid == True:
                    class_result_test_valid = class_result
                    final_test_valid = False
    # Calculate average training and evaluation times
    avg_train_time = total_train_time / total_epochs
    avg_valid_time = total_valid_time / total_epochs
    avg_test_time = total_test_time / total_epochs
    logger.log_end_of_training(avg_train_time, avg_valid_time, avg_test_time, class_result_train, class_result_valid,
                               class_result_test_train, class_result_test_valid)
    logger.log_start_of_outputs()
    # 训练完成后，绘制损失曲线
    logger.plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, is_save=True)
    logger.clean_up_logger()


if __name__ == "__main__":
    main()
