import argparse


COMMANDS = [
    (
        "PT0 fine-tune",
        "python fine_tune.py --use_pretrain false --run_name PT0_no_pretrain_seed42 --skip_test true",
    ),
    (
        "PT1 pretrain",
        "python scripts/run_pretrain_pt1_current.py",
    ),
    (
        "PT1 fine-tune",
        "python fine_tune.py --use_pretrain true --pretrained_path weights/PT1_current_masked_pretrain.pt --run_name PT1_current_masked_pretrain_finetune_seed42 --skip_test true",
    ),
    (
        "PT2 pretrain",
        "python scripts/run_pretrain_pt2_edge_weight.py",
    ),
    (
        "PT2 fine-tune",
        "python fine_tune.py --use_pretrain true --pretrained_path weights/PT2_edge_weight_pretrain.pt --pretrain_mode edge_weight --run_name PT2_edge_weight_finetune_seed42 --skip_test true",
    ),
    (
        "PT3 pretrain",
        "python scripts/run_pretrain_pt3_edge_type_weight.py",
    ),
    (
        "PT3 fine-tune",
        "python fine_tune.py --use_pretrain true --pretrained_path weights/PT3_edge_type_weight_pretrain.pt --pretrain_mode edge_type_weight --run_name PT3_edge_type_weight_finetune_seed42 --skip_test true",
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Print or run PT0-PT3 pretrain ablation commands.")
    parser.add_argument("--mode", choices=["commands_only", "run_all"], default="commands_only")
    args = parser.parse_args()

    if args.mode == "run_all":
        raise SystemExit("RUN_ALL_NOT_ENABLED: 本阶段只生成命令，不自动正式训练。")

    for title, command in COMMANDS:
        print(f"# {title}")
        print(command)
        print()


if __name__ == "__main__":
    main()
