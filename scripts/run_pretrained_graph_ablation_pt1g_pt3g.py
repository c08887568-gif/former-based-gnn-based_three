import argparse


EXPERIMENTS = [
    dict(
        name="PT1G",
        title="PT1G current pretrain + pretrained graph edges",
        pretrained_path="weights/PT1_current_masked_pretrain.pt",
        pretrain_mode="current",
        run_name="PT1G_finetune_40ep",
    ),
    dict(
        name="PT2G",
        title="PT2G edge weight pretrain + pretrained graph edges",
        pretrained_path="weights/PT2_edge_weight_pretrain.pt",
        pretrain_mode="edge_weight",
        run_name="PT2G_finetune_40ep",
    ),
    dict(
        name="PT3G",
        title="PT3G edge type + weight pretrain + pretrained graph edges",
        pretrained_path="weights/PT3_edge_type_weight_pretrain.pt",
        pretrain_mode="edge_type_weight",
        run_name="PT3G_finetune_40ep",
    ),
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def shell_join(lines):
    return " \\\n  ".join(lines)


def export_command(exp, topk):
    cache_path = f"cache/pretrained_graphs/{exp['name']}_topk{topk}"
    return shell_join(
        [
            "python scripts/export_pretrained_graph_edges.py",
            f"--pretrained_path {exp['pretrained_path']}",
            f"--pretrain_mode {exp['pretrain_mode']}",
            f"--output_cache {cache_path}",
            f"--topk {topk}",
        ]
    )


def finetune_command(exp, topk, epochs, skip_test):
    cache_path = f"cache/pretrained_graphs/{exp['name']}_topk{topk}"
    return shell_join(
        [
            "python fine_tune.py",
            "--use_pretrain true",
            f"--pretrained_path {exp['pretrained_path']}",
            f"--pretrain_mode {exp['pretrain_mode']}",
            f"--graph_cache_path {cache_path}",
            f"--run_name {exp['run_name']}",
            f"--epochs {epochs}",
            f"--skip_test {str(skip_test).lower()}",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Print PT1G/PT2G/PT3G command lines.")
    parser.add_argument("--mode", choices=["commands_only", "run_all"], default="commands_only")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--skip_test", type=str2bool, nargs="?", const=True, default=True)
    args = parser.parse_args()

    if args.mode == "run_all":
        raise SystemExit("RUN_ALL_NOT_ENABLED: 本脚本默认只生成命令，不自动正式训练。")

    for exp in EXPERIMENTS:
        print(f"# {exp['title']}")
        print(export_command(exp, args.topk))
        print()
        print(finetune_command(exp, args.topk, args.epochs, args.skip_test))
        print()


if __name__ == "__main__":
    main()
