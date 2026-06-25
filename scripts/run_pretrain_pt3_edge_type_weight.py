from pretrain_ablation_common import run_pretrain


if __name__ == "__main__":
    run_pretrain(
        default_run_name="PT3_edge_type_weight_pretrain",
        default_weight_path="weights/PT3_edge_type_weight_pretrain.pt",
        pretrain_mode="edge_type_weight",
    )
