from pretrain_ablation_common import run_pretrain


if __name__ == "__main__":
    run_pretrain(
        default_run_name="PT1_current_masked_pretrain_pretrain",
        default_weight_path="weights/PT1_current_masked_pretrain.pt",
        pretrain_mode="current",
    )
