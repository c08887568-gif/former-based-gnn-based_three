# PT2G_MSC_aux_v1 Report

## 对比范围

- PT2G_finetune_40ep
- PT2G_MSC_v1_finetune_40ep
- PT2G_MSC_SD_v1_finetune_40ep
- PT2G_MSC_RC_v1_finetune_40ep
- PT2G_MSC_RCSD_v1_finetune_40ep

## 核心结论

- valid macro-F1 最好：PT2G_MSC_RC_v1_finetune_40ep。
- test macro-F1 最好：PT2G_MSC_RC_v1_finetune_40ep。
- road-F1 最好：PT2G_MSC_RC_v1_finetune_40ep。
- field-F1 最好：PT2G_MSC_RC_v1_finetune_40ep。
- SD field_as_road 相对 MSC_v1 变化：187.000000。
- SD long_field_as_road 相对 MSC_v1 变化：-9.000000。
- RC road_as_field 相对 MSC_v1 变化：-328.000000。
- RC long_road_as_field 相对 MSC_v1 变化：-1.000000。
- RC field_as_road 相对 MSC_v1 变化：-271.000000。
- RCSD road_as_field 相对 MSC_v1 变化：-291.000000。
- RCSD field_as_road 相对 MSC_v1 变化：1138.000000。

## 分类坍塌检查

- PT2G_finetune_40ep: pred_road_rate=0.20224200669036607, pred_field_rate=0.7977579933096339, collapse_flag=False。
- PT2G_MSC_v1_finetune_40ep: pred_road_rate=0.2151193884292321, pred_field_rate=0.7848806115707679, collapse_flag=False。
- PT2G_MSC_SD_v1_finetune_40ep: pred_road_rate=0.2152976583718711, pred_field_rate=0.7847023416281289, collapse_flag=False。
- PT2G_MSC_RC_v1_finetune_40ep: pred_road_rate=0.2157171170604335, pred_field_rate=0.7842828829395665, collapse_flag=False。
- PT2G_MSC_RCSD_v1_finetune_40ep: pred_road_rate=0.2301045500781242, pred_field_rate=0.7698954499218759, collapse_flag=False。

## 保留建议

- 若 SD 同时降低 field_as_road 和 long_field_as_road，且 valid macro-F1 不低于 MSC_v1，优先保留 SD。
- 若 RC 降低 road_as_field 和 long_road_as_field，同时 field_as_road 不明显增加，优先保留 RC。
- 若 RCSD 的 test macro-F1 不低于 MSC_v1，且两类错误至少一类下降、另一类不恶化，优先进入下一阶段。
- 若三组指标缺失，说明训练或错误分析尚未完成，本报告只作为汇总模板。
