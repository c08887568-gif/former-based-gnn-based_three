# PTG fine-tune 40ep 实验报告

1. 完成情况：PT1G, PT2G, PT3G。

2. 三组 valid 指标：
- PT1G: valid_accuracy=0.8571149700011451, valid_macro_f1=0.7701736422338402, valid_road_f1=0.6288180610889774, valid_field_f1=0.911529223378703, pred_road_rate=0.15190653755939135, notes=completed
- PT2G: valid_accuracy=0.8809241349661838, valid_macro_f1=0.8193466190207233, valid_road_f1=0.7138753906461694, valid_field_f1=0.9248178473952771, pred_road_rate=0.18312834546220003, notes=completed
- PT3G: valid_accuracy=0.8787965739633521, valid_macro_f1=0.8211302262454413, valid_road_f1=0.7195686206296747, valid_field_f1=0.9226918318612078, pred_road_rate=0.1991640103446202, notes=completed

3. macro-F1 最好：PT3G。
4. road-F1 最好：PT3G。
5. field-F1 最好：PT2G。
6. 是否建议保留补充边方案：建议优先保留 PT3G 作为下一步候选；是否全面保留，需要与未加补充边的 PT1/PT2/PT3 valid/test 指标继续对比。
