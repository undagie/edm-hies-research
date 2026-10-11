# Paper Draft

<!-- AUTO_RESULTS_START -->

## Results (Auto-Generated)

### Best method per dataset (macro-F1; ties broken by joint_score, then MCC, then PR-AUC; dummy excluded)

            dataset method  f1_macro_mean  mcc_mean  joint_score_mean  dp_diff_mean
uci_dropout_success   hies         0.8552    0.7138            1.1027        0.1922
        uci_student   hies         0.9098    0.8227            1.2072        0.1053
               xapi   hies         0.9157    0.8330            1.2138        0.1818

### uci_dropout_success

                      method  f1_macro_mean  f1_macro_std  f1_macro_ci_low  f1_macro_ci_high  mcc_mean  dp_diff_mean  eod_diff_mean
                        hies         0.8552        0.0121           0.8523            0.8582    0.7138        0.1922         0.0504
                      voting         0.8538        0.0117           0.8510            0.8565    0.7086        0.2004         0.0492
                      tabpfn         0.8529        0.0131           0.8499            0.8561    0.7116        0.1809         0.0551
                     xgboost         0.8522        0.0130           0.8492            0.8555    0.7096        0.1839         0.0517
                    catboost         0.8513        0.0119           0.8485            0.8542    0.7032        0.2056         0.0537
                    lightgbm         0.8491        0.0121           0.8462            0.8520    0.6986        0.1995         0.0497
               random_forest         0.8478        0.0120           0.8451            0.8507    0.7050        0.1834         0.0620
                    stacking         0.8468        0.0130           0.8434            0.8499    0.6942        0.2167         0.0507
         logistic_regression         0.8457        0.0111           0.8431            0.8484    0.6925        0.2215         0.0768
                 extra_trees         0.8455        0.0126           0.8425            0.8485    0.7029        0.1688         0.0550
fairlearn_equalized_odds_xgb         0.8370        0.0170           0.8329            0.8412    0.6771        0.1465         0.0422

### uci_student

                      method  f1_macro_mean  f1_macro_std  f1_macro_ci_low  f1_macro_ci_high  mcc_mean  dp_diff_mean  eod_diff_mean
                        hies         0.9098        0.0310           0.9020            0.9176    0.8227        0.1053         0.0830
                     xgboost         0.9088        0.0305           0.9014            0.9166    0.8203        0.1047         0.0885
                      tabpfn         0.9084        0.0303           0.9009            0.9162    0.8197        0.1034         0.0905
fairlearn_equalized_odds_xgb         0.9076        0.0299           0.9002            0.9147    0.8179        0.1076         0.0914
         logistic_regression         0.9072        0.0337           0.8985            0.9160    0.8187        0.1378         0.0862
                      voting         0.9070        0.0320           0.8994            0.9154    0.8167        0.1030         0.0849
                    stacking         0.9066        0.0310           0.8991            0.9140    0.8166        0.1121         0.0799
                    catboost         0.9057        0.0311           0.8978            0.9136    0.8139        0.1037         0.0761
                    lightgbm         0.9030        0.0285           0.8960            0.9100    0.8085        0.1093         0.1048
               random_forest         0.8925        0.0346           0.8837            0.9009    0.7875        0.0965         0.0883
                 extra_trees         0.8059        0.0488           0.7936            0.8190    0.6338        0.0961         0.1380

### xapi

                      method  f1_macro_mean  f1_macro_std  f1_macro_ci_low  f1_macro_ci_high  mcc_mean  dp_diff_mean  eod_diff_mean
                        hies         0.9157        0.0330           0.9077            0.9244    0.8330        0.1818         0.1539
                    catboost         0.9137        0.0299           0.9065            0.9216    0.8297        0.1895         0.1431
                      voting         0.9103        0.0327           0.9024            0.9187    0.8227        0.1878         0.1492
               random_forest         0.9071        0.0352           0.8989            0.9160    0.8166        0.1548         0.1675
                 extra_trees         0.9068        0.0355           0.8975            0.9154    0.8157        0.1703         0.1516
                    stacking         0.9067        0.0328           0.8991            0.9148    0.8164        0.1931         0.1418
                     xgboost         0.9065        0.0332           0.8983            0.9153    0.8151        0.1854         0.1511
                      tabpfn         0.9041        0.0368           0.8950            0.9135    0.8099        0.1724         0.1555
fairlearn_equalized_odds_xgb         0.9031        0.0358           0.8942            0.9117    0.8101        0.1846         0.1669
                    lightgbm         0.9013        0.0334           0.8930            0.9098    0.8051        0.1884         0.1493
         logistic_regression         0.8949        0.0338           0.8861            0.9034    0.7958        0.1929         0.1106

<!-- AUTO_RESULTS_END -->
