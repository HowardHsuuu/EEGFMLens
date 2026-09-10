# EEGLens：生理資訊可讀出，是否代表任務依賴？

本輪三模型實驗已完成；這是範圍有限的研究結果，不是 EEGLens 全面 public-ready 的宣告。

## 問題與設計

我們測試 CBraMod、LaBraM、CSBrain 的中間層能否線性讀出左右運動皮質 μ／β 功率差異，以及移除相關方向後，下游任務與生理讀出是否受損。

EEGMMIDB 前 30 位受試者的左右手運動想像 runs 04、08、12，依事先固定的受試者切分形成 train / validation / test：18 / 6 / 6 人，810 / 270 / 270 trials。所有模型使用相同 19 電極、四秒訊號、160→200 Hz 重採樣與幅度尺度。這是共同實驗輸入配方，並非各模型官方前處理的完整重現。

中間層為第六個 transformer block（索引 5），最終 encoder 特徵取索引 11；只平均時間，保留電極。線性 ridge probe 只在 training 擬合、validation 選擇正則化。由 C4−C3 中間特徵預測 μ／β log-power asymmetry 的兩個係數方向形成 rank-two 子空間；在 C3、C4 全部時間片移除其中心化投影。下游讀出在干預後保持固定。

對照包含三個 rank-two 隨機子空間、兩個劑量，以及逐 trial 匹配概念干預 L2 強度的隨機控制。主要比較為全劑量概念干預減去三個等強度隨機控制的平均。另追蹤枕葉 alpha 與全域 RMS 的讀出損失；它們只涵蓋兩項非目標資訊，不代表所有非目標功能。

## 測試集結果

Task BA 是六位受試者 balanced accuracy 的平均；R² 是所有測試 trials 合併計算的中間層概念讀出，兩者聚合方式不同。

| 模型 | 預訓練 task BA | 隨機初始化 task BA | 預訓練 μ R² | 預訓練 β R² |
| --- | ---: | ---: | ---: | ---: |
| CBraMod | 0.5601 | 0.6111 | 0.1253 | −0.0088 |
| LaBraM | 0.5842 | 0.5831 | 0.2780 | 0.3753 |
| CSBrain | 0.6684 | 0.6339 | 0.0502 | −0.0566 |

95 維頻譜特徵基線 task BA 為 **0.5569**。隨機初始化使用先前固定的一個 constructor seed（812）；沒有 test fitting 或 seed 搜尋，但不能代表初始化分布。預訓練減去隨機初始化的 subject-bootstrap 描述區間分別為 −0.0510 [−0.1071, −0.0074]、0.0011 [−0.0919, 0.0779]、0.0345 [−0.0177, 0.0878]；不是預先註冊的顯著性檢定。

| 模型 | 概念−等強度隨機：生理 normalized MSE 增量 | 同一比較：task BA 變化 | 生理 / task Holm p |
| --- | ---: | ---: | ---: |
| CBraMod | −0.00491 [−0.06540, 0.04531] | −0.00246 [−0.01153, 0.00529] | 0.9375 / 0.9375 |
| LaBraM | 0.01995 [0.00068, 0.04135] | −0.00268 [−0.01120, 0.00789] | 0.46875 / 0.9375 |
| CSBrain | 0.00344 [−0.00193, 0.00880] | −0.01983 [−0.04149, −0.00463] | 0.6250 / 0.3750 |

方括號為 10,000 次 subject-bootstrap 的描述性 95% 區間，沒有多重比較校正；不能用它取代右欄的固定推論。生理損失是 μ／β MSE 除以 training variance 後取平均。非目標損失增量依模型為 0.00154、0.00481、−0.00052，其描述區間皆包含零，並不構成無非目標損害的等效性證據。

固定推論使用六位受試者的 exact one-sided sign flips，再對三模型 × 兩端點做 Holm 校正。六人的最小 raw p 為 1/64，最小可能 Holm p 至少 6/64 = 0.09375：**此設計本身無法在 familywise 0.05 下拒絕任何端點**。保留原始 protocol，誠實報告解析度限制；不能事後改檢定或縮減檢定家族來宣稱顯著。

## 可以得到的洞見

1. **跨受試者的概念讀出可靠性，是機制解釋的前提。** CBraMod 的 validation β R² 為 0.5567，test 卻是 −0.0088；CSBrain 也從 0.2532 降至 −0.0566。負 R² 表示預測誤差高於測試集常數平均值基準。在這些模型上，干預無效果可能反映概念方向沒有泛化，不能直接推論模型不依賴 β。
2. **線性生理讀出與任務能力在這個設置下沒有一致排序。** LaBraM 的 test 生理讀出最強，CSBrain 的 task BA 最高；CSBrain 出現約 1.98 百分點任務損傷線索，但沒有清楚的生理讀出損傷。這尚未證明不同機制，更可能需要檢驗概念方向是否混入其他任務資訊。
3. **還不能說我們解釋了預訓練收益。** CBraMod、LaBraM 沒有顯示高於此單一隨機初始化的 task 分數；CSBrain 的優勢也仍不確定。此研究測的是固定表徵與固定線性讀出的依賴，不能外推為 foundation model 全部能力或端到端微調後的機制。

## 下一個科學實驗的決策

目前不推進跨模型 steering 遷移。先做明確標為 exploratory 的概念方向診斷：在 training 內做 subject-wise cross-fitting，測試方向是否能在未參與擬合的受試者讀出 μ／β；干預後以重新擬合的 probe 檢查資訊是否仍能從其他方向恢復，並比較多層與前處理敏感性。這能區分「固定 probe 被破壞」與「資訊真的減少」，但新的 probe 仍不能證明所有非線性資訊都消失。

CSBrain 的任務變化值得追蹤，但需先確認它與可泛化的生理資訊具有選擇性關聯。LaBraM 可用作較穩定線性概念讀出的對照，須保留 pretraining exposure 限制。任何確認性延伸都要另定較大的獨立受試者樣本與 power 分析；現有 test 不再當成未見資料調參。

## Integration 與重現範圍

三模型所有公開 sites，在兩個真實 development EEG trials、batch 1 / 2、dense rank-three 全域與局部 C4/patch-1 干預下，共 **400 條件**與獨立 native hooks 輸出完全一致：CBraMod 196、LaBraM 148、CSBrain 56。局部干預的未選位置保持完全相等；CSBrain 電極排序與 LaBraM CLS 偏移由原生座標獨立指定。這些是 CPU float32、固定幾何的證據，不能推論其他八模型或所有配置同樣通過。

完整測試集干預的 clean 輸出與另行萃取的 clean feature/readout 結果完全一致，各 trial 隨機控制強度在 float32 容許誤差內匹配。既有十一模型 coordinate-subspace 驗收與本次 dense 驗收互補，並非完整 public-ready 保證。

另有重要限制：LaBraM 論文報告的預訓練包含 EEGMMIDB，因此 downstream subject-disjoint 不代表 pretraining-unseen；CBraMod、CSBrain 報告 TUEG 預訓練，但未獨立驗證個人身分完全無重疊。共同前處理省略官方部分濾波；左右視覺提示與想像標籤混淆；當下功率不等同有 baseline 的 ERD。來源見 [pretraining audit](PRETRAINING_AUDIT.md)。

程式與固定設定：`split-v1.json`、`test-protocol-v1.json`、`evaluate_probes.py`、`evaluate_spectral.py`、`summarize.py`、`infer_test.py`；dense 原生對照為 `../model_validation/validate_dense_mi.py`，結果在 `../model_validation/results/dense-mi-v1/`。工作區資料根目錄 `research/eeglens_mi/` 下保留 `test-responses-v1/{model}/summary.json`、`test-responses-v1/inference.json`、`{random-,}probes-v1/{model}/test-evaluation.json` 與 `spectral-v1/test-evaluation.json`。原始 EDF 與權重不納入套件。
