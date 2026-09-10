# 保留共同成分的平均差異干預

逐電極投影與只改平均 C4−C3 可以得到相同的離線 contrast，卻不是同一個原生操作。為了區分這兩者，新增 research helper `contrast_intervention.py`，透過 EEGLens 的 `Replacement` 注入 donor；尚未加入公開套件 API。

令每個 trial 的 `d = mean_patch(C4) − mean_patch(C3)`，training-only 中心為 `m`，欲移除的成分為 `v = (d − m) Q Qᵀ`。對所有 patch 使用相同修正：

- C3 加 `v/2`。
- C4 減 `v/2`。
- 其他電極及 CLS 不變。

因此平均差異變為 `d−v`，每個 patch 的 C3+C4 不變，各電極相對於其時間平均值的偏差也不變。若有 P 個等權 patch，平方擾動量為 `P/2 × ||v||²`。由平方和在固定總和下於均勻分配時最小，並將總差異均分至兩電極，可知這是達成指定平均 contrast 改變的最小 Frobenius-norm activation 修正。這不是原始 EEG 能量、最小生理影響或最小下游輸出影響的保證。

## 驗證

`test_contrast_intervention.py` 在 float64 的 bcpd 與 CLS token 兩種布局，檢查目標 contrast、共同成分、時間偏差、untouched values、理論平方擾動量，並比較另一個符合 contrast 約束但擾動較大的候選。兩個測試通過。

`validate_contrast_native.py` 使用 CBraMod、LaBraM、CSBrain 官方 checkpoint，outer fold 0 中兩個真實 EEG trials（原 training partition）與五個固定 ranks：2/4/8/16/32。十五個模型／rank 條件全部通過：

- EEGLens 下游 output 與獨立 native hook 零誤差一致。
- 未選電極的完整 activation，以及 LaBraM CLS，完全不變。
- 共同成分與各 patch 相對平均值的偏差，在 float32 atol/rtol=5e−5 內保留。
- 與離線 centered contrast 的最大絕對差為 2.13e−7；此容許不適用於 native output，它仍要求零誤差。
- 清除暫時 hooks 後，clean output 完全恢復。

結果在 `results/contrast-native-v2/`。這些是特定 trial／配置的實作驗證，不是全受試者機制發現。

## 接下來的科學比較

固定相同 Q，可以比較逐電極投影與這個最小擾動操作：兩者對平均 contrast 的目標相同，但對共同與時間成分的副作用不同。再加入 rank 與實際 activation delta norm 的隨機對照，才有機會區分任務損傷來自 contrast、本來連帶改動的成分，或一般擾動。

即使這個操作在座標上精確，也不自動讓 probe direction 成為生理概念。方向的跨受試者可讀性、離線線性 recovery 的限制與資料切分仍沿用 `ITERATIVE_RECOVERY.md` 的限制。這個新操作的三模型全受試者下游任務實驗已完成；結果與限制見 [task report](CONTRAST_TASK_RESULTS.md)。

重現原生檢查（parent workspace，既有研究環境與資料／checkpoint 布局）：

```bash
PYTHONPATH=eeglens/src python eeglens/research/mi/validate_contrast_native.py \
  --model labram --root research \
  --output research/eeglens_mi/contrast-native-reproduced/labram.json
```

## 全受試者任務分析（已完成）

`contrast_task_protocol.json` 固定兩種方法、五個 ranks 與三個 random seeds。每個方法／rank 比較 concept、三個同 rank random、三個逐 trial activation-norm-matched random，加上 clean 共 71 個條件。各折用 fitting 的九人擬合 clean final-layer task／descriptor readouts，以三位 validation subjects 選正則化，六位 evaluation subjects 接受模型內干預。各 trial 的 matched gain 保留不裁切；零 random norm 會拒絕而非產生假對照。

`contrast_task.py` 每折獨立執行，輸出 responses、實際 norm/gain、readout 係數與 provenance。會先比對 clean final features 與既有 feature artifact，並檢查逐 trial matched norm。`summarize_contrast_task.py` 只有在三折、18 人、810 個不重複 evaluation trials 齊全時，才產生各受試者結果與 concept 減去 matched-random 的描述性平均。交叉擬合的 training sets 重疊，因此不把 folds 當成獨立確認性樣本。

執行範例（parent workspace）：

```bash
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTHONPATH=eeglens/src \
python eeglens/research/mi/contrast_task.py --model cbramod --fold 0 \
  --output research/eeglens_mi/contrast-task-v1/cbramod/fold0
```

對其餘 folds 及模型分別執行，完成後使用 `summarize_contrast_task.py --results research/eeglens_mi/contrast-task-v1/cbramod --output research/eeglens_mi/contrast-task-v1/cbramod/summary.json`。三模型皆已完成三折、18 人、810 trials 與 71 conditions；完整結果見 [task report](CONTRAST_TASK_RESULTS.md)。

早期執行紀錄：CBraMod fold 0 已完成 270 trials × 71 conditions；8,100 個 matched norm 核對通過，所有 prediction/gain 有限，labels/descriptors 與 prepared data 完全一致。最大 matching gain 為 18.97，保留作為解釋限制。該時間點其餘 folds 尚在執行；現在皆已完成。`test_contrast_summary.py` 以已知 task/loss 效果驗證配對摘要，並確認即使重新計算 artifact hash，篡改 matched norm 仍會被拒絕且不產生成功報告。

彙整亦要求三折的 model、checkpoint、features、bases、runner/helper/fitter/builder hashes 一致，以及互斥的 9/3/6 人角色。已知答案測試另涵蓋混用模型／程式版本與 fitting/validation 重疊的拒絕案例。
