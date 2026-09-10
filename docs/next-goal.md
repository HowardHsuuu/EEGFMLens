# 下一個可驗收里程碑

EEGLens 的整體目標仍是可信、可重現、可供外部研究者使用的 EEG FM 干預工具。下一個里程碑聚焦於：**區分破壞一個 probe 的讀取方向，與降低模型表徵中可恢復的生理資訊，並檢驗後者與任務依賴的關係。**

## 為什麼追這個問題

現有三模型研究尚未建立共同機制。training-subject cross-fitting 顯示，移除原 μ／β probe 係數張成的 rank-two 空間後，LaBraM 的 β pooled R² 從 clean 0.511 降至重新擬合後的 0.453；但 recovered 個別受試者 R² 只有 8／18 大於零。這是值得追的探索性線索，並非普遍的冗餘編碼或跨受試者泛化證明。

因此，不能把固定 probe 失效直接解釋成資訊消失，也不能把 pooled 分數直接解釋成每個人都可用。來源與限制見 [recoverability report](../research/mi/RECOVERABILITY.md)。

## 科學交付

問題：在 CBraMod、LaBraM、CSBrain 中，逐步移除可線性讀出的 μ／β 資訊時，剩餘資訊、非目標資訊與任務表現如何共同變化？

1. 在新的分析 protocol 中事先固定 subject-disjoint folds、移除 rank、停止規則、readout 類別與比較方式。既有 test 已看過，後續利用它的結果只能標為探索性；確認性結論需要另定未使用的評估資料。
2. 先檢查各模型概念的受試者層級可讀性。報告個別受試者與合併分數，保留不可讀出的模型作為失敗案例，避免只挑成功模型。
3. 比較固定 probe 失效與重新擬合後的恢復曲線；迭代移除方向只用 fitting subjects 學習，驗證／評估 subjects 不參與方向建構。即使線性 recovery 消失，也只能聲稱已測線性 readout 下不可恢復。
4. 若要解釋下游依賴，必須在模型內實際施加干預並重新 forward。離線特徵投影結果不能代替此步。分別加入 rank-matched 與 perturbation-energy-matched 隨機對照，記錄干預強度、非目標 readout 與任務變化。

交付物是三模型的 subject-wise recovery 曲線、任務／非目標副作用曲線，以及有明確限制的判讀：讀法被破壞、仍有可恢復資訊、廣泛表徵損傷，或資料不足以區分。成功不要求三模型具有共同機制。

## 工具交付

把上述實驗整理成可重現範例，保留資料切分、checkpoint／source hashes、干預位置與 native 軸定義。分析 fitting 與 evaluation 的角色應明確分開，不能讓工具默默以評估 labels 建立干預方向。

十一模型的 integration 以逐模型 capability 與驗證範圍交付。目前已有 11 families／13 views 的 native 語義與抽樣輸入邊界紀錄；這不保證所有配置都正確。將範例涉及的操作接入既有 native oracle、cleanup 與 installed-wheel 驗收，對未支援的操作明確拒絕。

Public readiness 必須列明哪些平台、版本與操作已實測。TransformerLens-level quality 是長期品質方向，不能以模型數或通過測試數作為完成標準。

## 停止與決策條件

完成預定分析與可重現範例後，依證據決定是否值得推進跨模型 steering。若概念在未參與擬合的受試者上不穩定，或效果無法與廣泛損傷區分，就交付這個限制與下一個可辨識的問題；不靠持續加 rank、換 split 或挑受試者追求正結果。
