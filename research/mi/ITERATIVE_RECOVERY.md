# 逐步移除後的線性資訊恢復

這是觀察主研究與 rank-two recovery 後定義的探索性分析。只使用原 training 的 18 位受試者；三個 outer folds 每次 fitting/validation/evaluation 分別為 9/3/6 人。每個人恰好接受一次 outer 評估。

固定 rank 為 0、2、4、8、16、32。每次用 fitting subjects 的 μ／β 建立兩個新方向，再投影到既有移除空間的正交補空間，累積正交 basis。方向 ridge alpha 固定為 1；重新擬合的四種 descriptor readout 才用 inner validation 選 alpha。這與前一分析以 validation 選概念 probe alpha 的設計不同，因此 rank-two 結果不必相同，不能當成前次結果的直接重現。

## 結果

下表為 pooled out-of-fold β R²。每個 rank 都重新擬合 readout，未使用 outer 評估結果選 rank。

| 模型 | clean | concept rank 2 | rank 4 | rank 8 | rank 16 | rank 32 | random rank 32 範圍 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CBraMod | 0.463 | 0.082 | −0.313 | −0.282 | −0.001 | 0.027 | 0.321–0.458 |
| LaBraM | 0.511 | 0.446 | 0.357 | 0.073 | 0.002 | −0.023 | 0.505–0.512 |
| CSBrain | 0.145 | 0.108 | 0.122 | −0.018 | −0.037 | −0.040 | 0.110–0.166 |

LaBraM 在 rank 8 的 random 範圍為 0.505–0.516，concept 為 0.073。這支持目前方法對線性 readout 的影響與任意移除相同維度不同；它尚不能區分擾動能量差異、目標相關資訊與廣泛表示損傷。

受試者層級也必須保留：concept rank 8 後 β R² > 0 的人數，CBraMod／LaBraM／CSBrain 分別為 1／18、2／18、1／18；rank 32 分別為 1／18、0／18、1／18。不能把 pooled 正分數當成對每位受試者都可用。

## 強度與非目標限制

Concept rank 32 移除的 centered contrast 能量占比，三個 folds 範圍為 CBraMod 28.9–55.0%、LaBraM 25.8–39.8%、CSBrain 19.2–28.1%。這已經是大幅改動。記錄的是 C4−C3 特徵空間能量，不是原始 EEG 能量或模型內完整 activation 的擾動大小。

CBraMod 的 global RMS pooled R² 從 clean 0.393 降至 rank 32 的 0.132；CSBrain 從 0.400 降至 0.166，顯示非目標資訊也受影響。LaBraM 的非目標 readout 在 clean 就沒有正 R²，不能把它們的變化用來證明選擇性。

CBraMod 曲線不單調：rank 4 的 β R² 比 rank 32 更低。嵌套移除空間本身不會增加資訊，但有限樣本的正則化、重新擬合與泛化誤差可以讓 measured R² 非單調。不能把這種回升描述為模型重新創造了資訊。

## 可以決定什麼

前次 rank-two projection 留下可恢復資訊，這次以更多 fit-only 方向可進一步降低已測線性 readout 表現；因此「固定 probe 失效」與「剩餘資訊不可被新 probe 讀出」必須分開量測。但更高 rank 不是免費的精確干預：它同時改變大量能量與非目標資訊。

後續若要推論任務依賴，仍需在原生模型內施加相同方向、加入能量匹配控制，並重新 forward。這次僅對既有 middle features 做離線投影，不是下游模型干預，不證明共同機制、完整資訊移除或非線性不可讀性。所有 rank 都應完整呈現；不以這次曲線挑 rank 後再把同批受試者當成確認性 test。

## 重現

從 parent workspace 執行，NumPy 即可，不需載入 torch 或 checkpoint：

```bash
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python eeglens/research/mi/iterative_recovery.py \
  --features research/eeglens_mi/features-v1/train/labram.npz \
  --split eeglens/research/mi/split-v1.json \
  --protocol eeglens/research/mi/iterative_recovery_protocol.json \
  --output research/eeglens_mi/iterative-recovery-reproduced/labram
```

另以 cbramod、csbrain 的 training features 執行。輸出包括全部 condition predictions、每個 fold 的完整 basis、能量與四個目標的 pooled／individual R²，並記錄輸入／設定／程式雜湊；既有輸出目錄被拒絕。

`test_iterative_recovery.py` 大幅修改一個 outer fold 的 evaluation labels，驗證該 fold 所有條件的 predictions 與 concept basis 完全不變，並驗證 9/3/6 人互斥、18 人各評估一次。測試通過。這是特定 leakage 防線，不能取代研究設計審查。

## 離線 contrast 到原生 activation 的對應驗證

`validate_iterative_native.py` 使用三個官方 checkpoint、outer fold 0 的兩個 evaluation trials（仍屬原 training partition），比較五個 ranks。15 個模型／rank 條件的 EEGLens 下游輸出與獨立 native hook **完全一致**；未選電極的時間平均 activation 完全不變。C4−C3 時間平均後與離線投影的最大絕對差為 1.67e−7（預設 float32 核對容許 atol/rtol=5e−5），這個浮點容許不適用於 native 下游輸出，它仍要求零誤差。最後 clean forward 恢復一致且所有暫時 hook 清除。

這一步確認了中心的定義：令 `x = mean_time(C4) − mean_time(C3)`，離線操作為 `x − (x − mean_fit(x)) Q Qᵀ`。模型內若分別使用 fitting subjects 的 C3、C4 平均值作中心，兩者相減便得到相同式子；共用一個中心則會抵消，無法對應此 centered contrast。

但此對應**不是唯一的原生干預**。分別投影 C3、C4 也會影響兩者的共同成分，以及時間平均之外的變動。即使 mean contrast 完全一致，仍不能宣稱只改動了 lateralization。後續任務實驗需要明確決定保留哪些成分，並以原生擾動能量及非目標資訊作對照，不能只靠離線 contrast 的代數等價。

驗證結果在 `results/iterative-native-v1/{cbramod,labram,csbrain}.json`。從 parent workspace 執行：

```bash
PYTHONPATH=eeglens/src python eeglens/research/mi/validate_iterative_native.py \
  --model labram --root research \
  --output research/eeglens_mi/iterative-native-reproduced/labram.json
```

此驗證需先取得原生 checkpoint、prepared training EEG、training features 與 iterative bases；沿用主研究的本機資料布局。它只驗證已列出的兩個 trials、五個 ranks 與指定模型配置，沒有重新完成全受試者的任務效果研究。
