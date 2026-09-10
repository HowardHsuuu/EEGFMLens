# Final readout 的 pooling 診斷

這是在看見 CBraMod／LaBraM 全電極 final readout 泛化失敗後提出的探索性分析。沿用原 training subjects 的 9/3/6 三折角色、同一個 ridge alpha grid，只改變 clean final-layer 特徵的讀法：200 維時間平均 C4−C3，或 400 維 C3/C4 串接。未讀取原 validation/test partition，未替換正在執行的 task study readout。

| 模型 | C4−C3 μ pooled R² | C4−C3 β pooled R² | μ 正 R² 人數 | β 正 R² 人數 | 串接 β pooled R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| CBraMod | −0.046 | 0.005 | 3/18 | 8/18 | 0.028 |
| LaBraM | 0.090 | 0.316 | 5/18 | 7/18 | 0.201 |
| CSBrain | −0.080 | −0.022 | 2/18 | 1/18 | −0.061 |

LaBraM 的全電極 3,800 維 final readout β pooled R² 是 −0.160，改用 final C4−C3 則為 0.316。這顯示原 readout 失敗不能被解釋為 final representation 沒有 β 資訊。這個比較同時改變維度、空間先驗與正則化問題，尚不能單獨歸因於「維度太高」。CBraMod 沒有同樣明顯的 pooled 改善，CSBrain 在這兩個候選表示下仍弱。

即使 LaBraM pooled β 改善，也只有 7/18 人得到正 R²，不能宣稱已解決逐受試者泛化。這些替代 readout 是看過結果後挑出的診斷候選，不是新的確認性結果。若要以它們重新衡量干預，需要另行標示探索性重分析，保留原固定 readout 結果，不能把較好的 clean pooling 分數直接套用為已完成的 native effect。

重現（parent workspace，NumPy，已有 training features 與 iterative-fold summary）：

```bash
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python eeglens/research/mi/diagnose_final_pooling.py \
  --model labram --root research \
  --output research/eeglens_mi/final-pooling-reproduced/labram.json
```

三模型結果在 `results/final-pooling-v1/`，保留 feature／fold source／runner／fitter hashes、每折所選 alpha，以及個別受試者 R²。
