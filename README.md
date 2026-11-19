# car-acoustic-diag

汽車機械異音 AI 聲學診斷 PoC（Proof of Concept）

用多點聲學採集＋車速同步，把引擎／傳動／輪胎等旋轉機械的聲音轉到「階次域（Order Domain）」，
再以自編碼器重建誤差做早期異常偵測：在異音還很微弱、人耳還聽不出來時先發出警示。

## 架構

```
ESP32-S3 多點 I2S MEMS 麥克風 + 車速同步採集
        │  AudioFrame { sync_ms, speed_kmh, int16 pcm[256] }
        ▼
Order Resampling 階次對齊（時間域 → 角度／階次域，消除轉速漂移造成的頻譜塗抹）
        │
        ▼
Log-Mel 頻譜前處理（Mel 濾波器組 + log 壓縮）
        │  (n_mels, frames) 特徵圖
        ▼
PyTorch Conv-Autoencoder
        │  只用正常聲音訓練，異常聲音重建不回來
        ▼
重建誤差 > 門檻 → 早期異常警示
```

設計重點：

- **車速同步**：每一幀聲音都帶 `sync_ms` 時間戳與當下 `speed_kmh`，
  後端才能把「頻率（Hz）」換算成與轉速綁定的「階次（order = cycles/rev）」。
- **階次對齊**：轉速變化時，機械異音的 Hz 會跟著漂；對齊到階次域後，
  同一個零件的異音永遠落在同一個 order，模型不必學會所有轉速。
- **重建誤差偵測**：不需要收集大量故障樣本。Autoencoder 只學正常聲音，
  異常時重建誤差自然升高，以正常資料的誤差分位數（如 95%）定門檻。

## 目錄

```
car-acoustic-diag/
├── README.md
├── requirements.txt
├── firmware/
│   └── esp32_i2s_capture.ino   # ESP32-S3 I2S MEMS 採集 + AudioFrame 打包
├── dsp/
│   ├── __init__.py
│   ├── order_resampling.py     # 時間域 → 階次域重採樣、階次頻譜、車速→輪轉速
│   └── log_mel.py              # Log-Mel 頻譜（numpy 實作，無 librosa 依賴）
├── model/
│   ├── __init__.py
│   └── conv_autoencoder.py     # PyTorch Conv-Autoencoder + 重建誤差 / 門檻校準
└── tests/
    ├── test_order_resampling.py
    ├── test_log_mel.py
    └── test_conv_autoencoder.py   # 未安裝 torch 時自動 skip
```

## 快速開始

```bash
pip install -r requirements.txt
pytest -q
```

## 狀態

PoC 骨架：韌體可編譯原型、DSP 與模型皆有單元測試覆蓋核心行為。
下一步（未做）：CAN/OBD 車速實讀、無線傳輸協定、實車資料收集與門檻校準。
