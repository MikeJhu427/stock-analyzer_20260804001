# 📈 台股量化與基本面雙軌分析平台
**Taiwan Stock Quant & Fundamental Analysis Platform**

基於 Python 與 Streamlit 開發的全方位台股技術面量化與機構級基本面分析系統。本系統採用「核心邏輯抽離（Single Source of Truth）」的三層模組化架構，確保全市場自動化掃描與 Web 互動介面邏輯保持 100% 同步。

---

## 🏗️ 系統架構設計

為維持全市場自動化掃描與 Web 互動邏輯同步，系統將所有領域計算模型統一抽離至核心庫，並以三層架構進行分工：

```text
├── strategy_core.py       # 核心運算庫：指標計算、選股策略、財報診斷、幾何狀態機、三級備援爬蟲、雲端上傳
├── app.py                 # Streamlit 互動式視覺化分析平台（Web UI）
├── scan_task3.py          # 全市場自動化批次掃描引擎（CLI / GitHub Actions / Cron）
├── params_config.json     # 自訂策略參數設定檔（本機自動保存）
├── requirements.txt       # 依賴套件清單
└── README.md              # 專案說明文件
```

### 1) strategy_core.py（核心領域層）
集中維護以下能力：
- 技術指標計算
- 多週期背離分析
- 大盤濾網
- 各類量化選股策略
- 量化狀態機
- 三級備援爬蟲
- 雲端上傳整合

### 2) app.py（展示層 / Web UI）
提供直覺式操作介面，包含：
- 五大功能分頁
- 即時圖表視覺化
- 自選股批次傳送
- 參數配置管理
- 強制快取破壞（Cache Buster）
- 模組熱重載（Hot Reload）

### 3) scan_task3.py（排程層 / CLI）
負責全市場兩階段快速初篩與長天期深度複審，支援：
- CSV 輸出
- 定時掃描任務
- Webhook 自動備份

---

## ✨ 核心特色與功能模組

### 📊 Tab 1: 單檔深度解析（Technical Analysis）
針對單一個股進行深入的技術與籌碼結構分析，包含：

- 三大演算法計分：
  - 左側：底部翻轉判定
  - 右側：VCP 收斂判定
  - 雙指標共振分數
- 主力防守線追蹤：
  - 動態鎖定近 60 根 K 線的「最大量防守價」
  - 跌破即發出警訊
- 動能與極端訊號日誌：
  - 利用動能標準差（Momentum Force ±1.5σ）
  - 抓取異常爆量長紅 / 長黑
- VWMA 籌碼加權線：
  - 結合均量與加權均價
  - 判定短線實質解套賣壓與支撐
- 老手實戰交易濾網：
  - 整合 RSI 盤整區間
  - 日 / 週 KD 多週期趨勢共振
  - 建議持股水位

### 🚀 Tab 2: 全市場智慧掃描（Market Scanner）
全自動掃描台股上市櫃與 ETF 標的，具備三級備援容錯機制，發掘潛力飆股。

支援多策略引擎：
- 全部
- 底部翻轉
- VCP
- 指標共振
- 量價雙背離
- 長下引突破
- 回測走揚年線
- 低位長下引量縮突破

#### 兩階段（Two-Phase）掃描流程
- Phase 1：快速初篩
  - 利用短天期資料快速過濾打底未破與均線收斂標的
  - 大幅降低資料傳輸負載
- Phase 2：深度複審
  - 下載 2 年長天期資料
  - 驗證 240MA 斜率、年線支撐回測能力、週K 共振、日/60分K 雙級別背離

#### 其他功能
- 近 30 日觸發統計
- 時光機回測（指定歷史基準日）
- 設定檔管理（params_config.json）
- 跨頁連動
- CSV 匯出
- 一鍵傳送至基本面模組
- Google Drive 自動備份

### 📑 Tab 3: 財報三表交叉健檢 V5（Fundamental Audit）
超越傳統「紅燈數量 = 地雷」的固定邏輯，建立動態防誤判機制。

主要功能：
- 合約負債（Contract Liabilities）防誤判引擎
  - 自動抓取並交叉比對營收與現金流
  - 辨識「存貨增加」究竟是滯銷地雷，還是健康備貨
- 雙軌評分系統
  - 體質護城河分數（0–100）
  - 財務地雷風險分數（0–100）
- 長天期趨勢追蹤
  - 最新季度
  - 4 季短中期趨勢
  - 8 季歷史常態軌跡
- 七大專業診斷面板
  - 風險共振提示
  - 正向營運資金循環
  - 下一季追蹤事項
- 批次掃描
  - 可直接貼上自選股代號清單
  - 一鍵產生全清單評分與警訊比較總表

### 📐 Tab 4: 幾何型態滿足點測距（Geometric Pattern Projection）
利用極值演算法自動辨識波段轉折，計算對稱型態之目標價。

包含雙重型態模型：
- 模式一：實體收盤價模型
  - 過濾盤中突刺雜訊
  - 錨定主力與法人實體成本
- 模式二：影線極值模型
  - 涵蓋盤中斷頭恐慌踩踏與插針極端價

支援：
- 等幅與費氏延伸測距
- W 底突破與 M 頭跌破判定
- 1:1 等幅滿足位
- 1.618 費氏延伸目標價
- SPC 統計管制與 Z-Score

### 🤖 Tab 5: 量化型態狀態機（Quant Pattern State Machine）
結合狀態機概念，解決傳統形態學之滯後與誤判問題。

核心功能：
- 波段位階濾網（Regime Filter）
  - 近 60 日百分位統計
  - 排除高檔橫盤誤判 W 底與低檔頭頭高誤判 M 頭
- 斜向趨勢軌道與收斂楔形引擎
  - 動態上下軌
  - 支援上升/下降楔形、通道與對稱三角收斂
- 狀態機即時演化
  - FORMING
  - NECKLINE_TEST
  - DAY1_BREAK_WARNING
  - BREAKOUT_CONFIRMED
  - RETEST / BREAKOUT_FAILED
- 盤中即時模擬
  - 手動輸入盤中急殺/急拉價格
  - 提前模擬破線狀態
  - 評估風報比（Risk/Reward Ratio）

---

## 🛠️ 環境需求與安裝指引

### 1) 系統需求
- Python: 3.9 以上
- 作業系統: Windows / macOS / Linux

### 2) 安裝套件

請將以下內容存為 `requirements.txt`：

```txt
streamlit>=1.30.0
yfinance>=0.2.36
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
beautifulsoup4>=4.12.0
requests>=2.31.0
urllib3>=2.0.0
```

執行安裝：

```bash
pip install -r requirements.txt
```

---

## 🚀 執行方式

### 1) 啟動 Web 視覺化分析平台（Streamlit）
```bash
streamlit run app.py
```

啟動後，請在瀏覽器開啟：
```text
http://localhost:8501
```

### 2) 執行全市場無人值守掃描（CLI 批次腳本）
適合設定於伺服器排程（如 Crontab 或 GitHub Actions）每日盤後自動執行：

```bash
python scan_task3.py
```

---

## 🔒 雲端自動備份設定（選填）

若需將每日掃描產生的 CSV 報表自動上傳至 Google Drive，請設定 Google Apps Script（GAS）Webhook。

### Streamlit 應用（app.py）
在 `.streamlit/secrets.toml` 中加入：

```toml
GAS_WEBHOOK_URL = "https://script.google.com/macros/s/你的GAS發佈ID/exec"
GAS_SECRET_TOKEN = "你的自訂安全性Token"
```

### 批次掃描腳本（scan_task3.py）
設定系統環境變數：

```bash
export GAS_WEBHOOK_URL="https://script.google.com/macros/s/你的GAS發佈ID/exec"
export GAS_SECRET_TOKEN="你的自訂安全性Token"
```

---

## 📌 總結

本平台整合了：
- 技術面量化分析
- 全市場自動掃描
- 基本面財報診斷
- 幾何型態與狀態機分析
- 自動化排程與雲端備份

為台股愛好者與量化交易研究者提供一套完整的分析與策略驗證框架。
