# 📈 台股量化與基本面雙軌分析平台
**Taiwan Stock Quant & Fundamental Analysis Platform**

基於 Python 與 Streamlit 開發的全方位台股技術面量化、跨市場波動率與外資資金流向監控、實體留倉量籌碼過濾、純 Python 威科夫週期判定、空頭避風港選股與機構級基本面分析系統。本系統採用「核心邏輯抽離（Single Source of Truth）」的三層模組化架構，維持高度鬆散耦合，確保全市場自動化排程與 Web 互動介面邏輯保持 100% 同步。

---

## 🏗️ 系統架構設計

為維持全市場自動化掃描與 Web 互動邏輯同步，系統將所有領域計算模型統一抽離至核心庫，並以三層架構進行分工：

```text
├── strategy_core.py               # 核心運算庫：波動率監控、外資資金流向、實體留倉量、威科夫分析、方案B、8大選股策略、財報V5、幾何狀態機
├── app.py                         # Streamlit 互動式視覺化分析平台（Web UI，共 7 大分頁）
├── scan_task3.py                  # 全市場自動化批次掃描引擎（CLI / GitHub Actions / Cron，4 工作表 Excel 輸出）
├── params_config.json             # 自訂策略參數設定檔（本機自動保存）
├── foreign_fund_flow_history.xlsx # 外資資金流向本地歷史趨勢檔（自動依日期去重附加）
├── requirements.txt               # 依賴套件清單
└── README.md                      # 專案說明文件
```

### 1) `strategy_core.py`（核心領域層）
集中維護以下能力：
- **總經與籌碼風險監控**：跨市場波動率體制與避險／反彈監控（`CrossMarketVolatilityMonitor`，純 `yfinance` 資料源），以及外資實質資金流向監控（`fetch_foreign_fund_flow`，證交所 `BFI82U` ＋ `TWD=X` 匯率連動）。
- **技術指標與選股註冊中心**：KD、MACD、RSI、多週期背離分析、大盤濾網與鬆散耦合選股註冊中心（`ALGO_REGISTRY` / `ScanEngine`，涵蓋 8 大核心選股策略）。
- **入選標的后置深度檢核引擎（Phase 2）**：
  - **實體波段留倉量過濾**：`fetch_day_trading_volume` / `calc_real_volume_metrics` / `get_latest_real_volume_summary`（串接 FinMind 與證交所 `TWTB4U` 備援，扣除當沖並攔截 `UTAD_FAKE_VOLUME_TRAP` 假爆量誘多）。
  - **純 Python 威科夫週期判定**：`WyckoffPhaseAnalyzer`（100% 規則運算、零 AI API 呼叫，判定四大週期階段、關鍵事件、冰線支撐、阻力小溪與操作訊號）。
- **空頭避風港強勢股（方案 B）**：`BearMarketSafeHavenStrategy`（大盤雙波壓力測試閘門、純普通股過濾、60MA 防線與結構底背離超額報酬篩選）。
- **基本面與幾何引擎**：財報三表交叉健檢 V5（`FinancialAuditStrategy`，含合約負債防誤判）、幾何型態測距（`GeometricPatternStrategy`）與量化型態狀態機（`QuantPatternEngine`）。
- **基礎設施**：三級備援台股清單爬蟲、`Retry` 連線 Session 與 JSON + Base64 雲端無損上傳。

### 2) `app.py`（展示層 / Web UI）
提供直覺式操作介面，包含：
- 七大功能分頁（以「跨市場波動率監控」為預設首頁，並新增「空頭避風港 (方案B)」分頁）。
- 單檔解析與全市場掃描同步呈現「實體波段留倉量」與「威科夫週期階段」檢核結果。
- 即時圖表視覺化、手動重整快取控制、自選股批次傳送、Excel / CSV 下載與參數設定檔管理。

### 3) `scan_task3.py`（排程層 / CLI）
負責每日自動化盤後任務，包含：
- **Step 0**：執行跨市場波動率體制與避險／反彈監控並輸出終端機日誌。
- **Step 0.5**：執行外資實質資金流向監控，並與本地 `foreign_fund_flow_history.xlsx` 依日期去重合併保留趨勢。
- **Step 1 & Step 2**：全市場兩階段初篩（含 60 分 K 輔助）與 2 年長天期複審，並對入選標的自動執行「實體波段留倉量檢核」與「威科夫週期階段判定」。
- **方案 B 排程步驟**：檢核加權指數雙波測試閘門，達標時掃描純普通股並匯出 `plan_b_safe_haven_YYYY-MM-DD.csv`。
- **多工作表報表輸出**：產生 4 工作表 `.xlsx`（`全市場掃描結果`、`波動率體制監控`、`外資資金流向監控`、`空頭避風港_方案B`）並透過 Webhook 自動備份至 Google Drive（缺少 `openpyxl` 時自動降級為 `.csv`）。

---

## ✨ 核心特色與功能模組

### 🛡️ Tab 1: 跨市場波動率與外資資金流向監控
1. **跨市場波動率體制監控（`CrossMarketVolatilityMonitor`）**：
   - 專為海外機房部署設計，全程採用純 `yfinance`（`auto_adjust=False`、統一轉換為台北時間 UTC+8），監控 `^VIX`、`^VXN`、`^VVIX` 與 `^TWII` 本地波動替代指標（**HV20** 與 **ATR14%**）。
   - 自動切換 **【戰時恐慌警戒區】**（狀態 A 止跌反彈確認／狀態 B 尚未止血警戒）與 **【平時常態震盪區】**（狀態 C 大跌前避險預警／狀態 D 結構正常），提供燈號看板、8 組指標卡片與近 3 個月 4 聯走勢圖。
2. **外資實質資金流向監控（`fetch_foreign_fund_flow`）**：
   - 同步抓取台灣證交所 `BFI82U`「外資及陸資（不含外資自營商）」單日與當月累計買賣超（億元），以及 `yfinance` `TWD=X` 近 5 日匯率變動。
   - 依股匯連動自動判定四種流向標籤：**實質撤資匯出警報**、**帳上換股停泊（未匯出）**、**實質資金匯入偏多**、**中性常態波動**，並整合歷史趨勢表。

### 📊 Tab 2: 單檔深度解析（Technical Analysis）
針對單一個股進行近兩年技術與籌碼深度診斷，包含：
- 三大演算法計分（底部翻轉、VCP 收斂、指標共振）與量化雙底正背離模組。
- 主力 60 日最大量防守線、VWMA20 籌碼線、動能標準差（$\pm 1.5\sigma$）與老手實戰濾網（RSI 盤整、多週期 KD、建議水位）。
- **實體波段留倉量卡片**：即時顯示 `當沖率(%)`、`實體留倉量(張)`、`實體量比` 與 `量能結構標籤`，觸發 `UTAD_FAKE_VOLUME_TRAP` 時顯示紅框警示。
- **威科夫量價週期卡片**：即時顯示威科夫四大階段、核心事件、冰線支撐、阻力小溪、騙線警示與操作訊號。

### 🚀 Tab 3: 全市場智慧掃描（Market Scanner）
全自動掃描台股上市櫃與 ETF 標的，具備三級備援容錯機制：
- **支援 8 大策略引擎（`ALGO_REGISTRY`）**：全部、底部翻轉、VCP、指標共振、量價雙背離、長下引突破、*回測走揚年線、低位長下引量縮突破、雙底正背離與右肩回測。
- **Phase 2 後置深度複審**：
  - 長天期 120MA/240MA 斜率與空排、年線回測評分、長／短天期雙軌均線糾結、週 K 共振與日／60 分 K 背離。
  - **實體波段留倉量過濾**：自動扣除當沖成交張數，輸出 `當沖率(%)`、`實體留倉量(張)`、`實體量比`、`量能結構標籤`；觸發高當沖假爆量誘多（`UTAD_FAKE_VOLUME_TRAP`）時強制剔除多頭進場建議。
  - **純 Python 威科夫週期標註**：直接複用已下載的 2 年 `long_df`（零額外 API 請求、零 AI 呼叫），輸出 `威科夫階段`、`威科夫事件`、`威科夫信心分`、`冰線支撐`、`阻力小溪`、`威科夫操作訊號` 與判讀摘要。
- **其他功能**：近 30 日觸發統計、時光機回測、設定檔管理、Excel / CSV 匯出、一鍵傳送至基本面模組、Google Drive 自動備份。

### 📑 Tab 4: 財報三表地雷掃描（Fundamental Audit V5）
建立動態合約負債（Contract Liabilities）防誤判引擎，交叉比對損益表、資產負債表與現金流量表（涵蓋最新季、4 季與 8 季軌跡），區分滯銷地雷與健康接單備料，產出 0–100 分「體質分數」與「風險分數」雙軌評級，支援單檔與批次健檢匯出。

### 📐 Tab 5: 幾何型態滿足點測距（Geometric Pattern Projection）
提供「實體收盤價模型」與「影線極值模型」雙軌 W 底／M 頭辨識，自動測算 1:1 等幅滿足位、1.618 費氏延伸目標價及 SPC Z-Score 統計管制界限。

### 🤖 Tab 6: 量化型態狀態機（Quant Pattern State Machine）
結合 60 日波段位階濾網、斜向趨勢軌道／收斂楔形引擎與即時狀態演化，支援盤中即時價模擬與實質風險報酬比試算。

### 🛡️ Tab 7: 空頭避風港（方案 B — Bear Market Safe Haven）
專為台股大盤進入「第一波恐慌宣洩後、第二波回測前低或破底壓力測試期」設計的逆勢強勢股篩選引擎：
- **雙重大盤閘門**：驗證 `^TWII` 位於 60MA 之下且近 10 日至少 7 日破季線，同時近 7 日低點回測或跌破前段（第 -25 至 -8 日）低點 `Low_1` 的 $1.01$ 倍以內；未達條件時自動休眠不啟動掃描。
- **個股 5 項嚴格測試**：排除 ETF/權證/ETN/TDR 僅保留普通股；近 15 日全程守穩 60MA 且季線走平上揚、近 7 日低點高於前段低點至少 $+2\%$、15 日相對大盤超額報酬 $\text{RS} > +8\%$、5 日均量 $> 1,500$ 張。
- **訊號分級與匯出**：區分 `🚀 突破點火（右側試單訊號）` 與 `🛡️ 通過雙波測試（抗跌蓄勢觀察中）`，依 `15日超額報酬RS(%)` 降冪排序並支援 `UTF-8-SIG` CSV 匯出。

---

## 🛠️ 環境需求與安裝指引

### 1) 系統需求
- Python：3.9 以上
- 作業系統：Windows / macOS / Linux

### 2) 安裝套件
```bash
pip install -r requirements.txt
```
（核心依賴：`streamlit`、`yfinance`、`pandas`、`numpy`、`matplotlib`、`beautifulsoup4`、`requests`、`scipy`、`html5lib`、`lxml`、`openpyxl`、`plotly`）

---

## 🚀 執行方式

### 1) 啟動 Web 視覺化分析平台（Streamlit）
```bash
streamlit run app.py
```
啟動後開啟 `http://localhost:8501`，預設載入「🛡️ 跨市場波動率監控」首頁。

### 2) 執行全市場無人值守掃描（CLI 批次腳本）
```bash
python scan_task3.py
```

---

## 🔒 雲端自動備份與 API 設定（選填）

### Google Drive Webhook（`.streamlit/secrets.toml` 或系統環境變數）
```toml
GAS_WEBHOOK_URL = "https://script.google.com/macros/s/你的GAS發佈ID/exec"
GAS_SECRET_TOKEN = "你的自訂安全性Token"
```

### FinMind API Token（選填，提升當沖查詢額度）
```bash
export FINMIND_TOKEN="你的FinMind_Token"
```
（若未設定將自動使用免費額度，遇限流時無縫切換至證交所 `TWTB4U` 備援快取。）