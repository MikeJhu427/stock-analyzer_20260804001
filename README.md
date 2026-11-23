# 📈 台股量化與基本面雙軌分析平台
**Taiwan Stock Quant & Fundamental Analysis Platform**

基於 Python 與 Streamlit 開發的全方位台股技術面量化、總經波動率與外資資金流向監控、空頭避風港選股、威科夫吸籌（Wyckoff Accumulation）量化偵測、實體波段留倉量（Real Turnover Volume）籌碼判讀，以及機構級基本面分析系統。本系統採用「核心邏輯抽離（Single Source of Truth）」的三層模組化架構，維持高度鬆散耦合，確保全市場自動化排程掃描與 Web 互動介面邏輯保持 100% 同步。

---

## 🏗️ 系統架構設計

為維持全市場自動化掃描與 Web 互動邏輯同步，系統將所有領域計算模型統一抽離至核心庫，並以三層架構進行分工：

```text
├── strategy_core.py                   # 核心運算庫：波動率體制監控、外資資金流向、方案B空頭避風港、威科夫吸籌偵測、實體波段留倉量分析、指標計算(含ATR14)、8大選股策略、財報診斷、幾何狀態機、三級備援爬蟲、雲端上傳
├── app.py                             # Streamlit 互動式視覺化分析平台（Web UI，共 7 大分頁）
├── scan_task3.py                      # 全市場自動化批次掃描引擎（CLI / GitHub Actions / Cron，4 工作表 Excel、CSV 與方案 B CSV 輸出）
├── params_config.json                 # 自訂策略參數設定檔（本機自動保存）
├── foreign_fund_flow_history.xlsx     # 外資資金流向歷史紀錄（自動依日期去重附加）
├── plan_b_safe_haven_YYYY-MM-DD.csv   # 方案 B 空頭避風港盤後選股報表（大盤閘門通過時產生，UTF-8-SIG 編碼）
├── requirements.txt                   # 依賴套件清單
└── README.md                          # 專案說明文件
```

### 1) `strategy_core.py`（核心領域層）
集中維護以下能力：
- **全域參數集中管理**（`DEFAULT_CONFIG`）：統一控管第一階段流動性門檻（`phase1_liquidity_min_vol_ma20_lots = 300` 張）、8 大策略門檻與威科夫吸籌參數字典（`wyckoff_params`）。
- **跨市場波動率體制與避險／反彈監控**（`CrossMarketVolatilityMonitor`，純 `yfinance` 資料源，相容海外機房部署）。
- **外資實質資金流向監控**（`fetch_foreign_fund_flow`／`build_foreign_fund_flow_df`，串接證交所 `BFI82U` 與 `TWD=X` 匯率連動判讀）。
- **空頭避風港強勢股引擎（方案 B）**（`BearMarketSafeHavenStrategy`，大盤雙波回測閘門、普通股過濾、`ThreadPoolExecutor` 多執行緒批次掃描與快取複用）。
- **威科夫吸籌量化偵測模組**（`WyckoffAccumulationAnalyzer`／`WYCKOFF_COLUMNS`，獨立純運算函式，包含 14 日 ATR、TR 盤整區間界定、SC／努力與結果背離／Spring／SOS 事件偵測、嚴格順序 FSM 與破底重置、台股法人背離預留介面）。
- **實體波段留倉量後置分析模組**（`RealTurnoverVolumeAnalyzer`，以 FinMind `TaiwanStockDayTrading` 為主、證交所 `TWTB4U` 與櫃買中心當沖統計為備援，僅針對篩選後入選標的附加 8 項實體量與資料日期同步欄位，不覆蓋既有表面成交量與選股邏輯）。
- **技術指標與選股註冊中心**（KD、MACD、RSI、ATR14、多週期背離、大盤濾網 `MarketRegimeFilter`、老手實戰濾網 `ExpertTradingStrategy`，以及 `ALGO_REGISTRY`／`ScanEngine` 8 大策略）。
- **財報三表交叉健檢 V5**（`FinancialAuditStrategy`，含合約負債防誤判引擎與金融股辨識）。
- **幾何型態測距與量化型態狀態機**（`GeometricPatternStrategy`、`QuantPatternEngine`）。
- **基礎設施**：三級備援爬蟲（`fetch_all_tw_stocks`）與 JSON + Base64 雲端上傳整合（`GoogleDriveUploader`）。

### 2) `app.py`（展示層 / Web UI）
提供直覺式操作介面，包含：
- **七大功能分頁**（以「🛡️ 跨市場波動率監控」為預設首頁，尾端獨立配置「🛡️ 空頭避風港 (方案B)」分頁）。
- 即時圖表視覺化（近 3 個月 4 層波動率走勢圖、近 240 日個股 3 層解析圖）、大盤閘門即時看板與手動重整快取控制。
- 全市場智慧掃描第一階段強制套用月均量 $> 300$ 張流動性濾網（指定個股測試模式可繞過）；第二階段整合威科夫吸籌 5 項欄位與後置 8 項實體量欄位，支援 `.xlsx`（含數值格式與自動欄寬）與 `.csv`（`UTF-8-SIG`）雙格式下載、0 檔入選空表結構保留，以及 Google Drive 自動上傳。
- 自選股一鍵傳送至財報批次健檢、參數設定檔管理（`params_config.json`）、`@st.cache_data` 記憶體快取機制與 `importlib.reload` 模組熱重載。

### 3) `scan_task3.py`（排程層 / CLI）
負責每日自動化盤後任務，執行管線包含：
- **Step 0**：執行跨市場波動率體制與避險／反彈監控並輸出終端機日誌。
- **Step 0.5**：執行外資實質資金流向監控，並與本地 `foreign_fund_flow_history.xlsx`（或既有掃描報表）依日期去重合併保留趨勢。
- **Step 1（初篩與流動性過濾）**：全市場每 40 檔分批下載日線與 60 分 K，優先快取達標長度的上市櫃普通股日線供方案 B 複用，接著統一將成交量換算為「張」並執行**月均量（20 日均量）$> 300$ 張**前置流動性過濾，未達標標的直接跳過後續指標運算。
- **Step 2（長天期深度複審與威科夫偵測）**：針對初篩標的下載 2 年長天期日線，計算年線回測評分、雙軌均線糾結、週 K／日 K 背離，並呼叫 `WyckoffAccumulationAnalyzer.evaluate()` 產出 5 項威科夫吸籌欄位。
- **Step 2.5**：僅針對 Step 2 最終入選的候選標的，批次抓取近 20 個交易日當沖資料，附加 8 項實體波段留倉量與日期同步狀態欄位。
- **Step 3**：執行方案 B 空頭避風港大盤閘門檢核；若大盤通過條件 1 與條件 2，優先複用 Step 1 快取執行全市場普通股掃描並匯出本地 `plan_b_safe_haven_YYYY-MM-DD.csv`（即使主掃描為 0 檔仍會執行方案 B）。
- **報表輸出（`export_scan_reports`）**：同步更新本地 `foreign_fund_flow_history.xlsx`，支援透過環境變數 `SCAN_EXPORT_FORMAT` 切換 `xlsx`（預設）、`csv` 或 `both`；透過 `openpyxl` 產生 **4 工作表 `.xlsx`**（Sheet 1：`全市場掃描結果` 含自動欄寬與威科夫／實體量數值格式、Sheet 2：`波動率體制監控`、Sheet 3：`外資資金流向監控`、Sheet 4：`空頭避風港_方案B`）自動備份至 Google Drive。即使當日入選標的為 0 檔，仍保留完整原有欄位＋威科夫 5 項欄位＋8 項實體量欄位的空表標頭結構。

---

## ✨ 核心特色與功能模組

### 🛡️ Tab 1: 跨市場波動率監控（含外資實質資金流向連動）
專為 GitHub Actions 與 Streamlit Cloud 等海外機房 IP 設計，波動率監控全程採用純 `yfinance`（`auto_adjust=False, progress=False`，相容 MultiIndex 並統一轉換為台北時間 UTC+8），並整合外資匯率與現貨買賣超連動判讀與近 3 個月 4 層走勢圖。

- **跨市場波動率體制與四狀態訊號**：
  - 監控 `^VIX`、`^VXN`、`^VVIX` 與 `^TWII`（同步計算 20 日年化已實現波動率 `HV20` 與 `ATR14%` 作為台股本地波動替代指標，取代易遭封鎖之 VIXTWN）。
  - **【戰時恐慌警戒區】**（$\text{VIX} > 25$ 或 $\text{VXN} > 30$ 或突破 20 日布林上軌）：區分 **狀態 A（止跌反彈確認）** 與 **狀態 B（尚未止血警戒）**，自動偵測恐慌高點下移背離與科技股風險溢價（$\text{VXN} - \text{VIX}$）擴大警示。
  - **【平時常態震盪區】**（$\text{VIX} \le 25$ 且 $\text{VXN} \le 30$ 且未破布林上軌）：區分 **狀態 C（大跌前避險預警，由 VVIX 或台股 HV20／ATR14% 低分位突破觸發）** 與 **狀態 D（結構正常）**。
- **💱 外資資金流向監控（匯率＋現貨買賣超連動）**：
  - 自動抓取台灣證交所 `BFI82U`「外資及陸資（不含外資自營商）」單日買進、賣出、買賣超與當月累計買賣超（週末與休市自動回退），並連動 `yfinance` 的 `TWD=X` 計算單日匯率漲跌幅與變動角數。
  - 自動輸出四大流向判定：`實質撤資匯出警報`、`帳上換股停泊（未匯出）`、`實質資金匯入偏多`、`中性常態波動`，並合併本地歷史紀錄表單檢視。
- **📈 近 3 個月 4 層走勢圖**：依序繪製 `^TWII` 均線、`^VIX`／`^VXN` 與布林上軌、`^VVIX` 與科技股風險溢價，以及台股 `HV20%`／`ATR14%` 本地波動走勢。

### 📊 Tab 2: 單檔深度解析（Technical Analysis）
針對單一個股抓取近 2 年歷史資料進行技術與籌碼結構診斷：
- **三大演算法計分卡**：即時呈現底部翻轉分數、VCP 收斂分數、指標共振分數與達標門檻狀態。
- **老手實戰交易濾網**：提供 RSI 盤整過濾、日／週多週期 KD 狀態與三指標共振持股水位建議。
- **量化雙底正背離提示**：觸發 Signal A（底倉突破）或 Signal B（右肩回測加碼）時自動顯示左腳低點、破底停損價、頸線、右肩支撐與 1 段／3 段目標價。
- **位階摘要與近 240 日三層走勢圖**：彙整月線趨勢方向、VWMA20 籌碼支撐與近 60 日最大量主力防守線，並繪製價格防線圖、動能標準差（$\pm 1.5\sigma$）區間圖與反轉分數柱狀圖。

### 🚀 Tab 3: 全市場智慧掃描（Market Scanner + 威科夫吸籌 + 實體波段留倉量）
全自動掃描台股上市櫃與 ETF 標的（亦支援「指定個股測試」），具備三級備援容錯機制，透過 `ALGO_REGISTRY` 支援 8 大策略引擎（底部翻轉、VCP、指標共振、量價雙背離、長下引突破、*回測走揚年線、低位長下引量縮突破、雙底正背離與右肩回測）。
- **第一階段流動性前置過濾**：將 `yfinance` 股數統一除以 $1{,}000$ 換算為「張」，僅保留最新 20 日均量 $> 300$ 張之標的進入技術指標與策略評估，剔除低流動性冷門股並加速全市場掃描。
- **第二階段威科夫吸籌（Wyckoff Accumulation）量化偵測**：針對通過初篩之標的，以 2 年期長天期日線執行 14 日 ATR 標準化、TR 支撐／壓力界定與嚴格順序有限狀態機（`DOWNTREND` $\rightarrow$ `SC_DETECTED` $\rightarrow$ `TR_FORMED` $\rightarrow$ `SPRING_TRIGGERED` $\rightarrow$ `SOS_CONFIRMED`，含破底連續 3 日重置），附加 5 項欄位：
  - `威科夫狀態`、`威科夫TR支撐價`、`威科夫TR壓力價`、`威科夫最近事件`、`威科夫事件日期`。
- 提供兩階段掃描、選配 **60 分 K 雙底／右肩掃描**（Web 端預設關閉以維持互動速度，排程端 `scan_task3.py` 預設開啟）、長短天期雙軌均線糾結判定、近 30 日觸發統計、指定日期時光機回測與大盤環境濾網。
- **實體波段留倉量後置擴充（非破壞性附加）**：在既有篩選邏輯完全結束後，僅針對最終入選清單呼叫當沖 API，於報表最右側依序新增 8 項欄位：
  - `實體量資料日期`、`實體量是否更新至指定日期`（`已更新`／`未更新(最新:YYYY-MM-DD)`）、`當沖張數`、`當沖率(%)`、`實體留倉量(張)`、`實體20日均量(張)`、`實體量比`、`實體量能判讀`（`有效實體放量`／`⚠️高當沖假爆量陷阱`／`一般換手`）。
- 支援設定檔儲存／刪除（`params_config.json`）、`.xlsx` 與 `.csv`（`UTF-8-SIG`）雙格式下載（0 檔入選亦維持完整欄位標頭）、Google Drive 自動備份，以及一鍵將入選標的傳送至 Tab 4 進行財報批次健檢。

### 📑 Tab 4: 財報三表地雷掃描（Fundamental Audit V5）
透過損益表、資產負債表與現金流量表交叉勾稽，內建**合約負債（Contract Liabilities）防誤判引擎**，精準區分滯銷地雷與健康接單備料。提供體質綜合分數（0–100）與財務風險分數（0–100）雙軌評分，涵蓋最新季度、4 季短中期與 8 季歷史軌跡，支援單檔七大診斷面板與自選股批次健檢 Excel／CSV 下載。

### 📐 Tab 5: 幾何型態滿足點測距（Geometric Pattern Projection）
自動辨識波段轉折極值，提供「模式一：實體收盤價模型」與「模式二：影線極值模型」之 W 底／M 頭判定，同步計算 1:1 等幅滿足位、1.618 費氏延伸目標價與 SPC 統計管制（$\pm 2\sigma$ 與 Z-Score）。

### 🤖 Tab 6: 量化型態狀態機（程式 A / Quant Pattern State Machine）
結合 60 日波段位階濾網、斜向趨勢軌道／收斂楔形引擎與即時狀態演化（`FORMING`、`NECKLINE_TEST`、`DAY1_BREAK_WARNING`、`BREAKOUT_CONFIRMED`、`RETEST`、`BREAKOUT_FAILED`、`INVALIDATED`），支援參數微調、盤中即時價模擬與以頸線防守之實質風險報酬比試算。

### 🛡️ Tab 7: 空頭避風港 (方案B)（Bear Market Safe Haven Scanner）
專為大盤跌破季線並進入第二波回測壓力測試期打造的盤後強勢股掃描器，嚴格過濾上市（`.TW`）與上櫃（`.TWO`）4 碼普通股（自動排除 ETF、權證、ETN 與 TDR）。

- **大盤雙重閘門即時看板（`^TWII`）**：
  - **條件 1（時間閘門）**：加權指數最新收盤低於 `60MA`，且過去 10 個交易日至少 7 日收盤位於 `60MA` 之下。
  - **條件 2（雙波壓力測試）**：近 7 個交易日最低點再次回測至前段（第 $-25$ 至 $-8$ 交易日）波段低點 $\text{Low}_1 \times 1.01$ 以內，或直接破底創新低。
  - 若大盤未同時滿足條件 1 與條件 2，直接顯示鎖定提示並停止個股下載，避免於第一波無差別恐慌期過早進場。
- **個股三大嚴格篩選條件**：
  - **條件 3（守穩季線）**：過去 15 個交易日每日收盤皆大於當日 `60MA`，且最新 `60MA` 維持走平或上揚（$\ge$ 10 個交易日前 `60MA` $\times 0.99$）。
  - **條件 4（底背離與強勢超額報酬）**：近 7 日低點高於前段低點至少 $+2\%$，且近 15 日對大盤超額報酬 $\text{RS} > +8\%$。
  - **條件 5（流動性濾網）**：5 日均量 $> 1{,}500$ 張（$1{,}500{,}000$ 股）。
- **訊號分級與輸出**：
  - `🚀 突破點火（右側試單訊號）`：收盤突破近 10 日（不含當日）最高價，且當日量 $>$ 前一日 5 日均量 $\times 1.3$。
  - `🛡️ 通過雙波測試（抗跌蓄勢觀察中）`：符合全部 5 項核心條件但尚未帶量突破。
  - 結果依 `15日超額報酬RS(%)` 由高到低排序，輸出標準 9 欄表格並提供含當日日期之 `UTF-8-SIG` 編碼 CSV 下載。

---

## 🛠️ 環境需求與安裝指引

### 1) 系統需求
- Python：3.9 以上
- 作業系統：Windows / macOS / Linux

### 2) 安裝套件
確認 `requirements.txt` 包含以下依賴套件（多執行緒採用 Python 標準庫 `concurrent.futures`，不需額外安裝）：

```txt
streamlit
yfinance
pandas
numpy
matplotlib
beautifulsoup4
requests
urllib3
scipy
openpyxl
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

啟動後，請在瀏覽器開啟 `http://localhost:8501`，預設將直接載入「🛡️ 跨市場波動率監控」首頁，可透過上方分頁列快速切換至其餘 6 項分析功能。

### 2) 執行全市場無人值守掃描（CLI 批次腳本）
適合設定於伺服器排程（如 Crontab 或 GitHub Actions）每日盤後自動執行：

```bash
python scan_task3.py
```

可選環境變數切換輸出格式（預設為 `xlsx`）：

```bash
export SCAN_EXPORT_FORMAT="xlsx"   # 支援 xlsx / csv / both
```

---

## 🔒 雲端自動備份設定（選填）

若需將掃描產生的報表（`.xlsx` 或 `.csv`）自動上傳至 Google Drive，請設定 Google Apps Script（GAS）Webhook。

### Streamlit 應用（`app.py`）
在 `.streamlit/secrets.toml` 中加入（亦支援讀取同名環境變數）：

```toml
GAS_WEBHOOK_URL = "https://script.google.com/macros/s/你的GAS發佈ID/exec"
GAS_SECRET_TOKEN = "你的自訂安全性Token"
```

### 批次掃描腳本（`scan_task3.py`）
設定系統環境變數：

```bash
export GAS_WEBHOOK_URL="https://script.google.com/macros/s/你的GAS發佈ID/exec"
export GAS_SECRET_TOKEN="你的自訂安全性Token"
```

---

## 📌 總結

本平台整合了：
- 跨市場波動率體制與避險／反彈監控（純 `yfinance`、無封鎖風險）
- 外資實質資金流向監控（證交所 `BFI82U` 現貨買賣超連動 `USD/TWD` 匯率與本地歷史追蹤）
- 空頭避風港強勢股盤後掃描器（方案 B：大盤雙波回測閘門 + 季線抗跌與 RS 強勢篩選）
- 第一階段月均量 $> 300$ 張流動性濾網與技術面 8 大策略量化掃描（含 60 分 K 雙底、雙軌均線糾結與多級別背離）
- 第二階段威科夫吸籌（Wyckoff Accumulation）嚴格順序狀態機（TR 支撐／壓力、SC／努力結果背離／Spring／SOS 與破底重置）
- 入選標的實體波段留倉量後置分析（當沖率、實體 20 日均量、實體量比與日期同步檢核）
- 財報三表交叉診斷與合約負債防誤判引擎
- 幾何滿足點測距與量化型態狀態機
- 四工作表自動化排程、`.xlsx`／`.csv` 雙格式匯出（含空表結構保留）與 JSON + Base64 雲端無損備份