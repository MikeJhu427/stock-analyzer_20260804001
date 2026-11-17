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
