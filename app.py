class LongLowerShadowBreakoutStrategy:
    @staticmethod
    def evaluate(df):
        res = pd.Series(False, index=df.index)
        if len(df) < 20: return res  
        df = df.copy()
        
        range_hl = (df['High'] - df['Low']).replace(0, 1e-8) 
        max_oc = df[['Open', 'Close']].max(axis=1)
        min_oc = df[['Open', 'Close']].min(axis=1)
        
        df['Upper_Shadow_Ratio'] = (df['High'] - max_oc) / range_hl
        df['Lower_Shadow_Ratio'] = (min_oc - df['Low']) / range_hl
        df['K_Amplitude'] = range_hl / df['Low'] * 100  
        
        if 'Pct_Change' not in df.columns:
            df['Pct_Change'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
        if 'MA20' not in df.columns:
            df['MA20'] = df['Close'].rolling(window=20).mean()
        if 'Vol_MA20' not in df.columns:
            df['Vol_MA20'] = df['Volume_Lots'].rolling(window=20).mean()
        if 'BB_Lower' not in df.columns:
            std20 = df['Close'].rolling(20).std()
            df['BB_Lower'] = df['MA20'] - 2 * std20

        # 1. 當日條件：紅K、漲幅>=3.5%、上影線放寬至 < 25%，且「今日量 > 昨日量」
        cond_today = (
            (df['Close'] > df['Open']) & 
            (df['Pct_Change'] >= 3.5) & 
            (df['Upper_Shadow_Ratio'] < 0.25) &  # 優化1：放寬上影線容忍度
            (df['Volume_Lots'] > df['Volume_Lots'].shift(1)) 
        )
        
        # 2. 關鍵K線條件：下影線佔半、最低價破前低、振幅>=3.0%、跌破布林下軌、收盤在月線下
        target_k_mask = (
            (df['Lower_Shadow_Ratio'] > 0.50) & 
            (df['Low'] < df['Low'].shift(1)) &     # 優化2：修正破前低邏輯為最低價比較 (非跳空開低)
            (df['K_Amplitude'] >= 3.0) & 
            # (df['Volume_Lots'] > df['Vol_MA20']) & # 優化3：洗盤打底通常為量縮，移除強制大於月均量的限制
            (df['Low'] < df['BB_Lower']) &
            (df['Close'] < df['MA20']) 
        )
        
        # 3. 尋找前 5 日內是否符合並突破
        for i in range(1, 6):
            is_target_k_past = target_k_mask.shift(i)
            target_k_high = df['High'].shift(i)
            
            cond_breakout = df['Close'] > target_k_high
            
            valid = cond_today & is_target_k_past & cond_breakout
            res = res | valid.fillna(False)
            
        return res
