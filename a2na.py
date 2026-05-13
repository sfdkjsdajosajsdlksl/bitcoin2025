import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

# scikit-learn 설치 확인 및 에러 핸들링
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.error("머신러닝 라이브러리가 설치되지 않았습니다. 터미널에 'pip install scikit-learn'을 입력해주세요.")
    st.stop()

# 페이지 설정
st.set_page_config(page_title="비트코인 AI 예측 대시보드", layout="wide")

@st.cache_data
def load_data():
    try:
        # 데이터 읽기 (sep=';' 고정 및 잘못된 줄 무시 옵션 강화)
        df = pd.read_csv('bitcoin.csv', sep=';', quotechar='"', on_bad_lines='skip', engine='python')
        
        # 컬럼명 정리
        df.columns = df.columns.str.strip()
        
        # 날짜 데이터 변환
        df['timeOpen'] = pd.to_datetime(df['timeOpen'], errors='coerce')
        df = df.dropna(subset=['timeOpen']) # 날짜 없는 줄 제거
        df = df.sort_values('timeOpen')
        
        # 숫자 데이터 변환
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 데이터가 비어있는지 확인
        if df.empty:
            st.error("데이터 파일이 비어있거나 형식이 잘못되었습니다.")
            return None
            
        return df.dropna(subset=['close'])
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None

def predict_price(df):
    # 학습을 위해 날짜를 숫자로 변환
    df['date_num'] = np.arange(len(df))
    
    X = df[['date_num']].values
    y = df['close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 내일 가격 예측
    next_day_num = len(df)
    tomorrow_date = df['timeOpen'].max() + timedelta(days=1)
    prediction = model.predict([[next_day_num]])[0]
    
    return tomorrow_date, prediction, model

# 데이터 실행
df = load_data()

if df is not None:
    st.title("₿ 비트코인 AI 시세 예측 대시보드")
    st.markdown("---")

    # 예측 엔진 작동
    tomorrow_date, pred_price, model = predict_price(df)
    current_price = df['close'].iloc[-1]
    change = pred_price - current_price
    status = "상승 📈" if change > 0 else "하락 📉"

    # 상단 요약 정보
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("오늘 현재가", f"₩{current_price:,.0f}")
    with c2:
        st.metric(f"내일 ({tomorrow_date.strftime('%m/%d')}) 예측가", f"₩{pred_price:,.0f}", f"{change:,.0f} ({status})")
    with c3:
        st.info(f"**AI 예측:** 과거 추세를 분석한 결과 내일은 **{status}**할 것으로 보입니다.")

    st.markdown("---")

    # 차트 영역
    st.subheader("가격 추이 및 AI 추세선 (Linear Regression)")
    
    # 추세선 계산
    df['trend'] = model.predict(df[['date_num']].values)
    
    fig = go.Figure()
    # 캔들스틱 추가
    fig.add_trace(go.Candlestick(
        x=df['timeOpen'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="실제 시세"
    ))
    # 추세선 추가
    fig.add_trace(go.Scatter(
        x=df['timeOpen'], y=df['trend'], name="AI 추세선",
        line=dict(color='yellow', width=2, dash='dash')
    ))
    
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # 데이터 하단 정보
    with st.expander("원본 데이터 확인"):
        st.write(df.sort_values('timeOpen', ascending=False))
