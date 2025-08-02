import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# 페이지 설정
st.set_page_config(
    page_title="이동평균선 백테스팅 분석기",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'stock_list' not in st.session_state:
    st.session_state.stock_list = pd.DataFrame()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

@st.cache_data
def load_stock_list():
    """KRX 종목 리스트를 로드하여 캐시에 저장"""
    try:
        stock_list = fdr.StockListing('KRX')
        if not stock_list.empty:
            stock_list.set_index('Code', inplace=True)
        return stock_list
    except Exception as e:
        st.error(f"한국거래소 종목 목록을 불러오는 데 실패했습니다: {e}")
        return pd.DataFrame()

def get_stock_name(ticker, stock_list):
    """티커로 종목명 조회"""
    if not stock_list.empty and ticker in stock_list.index:
        return stock_list.loc[ticker, 'Name']
    return "종목명을 찾을 수 없습니다"

def backtest_ma_strategy(data, ma_period, selling_fee):
    """이동평균선 전략 백테스팅"""
    df = data.copy()
    
    # 이동평균 계산
    df[f'MA{ma_period}'] = df['Close'].rolling(window=ma_period).mean()
    
    # 매매 신호 생성
    df['Position'] = 0
    df['Signal'] = 0
    
    # 이동평균선 돌파 전략
    for i in range(1, len(df)):
        if pd.notna(df[f'MA{ma_period}'].iloc[i]) and pd.notna(df[f'MA{ma_period}'].iloc[i-1]):
            # 상향 돌파시 매수
            if (df['Close'].iloc[i] > df[f'MA{ma_period}'].iloc[i] and 
                df['Close'].iloc[i-1] <= df[f'MA{ma_period}'].iloc[i-1]):
                df.loc[df.index[i], 'Signal'] = 1
                df.loc[df.index[i], 'Position'] = 1
            # 하향 돌파시 매도
            elif (df['Close'].iloc[i] < df[f'MA{ma_period}'].iloc[i] and 
                  df['Close'].iloc[i-1] >= df[f'MA{ma_period}'].iloc[i-1]):
                df.loc[df.index[i], 'Signal'] = -1
                df.loc[df.index[i], 'Position'] = 0
            else:
                # 이전 포지션 유지
                df.loc[df.index[i], 'Position'] = df['Position'].iloc[i-1]
    
    # 수익률 계산
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Daily_Return'] * df['Position'].shift(1)
    df['Strategy_Return_with_Fee'] = df['Strategy_Return'].copy()
    
    # 매도 수수료 반영
    for i in range(len(df)):
        if df['Signal'].iloc[i] == -1:
            df.loc[df.index[i], 'Strategy_Return_with_Fee'] = df['Strategy_Return'].iloc[i] - selling_fee
    
    # 누적 수익률 계산
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return_with_Fee']).cumprod() - 1
    
    # 성과 지표 계산
    final_return = df['Cumulative_Strategy_Return'].iloc[-1] * 100
    years = (df.index[-1] - df.index[0]).days / 365.25
    annual_return = (1 + df['Cumulative_Strategy_Return'].iloc[-1]) ** (1/years) - 1
    
    # 최대낙폭 계산
    cumulative = (1 + df['Strategy_Return_with_Fee']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # 샤프비율 계산
    risk_free_rate = 0.02
    strategy_return_std = df['Strategy_Return_with_Fee'].std()
    
    if strategy_return_std == 0 or pd.isna(strategy_return_std):
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (annual_return - risk_free_rate) / strategy_return_std / np.sqrt(252)
    
    # 매매 횟수
    buy_signals = len(df[df['Signal'] == 1])
    sell_signals = len(df[df['Signal'] == -1])
    
    return {
        'ma_period': ma_period,
        'final_return': final_return,
        'annual_return': annual_return * 100,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'total_trades': buy_signals + sell_signals,
        'data': df
    }

def analyze_ma_optimization(data, config, progress_bar):
    """이동평균선 최적화 분석"""
    results = []
    total_tests = config['ma_range_end'] - config['ma_range_start'] + 1
    
    for i, ma_period in enumerate(range(config['ma_range_start'], config['ma_range_end'] + 1)):
        progress = (i + 1) / total_tests
        progress_bar.progress(progress, f"이동평균선 최적화 진행률: {i+1}/{total_tests}")
        
        result = backtest_ma_strategy(data, ma_period, config['selling_fee'])
        results.append(result)
        
    return results

def analyze_stability(data, config, progress_bar):
    """종합 안정성 분석"""
    ma_periods = list(range(config['stability_ma_start'], config['stability_ma_end'] + 1, config['stability_ma_step']))
    analysis_periods = list(range(config['stability_period_start'], config['stability_period_end'] + 1, config['stability_period_step']))
    
    results = []
    total_combinations = len(ma_periods) * len(analysis_periods)
    current_combination = 0
    
    for ma_period in ma_periods:
        for period_days in analysis_periods:
            current_combination += 1
            progress = current_combination / total_combinations
            progress_bar.progress(progress, f"안정성 분석 진행률: {current_combination}/{total_combinations}")
            
            end_date = data.index[-1]
            start_date = end_date - pd.Timedelta(days=period_days)
            period_data = data[data.index >= start_date].copy()
            
            if len(period_data) < ma_period + 10:
                continue
            
            result = backtest_ma_strategy(period_data, ma_period, config['selling_fee'])
            
            if result['total_trades'] < config['min_total_trades']:
                continue
            
            # 안정성 지표 계산
            actual_start_date = period_data.index[0]
            actual_end_date = period_data.index[-1]
            actual_days = len(period_data)
            
            # 시장 수익률 계산
            period_market_return = ((period_data['Close'].iloc[-1] / period_data['Close'].iloc[0]) - 1) * 100
            
            if actual_days >= 365:
                period_market_annual = (((period_data['Close'].iloc[-1] / period_data['Close'].iloc[0]) ** (365.25 / actual_days)) - 1) * 100
            else:
                period_market_annual = period_market_return
            
            # 안정성 점수 계산
            return_score = max(0, min(100, result['annual_return']))
            sharpe_score = max(0, min(100, (result['sharpe_ratio'] + 2) * 25))
            drawdown_score = max(0, min(100, 100 + result['max_drawdown'] * 2))
            
            trades_per_year = result['total_trades'] / (period_days / 365.25)
            ideal_trades = 12
            trade_deviation = abs(trades_per_year - ideal_trades)
            trade_score = max(0, min(100, 100 - trade_deviation * 5))
            
            stability_score = (return_score * 0.3 + sharpe_score * 0.25 + 
                             drawdown_score * 0.25 + trade_score * 0.2)
            
            results.append({
                'ma_period': ma_period,
                'analysis_days': period_days,
                'start_date': actual_start_date.strftime('%Y-%m-%d'),
                'end_date': actual_end_date.strftime('%Y-%m-%d'),
                'actual_days': actual_days,
                'final_return': result['final_return'],
                'annual_return': result['annual_return'],
                'market_return': period_market_return,
                'market_annual_return': period_market_annual,
                'max_drawdown': result['max_drawdown'],
                'sharpe_ratio': result['sharpe_ratio'],
                'total_trades': result['total_trades'],
                'trades_per_year': trades_per_year,
                'stability_score': stability_score
            })
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def create_price_chart(data, ma_period=None):
    """가격 및 이동평균선 차트 생성"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('주가 및 이동평균선', '거래량'),
        row_width=[0.7, 0.3]
    )
    
    # 종가 차트
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='종가', line=dict(color='blue')),
        row=1, col=1
    )
    
    # 이동평균선 차트
    if ma_period and f'MA{ma_period}' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data[f'MA{ma_period}'], name=f'{ma_period}일 이동평균', line=dict(color='red')),
            row=1, col=1
        )
    
    # 거래량 차트
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='거래량', marker_color='gray', opacity=0.5),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="날짜", row=2, col=1)
    fig.update_yaxes(title_text="주가 (원)", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    
    return fig

def create_returns_chart(data):
    """수익률 비교 차트 생성"""
    if 'Cumulative_Strategy_Return' not in data.columns:
        return None
    
    # 시장 수익률 계산
    data['Market_Return'] = (data['Close'] / data['Close'].iloc[0]) - 1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Cumulative_Strategy_Return'] * 100,
        name='전략 수익률',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Market_Return'] * 100,
        name='매수보유 수익률',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="전략 vs 매수보유 수익률 비교",
        xaxis_title="날짜",
        yaxis_title="수익률 (%)",
        height=400
    )
    
    return fig

def create_stability_charts(stability_results):
    """안정성 분석 결과 차트 생성"""
    if stability_results.empty:
        return None, None, None, None
    
    # 1. 안정성 점수 분포 히스토그램
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(
        x=stability_results['stability_score'],
        nbinsx=30,
        name='안정성 점수 분포',
        marker_color='lightblue',
        opacity=0.7
    ))
    fig1.update_layout(
        title="안정성 점수 분포",
        xaxis_title="안정성 점수",
        yaxis_title="빈도",
        height=400
    )
    
    # 2. 이동평균 기간별 평균 안정성 점수
    ma_grouped = stability_results.groupby('ma_period')['stability_score'].mean().reset_index()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ma_grouped['ma_period'],
        y=ma_grouped['stability_score'],
        mode='lines+markers',
        name='평균 안정성 점수',
        line=dict(color='green', width=2),
        marker=dict(size=6)
    ))
    fig2.update_layout(
        title="이동평균 기간별 평균 안정성 점수",
        xaxis_title="이동평균 기간 (일)",
        yaxis_title="평균 안정성 점수",
        height=400
    )
    
    # 3. 분석 기간별 평균 안정성 점수
    period_grouped = stability_results.groupby('analysis_days')['stability_score'].mean().reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=period_grouped['analysis_days'],
        y=period_grouped['stability_score'],
        mode='lines+markers',
        name='평균 안정성 점수',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ))
    fig3.update_layout(
        title="분석 기간별 평균 안정성 점수",
        xaxis_title="분석 기간 (일)",
        yaxis_title="평균 안정성 점수",
        height=400
    )
    
    # 4. 안정성 점수 히트맵 (이동평균 vs 분석기간)
    # 피벗 테이블 생성
    pivot_data = stability_results.pivot_table(
        values='stability_score',
        index='ma_period',
        columns='analysis_days',
        aggfunc='mean'
    )
    
    fig4 = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn',
        text=np.round(pivot_data.values, 1),
        texttemplate="%{text}",
        textfont={"size": 8},
        colorbar=dict(title="안정성 점수")
    ))
    fig4.update_layout(
        title="안정성 점수 히트맵 (이동평균 기간 vs 분석 기간)",
        xaxis_title="분석 기간 (일)",
        yaxis_title="이동평균 기간 (일)",
        height=500
    )
    
    return fig1, fig2, fig3, fig4

def create_performance_scatter(stability_results):
    """성과 지표 산점도 차트 생성"""
    if stability_results.empty:
        return None
    
    # 수익률 vs 위험(최대낙폭) 산점도
    fig = go.Figure()
    
    # 안정성 점수에 따라 색상 구분
    fig.add_trace(go.Scatter(
        x=stability_results['max_drawdown'],
        y=stability_results['annual_return'],
        mode='markers',
        marker=dict(
            color=stability_results['stability_score'],
            colorscale='RdYlGn',
            size=8,
            colorbar=dict(title="안정성 점수"),
            line=dict(width=1, color='black')
        ),
        text=[f"이평: {row['ma_period']}일<br>기간: {row['analysis_days']}일<br>안정성: {row['stability_score']:.1f}" 
              for _, row in stability_results.iterrows()],
        hovertemplate='%{text}<br>연수익률: %{y:.2f}%<br>최대낙폭: %{x:.2f}%<extra></extra>',
        name='전략 조합'
    ))
    
    fig.update_layout(
        title="위험-수익률 분석 (최대낙폭 vs 연평균 수익률)",
        xaxis_title="최대낙폭 (%)",
        yaxis_title="연평균 수익률 (%)",
        height=500
    )
    
    return fig

# 메인 앱
def main():
    st.title("📈 이동평균선 백테스팅 분석기")
    st.markdown("---")
    
    # 종목 리스트 로드
    if st.session_state.stock_list.empty:
        with st.spinner("종목 목록을 불러오는 중..."):
            st.session_state.stock_list = load_stock_list()
    
    # 사이드바 설정
    st.sidebar.header("📋 분석 설정")
    
    # 기본 설정
    st.sidebar.subheader("기본 정보")
    ticker = st.sidebar.text_input("종목 티커", value="005930", help="예: 삼성전자 - 005930")
    
    # 종목명 표시
    if ticker:
        stock_name = get_stock_name(ticker, st.session_state.stock_list)
        st.sidebar.info(f"**종목명:** {stock_name}")
    
    selling_fee = st.sidebar.number_input("매도 수수료 (%)", value=0.20, min_value=0.0, max_value=5.0, step=0.01)
    
    # 기간 설정
    st.sidebar.subheader("분석 기간")
    start_date = st.sidebar.date_input("시작일", value=datetime(2024, 1, 1))
    end_date = st.sidebar.date_input("종료일", value=datetime.now())
    
    # 이동평균선 최적화 설정
    st.sidebar.subheader("이동평균선 최적화")
    enable_ma_optimization = st.sidebar.checkbox("이동평균선 최적화 실행", value=True)
    
    if enable_ma_optimization:
        ma_start = st.sidebar.number_input("시작일수", value=2, min_value=1, max_value=500)
        ma_end = st.sidebar.number_input("종료일수", value=120, min_value=2, max_value=500)
    else:
        ma_start, ma_end = 2, 120
    
    # 안정성 분석 설정
    st.sidebar.subheader("종합 안정성 분석")
    enable_stability_analysis = st.sidebar.checkbox("종합 안정성 분석 실행", value=True)
    
    if enable_stability_analysis:
        min_trades = st.sidebar.number_input("최소 매매횟수", value=9, min_value=1, max_value=100)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            stability_ma_start = st.number_input("이평선 시작", value=5, min_value=1, max_value=500)
            stability_period_start = st.number_input("기간 시작(일)", value=60, min_value=30, max_value=2000)
        with col2:
            stability_ma_end = st.number_input("이평선 종료", value=240, min_value=2, max_value=500)
            stability_period_end = st.number_input("기간 종료(일)", value=720, min_value=60, max_value=2000)
        
        stability_ma_step = st.sidebar.number_input("이평선 간격", value=5, min_value=1, max_value=50)
        stability_period_step = st.sidebar.number_input("기간 간격(일)", value=10, min_value=1, max_value=100)
    else:
        min_trades = 9
        stability_ma_start, stability_ma_end, stability_ma_step = 5, 240, 5
        stability_period_start, stability_period_end, stability_period_step = 60, 720, 10
    
    # 분석 버튼
    if st.sidebar.button("🚀 분석 시작", type="primary"):
        if not ticker:
            st.error("종목 티커를 입력해주세요.")
            return
        
        try:
            # 데이터 불러오기
            with st.spinner("데이터를 불러오는 중..."):
                data = fdr.DataReader(ticker, start_date, end_date)
                
                if data.empty:
                    st.error("데이터를 불러올 수 없습니다. 종목 코드를 확인해주세요.")
                    return
            
            # 기본 정보 계산
            data['Daily_Return'] = data['Close'].pct_change()
            data['Cumulative_Market_Return'] = (1 + data['Daily_Return']).cumprod() - 1
            final_market_return = data['Cumulative_Market_Return'].iloc[-1] * 100
            years = (data.index[-1] - data.index[0]).days / 365.25
            annual_market_return = (1 + data['Cumulative_Market_Return'].iloc[-1]) ** (1/years) - 1
            
            # 분석 설정
            config = {
                'ticker': ticker,
                'selling_fee': selling_fee / 100,
                'start_date': start_date,
                'end_date': end_date,
                'enable_ma_optimization': enable_ma_optimization,
                'ma_range_start': int(ma_start),
                'ma_range_end': int(ma_end),
                'enable_stability_analysis': enable_stability_analysis,
                'min_total_trades': int(min_trades),
                'stability_ma_start': int(stability_ma_start),
                'stability_ma_end': int(stability_ma_end),
                'stability_ma_step': int(stability_ma_step),
                'stability_period_start': int(stability_period_start),
                'stability_period_end': int(stability_period_end),
                'stability_period_step': int(stability_period_step)
            }
            
            # 기본 정보 표시
            st.success("✅ 데이터 로드 완료!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("분석 종목", ticker)
            with col2:
                st.metric("분석 기간", f"{len(data)}일")
            with col3:
                st.metric("매수보유 총수익률", f"{final_market_return:.2f}%")
            with col4:
                st.metric("매수보유 연평균", f"{annual_market_return*100:.2f}%")
            
            # 기본 차트 표시
            st.subheader("📊 주가 차트")
            basic_chart = create_price_chart(data)
            st.plotly_chart(basic_chart, use_container_width=True)
            
            # 이동평균선 최적화
            if enable_ma_optimization:
                st.subheader("🏆 이동평균선 최적화 결과")
                
                progress_bar = st.progress(0, "이동평균선 최적화 준비 중...")
                results = analyze_ma_optimization(data, config, progress_bar)
                progress_bar.empty()
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # 최고 성과 지표
                    best_return = results_df.loc[results_df['final_return'].idxmax()]
                    
                    # 통합 안정성 점수 계산
                    results_df['stability_score'] = (
                        results_df['annual_return'] * 0.4 +
                        results_df['sharpe_ratio'].fillna(0) * 25 * 0.3 +
                        (100 + results_df['max_drawdown'] * 2) * 0.3
                    )
                    top_10 = results_df.nlargest(10, 'stability_score')
                    
                    # 최고 수익률 표시
                    st.info(f"📈 **최고 총수익률:** {best_return['ma_period']:.0f}일 이동평균 : {best_return['final_return']:.2f}%")
                    
                    # 상위 10개 결과 표시
                    st.write("**🎯 종합 안정성 점수 상위 10개**")
                    display_df = top_10[['ma_period', 'final_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'total_trades', 'stability_score']].copy()
                    display_df.columns = ['이평일수', '총수익률(%)', '연평균(%)', '샤프비율', '최대낙폭(%)', '매매횟수', '안정성점수']
                    display_df = display_df.round(2)
                    st.dataframe(display_df, use_container_width=True)
                    
                    # 최적 이동평균선으로 차트 업데이트
                    best_ma = int(best_return['ma_period'])
                    best_result = backtest_ma_strategy(data, best_ma, config['selling_fee'])
                    
                    st.subheader(f"📈 최적 이동평균선({best_ma}일) 분석 결과")
                    
                    # 차트 표시
                    chart_data = best_result['data']
                    price_chart = create_price_chart(chart_data, best_ma)
                    st.plotly_chart(price_chart, use_container_width=True)
                    
                    returns_chart = create_returns_chart(chart_data)
                    if returns_chart:
                        st.plotly_chart(returns_chart, use_container_width=True)
            
            # 종합 안정성 분석
            if enable_stability_analysis:
                st.subheader("🎯 종합 안정성 분석 결과")
                
                progress_bar = st.progress(0, "안정성 분석 준비 중...")
                stability_results = analyze_stability(data, config, progress_bar)
                progress_bar.empty()
                
                if not stability_results.empty:
                    stability_sorted = stability_results.sort_values('stability_score', ascending=False)
                    
                    # 상위 15개 결과 표시
                    st.write("**📊 종합 안정성 점수 순위 (상위 15개)**")
                    st.write("💡 표에서 행을 클릭하면 해당 조합의 상세 차트를 확인할 수 있습니다.")
                    
                    top_15 = stability_sorted.head(15)
                    display_cols = ['ma_period', 'analysis_days', 'actual_days', 'start_date', 'end_date', 
                                  'final_return', 'annual_return', 'market_return', 'sharpe_ratio', 
                                  'max_drawdown', 'stability_score']
                    display_df = top_15[display_cols].copy()
                    display_df.columns = ['이평일수', '분석기간', '실제일수', '시작일', '종료일', 
                                        '전략총수익(%)', '전략연수익(%)', '매수보유총(%)', '샤프비율', 
                                        '최대낙폭(%)', '안정성점수']
                    display_df = display_df.round(2)
                    
                    # 선택 가능한 데이터프레임
                    selected_rows = st.dataframe(
                        display_df, 
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="single-row"
                    )
                    
                    # 선택된 행이 있을 때 상세 차트 표시
                    if selected_rows.selection.rows:
                        selected_idx = selected_rows.selection.rows[0]
                        selected_combo = top_15.iloc[selected_idx]
                        
                        st.subheader(f"📈 선택된 조합 상세 분석 ({selected_combo['ma_period']:.0f}일 이동평균, {selected_combo['analysis_days']:.0f}일 기간)")
                        
                        # 선택된 조합의 기간 데이터 추출
                        end_date_combo = data.index[-1]
                        start_date_combo = end_date_combo - pd.Timedelta(days=selected_combo['analysis_days'])
                        selected_period_data = data[data.index >= start_date_combo].copy()
                        
                        # 선택된 조합으로 백테스팅 실행
                        selected_ma = int(selected_combo['ma_period'])
                        selected_result = backtest_ma_strategy(selected_period_data, selected_ma, config['selling_fee'])
                        
                        # 메트릭 표시
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("이동평균선", f"{selected_combo['ma_period']:.0f}일")
                            st.metric("총수익률", f"{selected_combo['final_return']:.2f}%")
                        with col2:
                            st.metric("분석기간", f"{selected_combo['analysis_days']:.0f}일")
                            st.metric("연평균수익률", f"{selected_combo['annual_return']:.2f}%")
                        with col3:
                            st.metric("샤프비율", f"{selected_combo['sharpe_ratio']:.3f}")
                            st.metric("최대낙폭", f"{selected_combo['max_drawdown']:.2f}%")
                        with col4:
                            st.metric("총 매매횟수", f"{selected_combo['total_trades']:.0f}회")
                            st.metric("안정성점수", f"{selected_combo['stability_score']:.1f}/100")
                        
                        # 초과수익률 계산
                        excess_return = selected_combo['final_return'] - selected_combo['market_return']
                        st.info(f"**🎯 전략 초과수익률:** {excess_return:.2f}%p")
                        
                        # 차트 표시
                        chart_data = selected_result['data']
                        
                        # 가격 및 이동평균선 차트
                        st.write("**📊 주가 및 이동평균선 차트**")
                        price_chart = create_price_chart(chart_data, selected_ma)
                        st.plotly_chart(price_chart, use_container_width=True)
                        
                        # 수익률 비교 차트
                        st.write("**📈 전략 vs 매수보유 수익률 비교**")
                        returns_chart = create_returns_chart(chart_data)
                        if returns_chart:
                            st.plotly_chart(returns_chart, use_container_width=True)
                    
                    # 안정성 분석 시각화
                    st.subheader("📈 안정성 분석 시각화")
                    
                    # 차트 생성
                    hist_chart, ma_chart, period_chart, heatmap_chart = create_stability_charts(stability_results)
                    scatter_chart = create_performance_scatter(stability_results)
                    
                    # 차트를 탭으로 구분하여 표시
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 점수 분포", "📈 이평기간별", "📅 분석기간별", "🔥 히트맵", "💎 위험-수익률"])
                    
                    with tab1:
                        if hist_chart:
                            st.plotly_chart(hist_chart, use_container_width=True)
                            st.info("안정성 점수의 전체적인 분포를 확인할 수 있습니다. 높은 점수일수록 안정적인 전략입니다.")
                    
                    with tab2:
                        if ma_chart:
                            st.plotly_chart(ma_chart, use_container_width=True)
                            st.info("이동평균 기간별로 평균 안정성 점수를 보여줍니다. 최적의 이동평균 기간을 찾는데 도움이 됩니다.")
                    
                    with tab3:
                        if period_chart:
                            st.plotly_chart(period_chart, use_container_width=True)
                            st.info("분석 기간별로 평균 안정성 점수를 보여줍니다. 어떤 기간에서 전략이 더 효과적인지 확인할 수 있습니다.")
                    
                    with tab4:
                        if heatmap_chart:
                            st.plotly_chart(heatmap_chart, use_container_width=True)
                            st.info("이동평균 기간과 분석 기간의 조합별 안정성 점수를 한눈에 볼 수 있습니다. 진한 녹색일수록 높은 점수입니다.")
                    
                    with tab5:
                        if scatter_chart:
                            st.plotly_chart(scatter_chart, use_container_width=True)
                            st.info("위험(최대낙폭) 대비 수익률을 보여주는 산점도입니다. 좌상단에 위치할수록 저위험 고수익 전략입니다.")
                    
                    # 최고 안정성 조합 상세 분석
                    best_combo = stability_sorted.iloc[0]
                    st.subheader("🏆 최고 안정성 조합 상세 분석")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("이동평균선", f"{best_combo['ma_period']:.0f}일")
                        st.metric("전략 총수익률", f"{best_combo['final_return']:.2f}%")
                    with col2:
                        st.metric("분석기간", f"{best_combo['analysis_days']:.0f}일")
                        st.metric("전략 연평균", f"{best_combo['annual_return']:.2f}%")
                    with col3:
                        st.metric("샤프비율", f"{best_combo['sharpe_ratio']:.3f}")
                        st.metric("최대낙폭", f"{best_combo['max_drawdown']:.2f}%")
                    with col4:
                        st.metric("총 매매횟수", f"{best_combo['total_trades']:.0f}회")
                        st.metric("안정성점수", f"{best_combo['stability_score']:.1f}/100")
                    
                    # 초과수익률 계산
                    excess_return = best_combo['final_return'] - best_combo['market_return']
                    st.info(f"**🎯 전략 초과수익률:** {excess_return:.2f}%p")
                    
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
    
    # 도움말
    with st.expander("📘 사용 방법 안내"):
        st.markdown("""
        ### 이동평균선 백테스팅 분석기 사용법
        
        **1. 기본 설정**
        - 종목 티커: 분석하고 싶은 주식의 종목 코드 입력 (예: 삼성전자 - 005930)
        - 매도 수수료: 매도 시 발생하는 수수료 및 세금의 총합 (%)
        - 분석 기간: 백테스팅을 수행할 시작일과 종료일 설정
        
        **2. 이동평균선 최적화**
        - 지정된 기간에서 가장 수익성이 좋은 이동평균선을 찾는 기능
        - 시작일수/종료일수: 테스트할 이동평균선의 범위 설정
        
        **3. 종합 안정성 분석**
        - 다양한 시장 상황에서 전략의 안정성을 검증하는 기능
        - 여러 이동평균선과 분석기간 조합을 테스트하여 가장 안정적인 조합 탐색
        
        **4. 결과 해석**
        - 안정성 점수: 수익률, 샤프비율, 최대낙폭, 매매빈도를 종합한 점수
        - 높은 점수일수록 안정적이고 지속가능한 전략을 의미
        """)

if __name__ == "__main__":
    main()
