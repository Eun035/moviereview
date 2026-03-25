import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from kiwipiepy import Kiwi
from transformers import pipeline

# ==========================================
# 1. 페이지 및 상태 설정
# ==========================================
st.set_page_config(page_title="영화 리뷰 감성 분석기", page_icon="🍿", layout="wide")

# 예시 버튼 클릭 시 텍스트 영역을 업데이트하기 위한 세션 상태 관리
if 'review_text' not in st.session_state:
    st.session_state.review_text = ""

def set_review(text):
    st.session_state.review_text = text

# ==========================================
# 2. 핵심 리소스 로드 (캐싱하여 속도 최적화)
# ==========================================
@st.cache_resource
def load_tools():
    # 1. 형태소 분석기
    kiwi = Kiwi()
    
    # 2. 기존 학습된 전통 ML 모델 로드
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        lr_model = joblib.load('lr_model.pkl')
    except Exception as e:
        st.error("⚠️ 'tfidf_vectorizer.pkl' 또는 'lr_model.pkl' 파일을 찾을 수 없습니다. 먼저 모델을 저장해주세요.")
        st.stop()
        
    # 3. HuggingFace 사전학습 모델 로드 (한국어 감성 분석 특화 모델)
    # 첫 실행 시 모델을 다운로드하므로 약간의 시간이 소요될 수 있습니다.
    hf_pipeline = pipeline(
        "text-classification", 
        model="jaehyeong/koelectra-base-v3-generalized-sentiment-analysis"
    )
    
    return kiwi, vectorizer, lr_model, hf_pipeline

kiwi, vectorizer, lr_model, hf_pipeline = load_tools()

# 감성 분석용 전처리 함수
def preprocess_text(text):
    tokens = kiwi.tokenize(text)
    return " ".join([token.form for token in tokens if token.tag in ['NNG', 'NNP', 'VV', 'VA', 'MAG']])

# ==========================================
# 3. UI 및 레이아웃 구성
# ==========================================
st.title("🍿 AI 영화 리뷰 감성 분석기")
st.markdown("전통적인 머신러닝(TF-IDF)과 최신 딥러닝(HuggingFace) 모델의 결과를 비교해 보세요!")

# 좌측/우측 단 나누기
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 리뷰 입력")
    # 텍스트 입력 (세션 상태와 연동)
    user_input = st.text_area(
        "영화 리뷰를 자유롭게 작성해 주세요.", 
        value=st.session_state.review_text,
        height=150,
        key="review_text"
    )
    
    # 모델 선택 라디오 버튼
    model_choice = st.radio(
        "분석에 사용할 AI 모델을 선택하세요:",
        ("전통 머신러닝 (TF-IDF + Logistic Regression)", "최신 딥러닝 (HuggingFace KoELECTRA)")
    )
    
    analyze_btn = st.button("🚀 분석 시작", use_container_width=True, type="primary")

    # 예시 버튼 UI
    st.markdown("---")
    st.markdown("💡 **예시 리뷰 클릭해보기**")
    examples = [
        "배우들 연기는 좋은데 스토리가 너무 지루해서 중간에 잤어요.",
        "올해 본 영화 중 단연 최고! 연출과 음악 모두 완벽합니다. 강추!",
        "시간 때우기용으로는 나쁘지 않은데 굳이 극장에서 볼 필요는...",
        "진짜 돈 아깝다. 내 2시간 돌려내라 최악의 영화.",
        "처음엔 뻔한 스토리인 줄 알았는데 후반부 반전이 미쳤네요. 꿀잼!"
    ]
    for ex in examples:
        st.button(ex, on_click=set_review, args=(ex,))

# ==========================================
# 4. 분석 로직 및 시각화 (우측 단)
# ==========================================
with col2:
    if analyze_btn and user_input.strip():
        st.subheader("📊 분석 결과")
        
        with st.spinner('AI가 리뷰의 감성을 분석하고 있습니다...'):
            is_positive = False
            confidence = 0.0
            
            # --- 1) 모델별 추론 로직 ---
            if "TF-IDF" in model_choice:
                # 전처리 및 추론
                processed_text = preprocess_text(user_input)
                vec_input = vectorizer.transform([processed_text])
                
                # 예측 및 확률 계산
                prediction = lr_model.predict(vec_input)[0] # 1(긍정) or 0(부정)
                probabilities = lr_model.predict_proba(vec_input)[0]
                
                is_positive = bool(prediction == 1)
                confidence = probabilities[1] if is_positive else probabilities[0]
                
            else:
                # HuggingFace 추론
                hf_result = hf_pipeline(user_input)[0]
                label = hf_result['label']
                confidence = hf_result['score']
                is_positive = True if label == '1' else False
                
            # --- 2) 게이지 차트(Gauge Chart) 시각화 ---
            sentiment_text = "긍정 😃" if is_positive else "부정 😡"
            color = "green" if is_positive else "red"
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                title = {'text': f"<b>{sentiment_text}</b> (신뢰도)", 'font': {'size': 24, 'color': color}},
                number = {'suffix': "%", 'font': {'size': 40, 'color': color}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gainsboro"},
                        {'range': [80, 100], 'color': "whitesmoke"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # --- 3) 판단 근거 시각화 (TF-IDF 선택 시에만) ---
            if "TF-IDF" in model_choice:
                st.markdown("#### 🔍 AI의 판단 근거 (영향을 준 단어)")
                
                # 입력된 문장의 단어 벡터와 모델 가중치를 곱하여 기여도 산출
                feature_names = vectorizer.get_feature_names_out()
                coef = lr_model.coef_[0]
                
                contributions = []
                # vec_input은 희소 행렬(Sparse Matrix)이므로 0이 아닌 값만 순회
                for col in vec_input.nonzero()[1]:
                    word = feature_names[col]
                    tfidf_val = vec_input[0, col]
                    weight = coef[col]
                    score = tfidf_val * weight
                    contributions.append({'단어': word, '기여도': score})
                
                if contributions:
                    df_contrib = pd.DataFrame(contributions)
                    # 절대값 기준으로 정렬하여 상위 10개 추출
                    df_contrib['Abs'] = df_contrib['기여도'].abs()
                    df_contrib = df_contrib.sort_values(by='Abs', ascending=False).head(10)
                    
                    # 색상 지정 (긍정은 초록, 부정은 빨강)
                    df_contrib['색상'] = df_contrib['기여도'].apply(lambda x: '#2ca02c' if x > 0 else '#d62728')
                    
                    fig_bar = px.bar(
                        df_contrib, 
                        x='기여도', 
                        y='단어', 
                        orientation='h',
                        color='색상',
                        color_discrete_map="identity"
                    )
                    fig_bar.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
                    # 0을 기준으로 선 긋기
                    fig_bar.add_vline(x=0, line_width=1, line_color="black")
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("모델이 학습한 단어장에 이 문장의 핵심 단어가 포함되어 있지 않아 근거를 추출할 수 없습니다.")
                    