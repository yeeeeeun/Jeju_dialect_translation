import streamlit as st
from transformers import pipeline, PreTrainedTokenizerFast, BartForConditionalGeneration

# Streamlit 웹 애플리케이션 제목
st.title("Dialect Translation")

# 언어 선택 드롭다운
languages = {"Translating Jeju dialect into a standard language": "제주어"}
selected_language = st.selectbox("Select Language", list(languages.keys()))


# 사용자 입력을 위한 텍스트 박스
st.markdown("<h3 style='font-size:20px;'>Enter text to jeju dialect:</h3>", unsafe_allow_html=True)
user_input = st.text_area("", key="user_input")

# 모델과 토크나이저 로드
model_dir = 'C:/Users/82107/PycharmProjects/ImageProcessing/NLP/pretrained-20240617T080151Z-001/pretrained/checkpoint-500'
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
model = BartForConditionalGeneration.from_pretrained(model_dir)

# 파이프라인 생성
nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# 분석 버튼
if st.button("Compute"):
    if user_input:
        # 번역 수행
        result = nlp(user_input)
        translated_text = result[0]['generated_text']

        # 결과 출력
        st.markdown("<h3 style='font-size:20px;'>Translated Text:</h3>", unsafe_allow_html=True)
        st.text_area("", translated_text, key="translated_text")
    else:
        st.write("Please enter text to analyze.")
