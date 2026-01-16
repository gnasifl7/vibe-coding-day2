import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

# =====================
# 페이지 설정
# =====================
st.set_page_config(
    page_title="PDF 챗봇",
    page_icon="📚",
    layout="centered"
)

# =====================
# 스타일
# =====================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# 헤더
# =====================
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📚 PDF 문서 챗봇")
st.caption("test.pdf 문서를 기반으로 질문에 답변합니다")
st.markdown('</div>', unsafe_allow_html=True)

# =====================
# API 키 설정
# =====================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    st.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")
    st.stop()

# =====================
# PDF 텍스트 추출
# =====================
@st.cache_data
def extract_pdf_text(pdf_path):
    if not os.path.exists(pdf_path):
        return None
    
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n[페이지 {i+1}]\n{page_text}\n"
    return text

# =====================
# PDF 로드
# =====================
pdf_text = extract_pdf_text("test.pdf")

if pdf_text is None:
    st.error("⚠️ test.pdf 파일을 찾을 수 없습니다.")
    st.stop()

st.success("✅ PDF 문서가 로드되었습니다. 질문을 입력하세요!")

# =====================
# 세션 상태 초기화
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=f"""당신은 PDF 문서 기반 질문 답변 전문가입니다.

아래 문서 내용을 기반으로 사용자 질문에 정확하게 답변하세요.
문서에 없는 내용은 "문서에서 해당 내용을 찾을 수 없습니다"라고 답변하세요.
답변 마지막에 참조한 페이지 번호를 알
