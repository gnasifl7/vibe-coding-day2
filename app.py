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
            text += "\n[페이지 " + str(i+1) + "]\n" + page_text + "\n"
    return text

# =====================
# 시스템 프롬프트 생성
# =====================
def get_system_prompt(pdf_text):
    return (
        "당신은 PDF 문서 기반 질문 답변 전문가입니다.\n\n"
        "아래 문서 내용을 기반으로 사용자 질문에 정확하게 답변하세요.\n"
        "문서에 없는 내용은 '문서에서 해당 내용을 찾을 수 없습니다'라고 답변하세요.\n"
        "답변 마지막에 참조한 페이지 번호를 알려주세요.\n\n"
        "=== 문서 내용 ===\n" + pdf_text
    )

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
        system_instruction=get_system_prompt(pdf_text)
    )
    st.session_state.chat = model.start_chat(history=[])

# =====================
# 사이드바
# =====================
with st.sidebar:
    st.header("ℹ️ 사용 방법")
    st.markdown("""
    1. 채팅창에 질문 입력
    2. PDF 내용 기반 답변 제공
    3. 대화 맥락 유지
    """)
    
    st.divider()
    
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=get_system_prompt(pdf_text)
        )
        st.session_state.chat = model.start_chat(history=[])
        st.rerun()
    
    st.divider()
    st.caption("Powered by Gemini 2.5 Flash")

# =====================
# 채팅 기록 표시
# =====================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================
# 사용자 입력 처리
# =====================
if prompt := st.chat_input("PDF 문서에 대해 질문하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                response = st.session_state.chat.send_message(prompt)
                answer = response.text
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = "⚠️ 오류: " + str(e)
                st.error(error_msg)
