import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← 여기 수정!
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
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
# 스타일 적용
# =====================
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
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
except KeyError:
    st.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다. Streamlit Secrets를 확인하세요.")
    st.stop()

# =====================
# 세션 상태 초기화
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# =====================
# RAG 체인 초기화 함수
# =====================
@st.cache_resource
def initialize_rag_chain():
    """PDF 로드 및 RAG 체인 구성"""
    
    # PDF 파일 경로
    pdf_path = "test.pdf"
    
    if not os.path.exists(pdf_path):
        st.error(f"⚠️ {pdf_path} 파일을 찾을 수 없습니다.")
        return None
    
    with st.spinner("📄 PDF 문서를 분석 중입니다..."):
        # 1. PDF 로드
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # 2. 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        
        # 3. 임베딩 생성
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GEMINI_API_KEY
        )
        
        # 4. 벡터 스토어 생성
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # 5. LLM 설정
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        # 6. 대화형 검색 체인 생성
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=st.session_state.memory,
            return_source_documents=True,
            output_key="answer"
        )
        
    return chain

# =====================
# RAG 체인 로드
# =====================
if st.session_state.chain is None:
    st.session_state.chain = initialize_rag_chain()

# 체인 로드 실패 시 중단
if st.session_state.chain is None:
    st.stop()

# 로드 성공 표시
st.success("✅ PDF 문서가 로드되었습니다. 질문을 입력하세요!")

# =====================
# 사이드바
# =====================
with st.sidebar:
    st.header("ℹ️ 사용 방법")
    st.markdown("""
    1. 아래 채팅창에 질문을 입력하세요
    2. PDF 문서 내용을 기반으로 답변합니다
    3. 대화 기록이 유지됩니다
    """)
    
    st.divider()
    
    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
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
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            try:
                response = st.session_state.chain.invoke({"question": prompt})
                answer = response["answer"]
                
                # 소스 문서 정보 추가 (선택적)
                if response.get("source_documents"):
                    sources = response["source_documents"]
                    unique_pages = set([doc.metadata.get("page", 0) + 1 for doc in sources])
                    source_info = f"\n\n---\n📖 *참조 페이지: {', '.join(map(str, sorted(unique_pages)))}*"
                    answer += source_info
                
                st.markdown(answer)
                
                # 응답 저장
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"⚠️ 오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
