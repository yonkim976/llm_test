import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
import os

# Streamlit 페이지 설정
st.set_page_config(page_title="기업 문서 분석 시스템", layout="wide")
st.title("기업 문서 분석 시스템")

# OpenAI API 키 입력 섹션
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state['OPENAI_API_KEY'] = ''

api_key = st.sidebar.text_input("OpenAI API 키를 입력하세요:", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.session_state['OPENAI_API_KEY'] = api_key

if st.session_state['OPENAI_API_KEY']:
    # embeddings 객체 생성
    embeddings = OpenAIEmbeddings()
    
    # 벡터 저장소 로드
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = FAISS.load_local(
            './db/faiss', 
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("저장된 데이터베이스를 불러왔습니다!")

    # FAISS 리트리버 생성
    retriever = st.session_state['vectorstore'].as_retriever(
        search_kwargs={"k": 3}
    )

    # 채팅 인터페이스
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # 이전 대화 표시
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 프롬프트 템플릿
    prompt = PromptTemplate.from_template(
        """당신은 기업 분석의 전문가입니다.
        제가 질문하는 내용에 알맞은 내용을 깊게 생각한 후 정확하게 답변해주고,
        만약 모른다면 "모르겠습니다"라고 답변해주세요.
        
        이전 대화 내역:
        {chat_history}
        
        현재 질문: {question}
        
        관련 문서 내용: {context}
        
        답변:"""
    )

    # 채팅 기록 포맷팅
    def format_chat_history(messages):
        formatted = ""
        for msg in messages:
            formatted += f"{msg['role']}: {msg['content']}\n"
        return formatted

    # 사용자 입력 처리
    if question := st.chat_input("질문을 입력하세요"):
        st.session_state['messages'].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 채팅 기록 가져오기
        chat_history = format_chat_history(st.session_state['messages'][:-1])

        # 답변 생성
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)
        
        # 검색된 문서들을 문자열로 변환하는 함수
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 관련 문서 검색 결과 표시 (디버깅용)
        with st.expander("검색된 관련 문서"):
            relevant_docs = retriever.get_relevant_documents(question)
            for i, doc in enumerate(relevant_docs, 1):
                st.write(f"문서 {i}:")
                st.write(doc.page_content)
                st.write("---")

        chain = (
            {
                "context": lambda x: format_docs(retriever.get_relevant_documents(x)),
                "question": RunnablePassthrough(),
                "chat_history": lambda x: chat_history
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            response = chain.invoke(question)
            st.markdown(response)
            st.session_state['messages'].append({"role": "assistant", "content": response})

else:
    st.warning("OpenAI API 키를 입력해주세요.")