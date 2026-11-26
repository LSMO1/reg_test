import streamlit as st
import tiktoken
import os
import tempfile

from loguru import logger

from langchain_classic.chains import ConversationalRetrievalChain

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI


def main():

    st.set_page_config(
        page_title="Streamlit_Rag",
        page_icon=":books:"
    )

    st.title("_Private Data :red[Q/A Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # 파일 업로드 및 API Key 입력
    with st.sidebar:

        uploaded_files = st.file_uploader(
            "Upload your file",
            type=['pdf', 'docx', 'pptx'],
            accept_multiple_files=True
        )

        google_api_key = st.text_input(
            "Google API Key",
            key="chatbot_api_key",
            type="password"
        )

        process = st.button("Process")

    # 업로드한 문서 처리
    if process:

        if not google_api_key:
            st.info("Please add your Google API key to continue.")
            st.stop()

        if not uploaded_files:
            st.info("Please upload your files.")
            st.stop()

        st.session_state.processComplete = False
        st.session_state.conversation = None

        with st.spinner("Processing documents..."):

            files_text = get_text(uploaded_files)

            if not files_text:
                st.warning("문서에서 텍스트를 추출하지 못했습니다.")
                st.stop()

            text_chunks = get_text_chunks(files_text)

            if not text_chunks:
                st.warning("처리할 텍스트가 없습니다.")
                st.stop()

            vetorestore = get_vectorstore(text_chunks)

            st.session_state.conversation = get_conversation_chain(
                vetorestore,
                google_api_key
            )

            st.session_state.processComplete = True
            st.session_state.chat_history = None

            # 새로운 문서 처리 시 기존 대화 초기화
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"
                }
            ]

        st.sidebar.success("Processing Complete!")

    # 초기 채팅 메시지 설정
    if 'messages' not in st.session_state:

        st.session_state['messages'] = [
            {
                "role": "assistant",
                "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"
            }
        ]

    # 기존 채팅 기록 출력
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input(
        "질문을 입력해주세요.",
        disabled=not st.session_state.processComplete
    ):

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            chain = st.session_state.conversation

            with st.spinner("Thinking..."):

                result = chain.invoke({
                    "question": query
                })

                st.session_state.chat_history = result['chat_history']

                response = result['answer']
                source_documents = result['source_documents']

            st.markdown(response)

            # 참고 문서 출력
            with st.expander("참고 문서 확인"):

                for source_document in source_documents[:3]:

                    st.markdown(
                        source_document.metadata.get(
                            'source',
                            'Unknown'
                        ),
                        help=source_document.page_content
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )


# 텍스트 토큰 개수 계산
def tiktoken_len(text):

    tokenizer = tiktoken.get_encoding("cl100k_base")

    tokens = tokenizer.encode(text)

    return len(tokens)


# 업로드한 문서에서 텍스트 추출
def get_text(docs):

    doc_list = []

    for doc in docs:

        file_name = doc.name
        extension = os.path.splitext(file_name)[1].lower()

        # 업로드된 파일을 임시 디렉터리에 저장
        with tempfile.TemporaryDirectory() as temp_dir:

            file_path = os.path.join(
                temp_dir,
                os.path.basename(file_name)
            )

            with open(file_path, "wb") as file:
                file.write(doc.getvalue())

            logger.info(f"Uploaded {file_name}")

            # PDF 파일 처리
            if extension == '.pdf':

                loader = PyPDFLoader(file_path)
                documents = loader.load()

            # DOCX 파일 처리
            elif extension == '.docx':

                loader = Docx2txtLoader(file_path)
                documents = loader.load()

            # PPTX 파일 처리
            elif extension == '.pptx':

                loader = UnstructuredPowerPointLoader(file_path)
                documents = loader.load()

            else:
                continue

            # 참고 문서에는 원본 파일명 저장
            for document in documents:
                document.metadata['source'] = file_name

            doc_list.extend(documents)

    return doc_list


# 추출한 텍스트를 Chunk 단위로 분리
def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )

    chunks = text_splitter.split_documents(text)

    return chunks


# 임베딩 생성 및 FAISS Vector Store 구축
def get_vectorstore(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={
            'device': 'cpu'
        },
        encode_kwargs={
            'normalize_embeddings': True
        }
    )

    vectordb = FAISS.from_documents(
        text_chunks,
        embeddings
    )

    return vectordb


# Gemini LLM 및 대화 검색 체인 생성
def get_conversation_chain(vetorestore, google_api_key):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=google_api_key,
        temperature=0
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(

        llm=llm,

        chain_type="stuff",

        retriever=vetorestore.as_retriever(
            search_type='mmr'
        ),

        memory=ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        ),

        return_source_documents=True,

        verbose=True
    )

    return conversation_chain


if __name__ == '__main__':
    main()