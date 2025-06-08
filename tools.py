# tools.py
import chromadb
from chromadb.config import Settings
from typing import Literal, Dict, List, Optional, Any
import logging
import json
import os
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId
from typing import Annotated
from langgraph.types import Command   
import sqlite3
import pandas as pd
from datetime import datetime
from pydantic import BaseModel, Field, validator

# 환경 변수 로드
load_dotenv(override=True)

# --- 로깅 설정 ---
logger = logging.getLogger(__name__)

# ChromaDB 클라이언트 캐시 (pdf_processor와 공유)
try:
    from pdf_processor import _chroma_clients
except ImportError:
    # pdf_processor를 import할 수 없는 경우 자체 캐시 생성
    _chroma_clients = {}

@tool
async def get_textbook_content(
    mode: Literal["info", "content"],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
):
    """
    교재 정보 조회 및 내용 열람 도구
    
    Args:
        mode: 'info' (교재 정보) 또는 'content' (내용 조회)
        start_page: 시작 페이지 (content 모드에서 사용)
        end_page: 끝 페이지 (content 모드에서 사용)
        
    사용법 가이드:
    1. 첫 번째: info 모드로 교재 전체 분량과 기본 정보를 파악
       - get_textbook_content(mode="info")
       
    2. 두 번째: content 모드로 1~5페이지 정도 확인하여 목차 구조 파악
       - get_textbook_content(mode="content", start_page=1, end_page=5)
       
    3. 세 번째: 목차 정보를 바탕으로 필요한 범위를 여러 번에 걸쳐 조회
       - 한 번에 최대 20페이지까지만 조회하여 context 과부하 방지
       - get_textbook_content(mode="content", start_page=10, end_page=25)
       
    주의사항:
    - content 모드에서는 적절한 페이지 범위 설정 (최대 20페이지 권장)
    - 너무 많은 내용을 한 번에 조회하면 응답 품질이 저하될 수 있음
    """
    try:
        session_id = state.get("session_id", "default")

        def _get_collection(sid: str):
            """세션별 textbook 컬렉션 핸들 얻기 (캐시 활용)"""
            textbook_path = f"DB/textbook/textbook_{sid}"

            # 캐시
            if textbook_path not in _chroma_clients:
                import chromadb.config
                settings = chromadb.config.Settings(
                    persist_directory=textbook_path,
                    anonymized_telemetry=False,
                )
                _chroma_clients[textbook_path] = chromadb.PersistentClient(settings=settings)
                logger.debug(f"ChromaDB 클라이언트 생성 및 캐시: {textbook_path}")

            return _chroma_clients[textbook_path].get_collection("textbook")

        collection = _get_collection(session_id)
        
        if mode == "info":
            meta = collection.metadata or {}
            title = meta.get("title", meta.get("filename", "Unknown"))
            n_page = meta.get("n_page", collection.count())
            contents_index = meta.get("contents_index", "")

            info_text = f"**📚 교재 정보**\n\n"
            info_text += f"- **교재명:** {title}\n"
            info_text += f"- **총 페이지:** {n_page}페이지\n"
            if contents_index:
                info_text += "\n**목차**\n" + contents_index

            return ToolMessage(
                content=info_text,
            tool_call_id=tool_call_id,
                name="get_textbook_content"
                )
        
        elif mode == "content":
            # 페이지 범위 검증
            if start_page is None or end_page is None:
                return ToolMessage(
                    content="content 모드에서는 start_page와 end_page를 모두 지정해야 합니다.",
            tool_call_id=tool_call_id,
                    name="get_textbook_content"
                    )
        
            if end_page - start_page + 1 > 20:
                return ToolMessage(
                    content="한 번에 최대 20페이지까지만 조회할 수 있습니다. 페이지 범위를 줄여주세요.",
                    tool_call_id=tool_call_id,
                    name="get_textbook_content"
                )
            
            # 페이지 범위로 내용 조회 (개별 페이지별로 가져오기)
            documents = []
            metadatas = []
            
            for page_num in range(start_page, end_page + 1):
                try:
                    page_results = collection.get(
                        ids=[f"page_{page_num}"],
                        include=["documents", "metadatas"]
                    )
                    if page_results["documents"]:
                        documents.extend(page_results["documents"])
                        metadatas.extend(page_results["metadatas"])
                except Exception as e:
                    logger.warning(f"페이지 {page_num} 조회 실패: {e}")
                    continue
            
            if not documents:
                return ToolMessage(
                    content=f"페이지 {start_page}-{end_page} 범위에서 내용을 찾을 수 없습니다.",
                    tool_call_id=tool_call_id,
                    name="get_textbook_content"
                )
            
            # 페이지별로 정렬하여 내용 정리
            page_contents = {}
            for doc, metadata in zip(documents, metadatas):
                page = metadata.get("page", 0)
                if page not in page_contents:
                    page_contents[page] = []
                page_contents[page].append(doc)
            
            # 결과 포맷팅
            content_text = f"**📖 페이지 {start_page}-{end_page} 내용**\n\n"
            
            for page in sorted(page_contents.keys()):
                content_text += f"**=== 페이지 {page} ===**\n"
                for i, content in enumerate(page_contents[page], 1):
                    content_text += f"{i}. {content}\n\n"
                content_text += "\n"
            
            return ToolMessage(
                content=content_text,
                tool_call_id=tool_call_id,
                name="get_textbook_content"
            )
            
        else:
            return ToolMessage(
                content="mode는 'info' 또는 'content'만 사용할 수 있습니다.",
            tool_call_id=tool_call_id,
            name="get_textbook_content"
        )
           
    except Exception as e:
        logger.error(f"Textbook content retrieval error: {e}")
        return ToolMessage(
            content=f"교재 조회 중 오류 발생: {e}\n(ChromaDB가 올바르게 설정되어 있는지 확인해주세요)",
            tool_call_id=tool_call_id,
            name="get_textbook_content"
        )
        
# -----------------------------
# 🔄  상태 기반 Task 관리  (State에 직접 저장)
# -----------------------------

from model_config import TaskState, FeedbackState

# ───────────────────────────────────────────────
# State 전용 Task 도구 (통합)
# ───────────────────────────────────────────────

@tool
async def update_task_list(
    final_task_list: List[TaskState],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """
    전체 task_list를 최종 상태로 업데이트합니다.
    - Task_no는 1부터 시작하는 Task 순서 번호. 날마다 1로 초기화
    - 하루의 학습 분량을 여러 개의 Task로 쪼개어 생성
    - title : 단원명, 주제명
    - page_range : 페이지 범위
    - summary : Task의 주요 내용을 서술형으로 요약
    
    Args:
        final_task_list: 업데이트할 최종 task 목록
        state: 현재 상태
        tool_call_id: 도구 호출 ID
    """
    from langgraph.types import Command
    
    #logger.info(f"final_task_list: {final_task_list}")
    
    # TaskState 객체들을 검증하고 정리
    validated_tasks = []
    for task in final_task_list:
        if isinstance(task, TaskState):
            validated_tasks.append(task)
            #logger.info(f"Validated task: {task}")
        elif isinstance(task, dict):
            try:
                validated_tasks.append(TaskState(**task))
            except Exception as e:
                logger.error(f"Invalid task data: {task}, error: {e}")
        else:
            logger.error(f"Unknown task type: {type(task)}")

    # Command를 통해 전체 task_list 교체 (for 루프 종료 후 한 번만 반환)
    return Command(
        update={
        "task_list": validated_tasks,
        "messages": [ToolMessage(content="Task list updated", tool_call_id=tool_call_id, name="update_task_list")]
            }
        )

@tool
async def update_feedback_list(
    final_feedback_list: List[FeedbackState],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
):
    """전체 feedback_list를 최종 상태로 업데이트합니다.
    
    Args:
        final_feedback_list: 날짜별 성찰 피드백의 완전한 목록
    
    Returns:
        Command: feedback_list를 업데이트하는 명령
    """
    from langgraph.types import Command
    
    #logger.info(f"final_feedback_list: {final_feedback_list}")
    
    # FeedbackState 객체들을 검증하고 정리
    validated_feedbacks = []
    for feedback in final_feedback_list:
        if isinstance(feedback, FeedbackState):
            validated_feedbacks.append(feedback)
            #logger.info(f"Validated feedback: {feedback}")
        elif isinstance(feedback, dict):
            try:
                validated_feedbacks.append(FeedbackState(**feedback))
            except Exception as e:
                logger.error(f"Invalid feedback data: {feedback}, error: {e}")
        else:
            logger.error(f"Unknown feedback type: {type(feedback)}")

    return Command(
        update={
        "feedback_list": validated_feedbacks,
        "messages": [ToolMessage(content="Feedback list updated", tool_call_id=tool_call_id, name="update_feedback_list")]
            }
        )