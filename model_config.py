import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from message_reducer import merge_messages
from typing import Any, Dict, Optional
from pydantic import Field, BaseModel, validator
from datetime import datetime
import dotenv
import httpx    
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

dotenv.load_dotenv()

# TaskState 모델 정의
class TaskState(BaseModel):
    """프론트/백엔드 공용 task 구조 (단일 과제)"""
    month: str = Field(..., description="월(예: '01', '12')")
    date: str = Field(..., description="YYYY-MM-DD 형식의 날짜")
    task_no: int = Field(..., description="해당 날짜 내 순번 (1부터 시작)")
    start_pg: int = Field(..., description="시작 페이지 번호")
    end_pg: int = Field(..., description="끝 페이지 번호")
    title: str = Field(..., description="학습 주제")
    summary: str = Field(..., description="간단 요약")
    is_completed: bool = Field(default=False, description="완료 여부")

    @validator("date")
    def validate_date(cls, v):
        # YYYY-MM-DD 검증
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("date 필드는 YYYY-MM-DD 형식이어야 합니다")

# Task list reducer function
def merge_task_lists(existing: List[TaskState], new: List[TaskState]) -> List[TaskState]:
    """
    Task list를 병합하는 reducer 함수.
    새로운 task_list가 들어오면 기존 것을 완전히 대체합니다.
    """
    return new

# Shared State Schema for all agents
class MultiAgentState(AgentState):
    messages: Annotated[List[BaseMessage], merge_messages]
    task_list: Annotated[List[TaskState], merge_task_lists] = []  # 세션별 task 관리를 위한 필드
    professor_type: str = "T형"  # 교수자 타입 (T형 또는 F형)
    session_id: str = "default"  # 세션 ID

# Model Configurations
Model_gpt4_1 = ChatOpenAI(
    model="gpt-4.1-2025-04-14",
    max_tokens=10000,
    max_retries=10,
)