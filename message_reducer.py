from __future__ import annotations

import uuid
import logging
from typing import List, Optional, Literal, Dict, Any, cast, Union

from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import convert_to_messages, message_chunk_to_message
from langgraph.graph.message import RemoveMessage, REMOVE_ALL_MESSAGES

# 로그 설정
logger = logging.getLogger(__name__)

Messages = List[BaseMessage]

def merge_messages(
    left: Messages | BaseMessage | dict,
    right: Messages | BaseMessage | dict,
    *,
    format: Optional[Literal["langchain-openai"]] = None,
    strip_function_call: bool = True,
) -> Messages:
    """
    모든 LLM 제공업체(OpenAI, Gemini, Claude)의 메시지를 안전하게 병합하는 리듀서.
    
    * list, 단일 메시지, dict 모두 허용
    * ID 없으면 UUID 자동 부여
    * 다양한 도구 호출 형식 지원 (OpenAI, Gemini, Claude)
    * RemoveMessage(id="*") 지원 — 기존 메시지 삭제/전체 삭제
    * OpenAI `function_call` 필드 안전 처리(option)
    * `format="langchain-openai"` 로 BaseMessage → OpenAI 포맷 변환
    """
    # ---- 1. 리스트 & BaseMessage 로 강제 변환 ------------------------------
    def _to_msg_list(x) -> Messages:
        if not isinstance(x, list):
            x = [x]
        return [
            message_chunk_to_message(cast(BaseMessageChunk, m))
            for m in convert_to_messages(x)
        ]

    left_msgs, right_msgs = _to_msg_list(left), _to_msg_list(right)

    # ---- 2. ID 채우기 & 도구 호출 관련 필드 처리 -------------------------
    def _prepare(msgs: Messages):
        for m in msgs:
            # ID 할당
            if m.id is None:  # type: ignore[attr-defined]
                m.id = str(uuid.uuid4())  # type: ignore[attr-defined]
            
            # OpenAI 함수 호출 처리
            if strip_function_call and getattr(m, "additional_kwargs", None):
                # OpenAI function_call 필드 제거 (필요시)
                m.additional_kwargs.pop("function_call", None)
                # reasoning 필드 제거 (OpenAI o4 모델 에러 방지)
                m.additional_kwargs.pop("reasoning", None)
            
            # Gemini 도구 호출 처리
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                # ID 없는 도구 호출에 ID 부여
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict) and "id" not in tool_call:
                        tool_call["id"] = str(uuid.uuid4())
            
            # Claude 도구 호출 처리 (tool_use 필드)
            if getattr(m, "additional_kwargs", {}).get("tool_use"):
                tool_use = m.additional_kwargs.get("tool_use", {})
                if isinstance(tool_use, dict) and tool_use.get("id") is None:
                    tool_use["id"] = str(uuid.uuid4())
            
            # 도구 응답 정보 보존
            if isinstance(m, ToolMessage) and not getattr(m, "name", None):
                # 일부 모델은 ToolMessage에 name 필드를 채우지 않음
                if hasattr(m, "tool_call_id") and m.tool_call_id:
                    logger.debug(f"도구 응답에 이름 필드 누락: {m.tool_call_id}")

    _prepare(left_msgs)
    _prepare(right_msgs)

    # ---- 3. right에 전체 삭제 지시(RemoveMessage("*"))가 있는가? ----------
    for idx, m in enumerate(right_msgs):
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            merged = right_msgs[idx + 1 :]
            break
    else:
        # ---- 4. 일반 머지 로직 ---------------------------------------------
        merged = left_msgs.copy()
        index_by_id = {m.id: i for i, m in enumerate(merged)}  # type: ignore[attr-defined]
        ids_to_remove: set[str] = set()

        for m in right_msgs:
            mid = m.id  # type: ignore[attr-defined]
            if mid in index_by_id:
                if isinstance(m, RemoveMessage):
                    ids_to_remove.add(mid)
                else:
                    ids_to_remove.discard(mid)
                    merged[index_by_id[mid]] = m
            else:
                if isinstance(m, RemoveMessage):
                    logger.warning(f"존재하지 않는 메시지 ID 삭제 시도: id='{mid}'")
                else:
                    index_by_id[mid] = len(merged)
                    merged.append(m)

        merged = [m for m in merged if m.id not in ids_to_remove]  # type: ignore[attr-defined]

    # ---- 5. 필요 시 OpenAI 포맷으로 변환 ---------------------------------
    if format == "langchain-openai":
        try:
            from langgraph.graph.message import _format_messages
            merged = _format_messages(merged)
        except ImportError:
            logger.error("langgraph.graph.message._format_messages를 가져올 수 없습니다.")
    elif format:
        raise ValueError("format은 'langchain-openai' 또는 None이어야 합니다")

    return merged

def debug_message_types(messages: Messages) -> None:
    """
    메시지 타입 디버깅용 함수
    """
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", None)
        additional = getattr(msg, "additional_kwargs", {})
        
        logger.debug(f"{i}: {msg_type}, id={msg.id}")
        
        if tool_calls:
            logger.debug(f"  - tool_calls: {len(tool_calls)}")
            for tc in tool_calls:
                logger.debug(f"    - {tc.get('name', '?')}: {tc.get('id', '?')}")
        
        if additional:
            logger.debug(f"  - additional_kwargs: {list(additional.keys())}")