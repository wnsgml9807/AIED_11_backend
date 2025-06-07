from dataclasses import Field
import logging
import json
import os
import time
import glob
import re
import sqlite3
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List

import aiosqlite
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Cookie, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import tempfile
import aiofiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    AIMessage,
    AIMessageChunk,
)
from pydantic import BaseModel

from graph_factory import create_compiled_graph
from model_config import TaskState
from pdf_processor import process_pdf_to_vectordb, get_current_textbook

# ───────────────────────────────────────────────
# 환경 설정 & 로깅
# ───────────────────────────────────────────────
load_dotenv(override=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────
# DB / 세션 설정
# ───────────────────────────────────────────────
DB_DIR = "DB/checkpointer"
SESSION_EXPIRY_SECONDS = 3 * 60 * 60  # 3시간으로 변경

Path(DB_DIR).mkdir(parents=True, exist_ok=True)


def get_new_db_path(session_id: str) -> str:
    """새로운 DB 파일 경로를 생성한다 (timestamp 포함)."""
    ts = int(time.time())
    return os.path.join(DB_DIR, f"{session_id}_{ts}.db")


def find_latest_db_path(session_id: str) -> str:
    """세션에 해당하는 가장 최근 DB 파일을 찾고, 없으면 새 경로 반환."""
    pattern = os.path.join(DB_DIR, f"{session_id}_*.db")
    files = glob.glob(pattern)
    if files:
        files.sort(
            key=lambda p: int(re.search(r"_(\d+)\.db$", p).group(1)),
            reverse=True,
        )
        return files[0]
    return get_new_db_path(session_id)


def _file_is_expired(path: str) -> bool:
    age_sec = time.time() - os.path.getmtime(path)
    return age_sec > SESSION_EXPIRY_SECONDS # 변경된 상수 사용


async def cleanup_old_sessions() -> None:
    """SESSION_EXPIRY_SECONDS보다 오래된 DB 파일 삭제"""
    for db_file in glob.glob(os.path.join(DB_DIR, "*.db")):
        try:
            if _file_is_expired(db_file):
                os.remove(db_file)
                logger.info(f"만료된 세션 DB 삭제: {db_file}")
        except Exception as e:
            logger.error("세션 DB 정리 오류", exc_info=e)


# ───────────────────────────────────────────────
# FastAPI 초기화 & 라이프사이클
# ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await cleanup_old_sessions()
    logger.info("⏳ 오래된 세션 DB 파일 정리 완료")
    yield
    logger.info("🚪 서버 종료")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.streamlit.app", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────
# 데이터 모델
# ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt: str
    session_id: str


class TaskUpdateRequest(BaseModel):
    date: str
    task_no: int
    completed: bool
    session_id: str


class ProfessorTypeRequest(BaseModel):
    professor_type: str  # "T형" 또는 "F형"


# ───────────────────────────────────────────────
# 세션 캐시 & 그래프 생성
# ───────────────────────────────────────────────

session_graphs: Dict[str, Dict] = {}


async def get_session_graph(session_id: str):
    """세션별 LangGraph 컴파일 객체를 캐시하거나 재사용."""
    if session_id in session_graphs:
        return session_graphs[session_id]

    # 1. 최신 DB 찾기 (없으면 새 DB 경로 반환)
    db_path = find_latest_db_path(session_id)

    # 2. 연결 및 Saver 준비
    memory = await aiosqlite.connect(db_path)
    saver = AsyncSqliteSaver(memory)
    await saver.setup()

    # 3. 그래프 컴파일
    graph = create_compiled_graph(memory=saver)

    # 4. 캐시 보관
    session_graphs[session_id] = {
        "graph": graph,
        "memory": memory,
        "saver": saver,
        "db_path": db_path,
        "created_at": time.time(),
    }
    logger.info(f"세션 {session_id} 그래프 생성·캐시 (DB: {Path(db_path).name})")
    return session_graphs[session_id]


async def cleanup_session_graph(session_id: str):
    """메모리·커넥션 정리 후 캐시 제거"""
    data = session_graphs.pop(session_id, None)
    if data and (mem := data.get("memory")):
        await mem.close()
        logger.info(f"세션 {session_id} 메모리 리소스 닫힘")


# ───────────────────────────────────────────────
# 스트리밍 핸들러
# ───────────────────────────────────────────────

def serialize_task_list(task_list: List) -> List[Dict]:
    """TaskState 객체들을 직렬화 가능한 dictionary로 변환"""
    serialized_list = []
    for task in task_list:
        serialized_list.append(task.model_dump())
    return serialized_list

async def stream_agent_response(req: ChatRequest):
    await cleanup_old_sessions() # 스트림 시작 시 오래된 세션 정리
    session_data = await get_session_graph(req.session_id)
    graph = session_data["graph"]

    inputs = {"messages": [HumanMessage(content=req.prompt)], "session_id": req.session_id}
    cfg = {"configurable": {"thread_id": req.session_id}, "recursion_limit": 100}

    try:
        async for chunk in graph.astream(inputs, config=cfg, subgraphs=True, stream_mode="messages"):
            # chunk: ((path_tuple), (msg, metadata))
            try:
                path, payload = chunk
                msg, meta = payload
                agent = (path[0].split(":")[0] if path else "unknown")

                # ToolMessage
                if isinstance(msg, ToolMessage):
                    # Task 업데이트 메시지인지 확인
                    yield json.dumps({
                        "type": "tool",
                        "text": str(msg.content),
                        "tool_name": msg.name,
                        "response_agent": agent,
                    })
                    continue

                # AIMessage / Chunk
                if isinstance(msg, (AIMessage, AIMessageChunk)) and msg.content:
                    text = ""
                    if isinstance(msg.content, str):
                        text = msg.content
                    elif isinstance(msg.content, list):
                        text = "".join(item.get("text", "") for item in msg.content if isinstance(item, dict))
                    yield json.dumps({
                        "type": "message",
                        "text": text,
                        "response_agent": agent,
                        "provider": meta.get("ls_provider", "unknown") if meta else "unknown",
                    })
            except Exception as ie:
                logger.error("메시지 처리 오류", exc_info=ie)
                yield json.dumps({"type": "error", "text": str(ie), "response_agent": "system"})
    except Exception as e:
        logger.error("스트리밍 오류", exc_info=e)
        yield json.dumps({"type": "error", "text": str(e), "response_agent": "system"})
    finally:
        # 스트리밍 끝날 때 최종 task_list 상태 전송
        try:
            final_state = await graph.aget_state(cfg)
            
            if final_state.values and "task_list" in final_state.values:
                #logger.info(f"최종 task_list: {final_state.values['task_list']}")
                task_list = final_state.values["task_list"]
                serialized_tasks = serialize_task_list(task_list)
                yield json.dumps({
                    "type": "task_update",
                    "text": json.dumps(serialized_tasks),
                    "response_agent": "system"
                })
        except Exception as e:
            logger.error(f"최종 task_list 전송 오류: {e}")
        
        yield json.dumps({"type": "end", "text": "[STREAM_END]", "response_agent": "system"})


# ───────────────────────────────────────────────
# API 엔드포인트
# ───────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_gen():
        async for item in stream_agent_response(req):
            yield item + "\n"
    return StreamingResponse(event_gen(), media_type="application/json")


@app.post("/tasks/update")
async def update_task(req: TaskUpdateRequest):
    """Task 완료 상태를 업데이트합니다 (aupdate_state API 사용)"""
    try:
        # 세션의 그래프 가져오기  
        session_data = await get_session_graph(req.session_id)
        graph = session_data["graph"]
        
        cfg = {"configurable": {"thread_id": req.session_id}}
        
        #logger.info(f"Task update request: {req}")
        
        # 현재 state 가져와서 task_list 업데이트
        try:
            current_state = await graph.aget_state(cfg, subgraphs=True)
        except Exception as e:
            logger.error(f"State 가져오기 오류: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
        task_list = current_state.values.get("task_list", []) if current_state.values else []
        
        #logger.info(f"Current state: {task_list}")
        
        # task_list가 비어있으면 빈 리스트로 초기화
        if not task_list:
            logger.info(f"세션 {req.session_id}의 task_list가 비어있습니다. 초기화합니다.")
            await graph.aupdate_state(cfg, {"task_list": []})
            raise HTTPException(status_code=404, detail="No tasks found. Please create tasks first.")
        
        # 해당 task 찾아서 업데이트
        found = False
        updated_task_list = []
    
        for task in task_list:
            month = task.month
            date = task.date
            task_no = task.task_no
            start_pg = task.start_pg
            end_pg = task.end_pg
            title = task.title
            summary = task.summary
            
            if date == req.date and task_no == req.task_no:
                updated_task = TaskState(
                    month=month,
                    date=date,
                    task_no=task_no,
                    start_pg=start_pg,
                    end_pg=end_pg,
                    title=title,
                    summary=summary,
                    is_completed=req.completed)
                updated_task_list.append(updated_task)
                #logger.info(f"Updated task: {updated_task}")
                found = True
            else:
                # 변경되지 않은 task는 TaskState로 변환하여 추가
                updated_task_list.append(task)
                
        if not found:
            raise HTTPException(status_code=404, detail=f"Task not found: {req.date} task_no {req.task_no}")

        # aupdate_state API로 state 직접 업데이트
        await graph.aupdate_state(cfg, {"task_list": updated_task_list})
        
        #logger.info(f"prev_task_list: {task_list}")
        #logger.info(f"update_task_list: {updated_task_list}")
        
        return {"status": "success", "message": "Task updated successfully"}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Task update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    await cleanup_session_graph(session_id)
    return {"message": f"세션 {session_id} 정리 완료"}


@app.get("/maintenance/cleanup")
async def manual_cleanup():
    await cleanup_old_sessions()
    return {"message": "만료 세션 파일 정리 완료"}


@app.post("/data/upload")
async def upload_textbook(file: UploadFile = File(...), session_id: str = Form("default")):
    """PDF 문제집을 업로드하고 벡터화합니다."""
    try:
        logger.info(f"업로드 요청 수신 - 파일명: {file.filename}, 세션ID: {session_id}")
        
        # 파일 타입 검증
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
        
        logger.info(f"파일 업로드 시작: {file.filename}, 세션: {session_id}")
        
        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"임시 파일 저장 완료: {temp_file_path}")
        
        # PDF 처리 및 벡터화 (비동기, 세션별)
        logger.info(f"PDF 처리 시작 - 경로: {temp_file_path}, 파일명: {file.filename}, 세션: {session_id}")
        result = await process_pdf_to_vectordb(temp_file_path, file.filename, session_id)
        logger.info(f"PDF 처리 결과: {result.get('success', 'Unknown')}")
        
        # 임시 파일 삭제
        os.remove(temp_file_path)
        logger.info("임시 파일 삭제 완료")
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "collection_name": result["collection_name"],
                "total_pages": result["total_pages"],
                "filename": file.filename
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"파일 업로드 처리 중 오류: {e}", exc_info=True)  # 전체 스택 트레이스 출력
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류가 발생했습니다: {str(e)}")


@app.get("/data/textbook")
async def get_textbook(session_id: str = "default"):
    """현재 업로드된 문제집 정보를 반환합니다."""
    try:
        textbook = get_current_textbook(session_id)
        if textbook:
            return {
                "success": True,
                "textbook": textbook
            }
        else:
            return {
                "success": True,
                "textbook": None
            }
    except Exception as e:
        logger.error(f"문제집 정보 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문제집 정보 조회 중 오류: {str(e)}")


@app.post("/settings/professor-type")
async def set_professor_type(req: ProfessorTypeRequest):
    """세션별 교수자 타입을 설정합니다."""
    try:
        if req.professor_type not in ["T형", "F형"]:
            raise HTTPException(status_code=400, detail="교수자 타입은 'T형' 또는 'F형'이어야 합니다.")
        
        return {
            "success": True,
            "message": f"교수자 타입이 '{req.professor_type}'으로 설정되었습니다.",
            "professor_type": req.professor_type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"교수자 타입 설정 오류: {e}")
        raise HTTPException(status_code=500, detail=f"교수자 타입 설정 중 오류: {str(e)}")


@app.post("/sessions/{session_id}/professor-type")
async def update_session_professor_type(session_id: str, req: ProfessorTypeRequest):
    """특정 세션의 교수자 타입을 업데이트합니다."""
    try:
        if req.professor_type not in ["T형", "F형"]:
            raise HTTPException(status_code=400, detail="교수자 타입은 'T형' 또는 'F형'이어야 합니다.")
        
        # 세션의 그래프 가져오기
        session_data = await get_session_graph(session_id)
        graph = session_data["graph"]
        
        cfg = {"configurable": {"thread_id": session_id}}
        
        # state 업데이트
        await graph.aupdate_state(cfg, {"professor_type": req.professor_type})
        
        return {
            "success": True,
            "message": f"세션 {session_id}의 교수자 타입이 '{req.professor_type}'으로 설정되었습니다.",
            "professor_type": req.professor_type
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"세션 교수자 타입 설정 오류: {e}")
        raise HTTPException(status_code=500, detail=f"세션 교수자 타입 설정 중 오류: {str(e)}")


@app.get("/sessions/{session_id}/professor-type")
async def get_session_professor_type(session_id: str):
    """특정 세션의 교수자 타입을 조회합니다."""
    try:
        # 세션의 그래프 가져오기
        session_data = await get_session_graph(session_id)
        graph = session_data["graph"]
        
        cfg = {"configurable": {"thread_id": session_id}}
        
        # 현재 state 조회
        current_state = await graph.aget_state(cfg)
        professor_type = current_state.values.get("professor_type", "T형") if current_state.values else "T형"
        
        return {
            "success": True,
            "professor_type": professor_type
        }
    except Exception as e:
        logger.error(f"세션 교수자 타입 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"세션 교수자 타입 조회 중 오류: {str(e)}")





# ───────────────────────────────────────────────
# 로컬 실행 디버깅 용
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
