import logging
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent
from tools import *
from model_config import *
from agents_prompt.unified_teacher import unified_teacher_system_prompt

logger = logging.getLogger(__name__)    

supervisor_tools = [
    get_textbook_content,
    update_task_list,
]



DEBUG = False

# --- 그래프 생성 및 컴파일 함수 ---
def create_compiled_graph(memory):
    """멀티 에이전트 그래프를 생성하고 컴파일합니다."""

    # 에이전트 생성 - Supervisor만 사용 (통합 시스템 프롬프트 사용)
    supervisor_agent = create_react_agent(
        model=Model_gpt4_1,
        state_schema=MultiAgentState,
        tools=supervisor_tools,
        prompt=unified_teacher_system_prompt,
    )
    
    # 그래프 정의
    builder = StateGraph(MultiAgentState)
    
    # 노드 추가 - supervisor만
    builder.add_node("supervisor", supervisor_agent)
    
    builder.add_edge(START, "supervisor")

    # 그래프 컴파일
    logger.info("그래프 컴파일 시작...")
    compiled_graph = builder.compile(checkpointer=memory, debug=DEBUG)
    logger.info("그래프 컴파일 완료")

    return compiled_graph


