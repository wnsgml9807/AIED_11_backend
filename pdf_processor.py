#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import logging
import asyncio
from typing import Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import fitz  # PyMuPDF
import unicodedata
from openai import AsyncOpenAI
from PIL import Image
import io

# 로깅 설정
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini async client (for contents index)
gemini_client: Optional[AsyncOpenAI] = None
if GEMINI_API_KEY:
    gemini_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join("DB", "textbook")

# 비동기 처리 설정
BATCH_SIZE = 100  # 임베딩 배치 크기

# ChromaDB 클라이언트 캐시 (설정 충돌 방지)
_chroma_clients = {}

def clean_filename(filename: str) -> str:
    """파일명을 안전한 폴더명으로 변환합니다."""
    # 확장자 제거
    name = os.path.splitext(filename)[0]
    # 특수문자 제거 및 공백을 언더스코어로 변경
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    # 유니코드 정규화
    name = unicodedata.normalize('NFKC', name)
    return name

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    PDF에서 페이지별로 텍스트를 추출합니다.
    
    Returns:
        List[Dict]: 각 페이지의 텍스트와 메타데이터
    """
    try:
        doc = fitz.open(pdf_path)
        pages_data = []
        
        logger.info(f"PDF 열기 성공: {pdf_path}, 총 {len(doc)} 페이지")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # 텍스트 정리
            text = re.sub(r'\n+', '\n', text)  # 연속된 개행 제거
            text = re.sub(r'\s+', ' ', text)   # 연속된 공백 제거
            text = text.strip()
            
            if text:  # 텍스트가 있는 페이지만 포함
                pages_data.append({
                    "page_number": page_num + 1,  # 1부터 시작
                    "text": text,
                    "metadata": {
                        "page": page_num + 1,
                        "chapter": f"Page_{page_num + 1}"  # 기본 챕터 정보
                    }
                })
                logger.info(f"페이지 {page_num + 1} 텍스트 추출 완료: {len(text)}자")
            else:
                logger.warning(f"페이지 {page_num + 1}: 텍스트 없음")
        
        doc.close()
        logger.info(f"PDF 텍스트 추출 완료: {len(pages_data)}개 페이지")
        return pages_data
        
    except Exception as e:
        logger.error(f"PDF 텍스트 추출 중 오류: {e}")
        return []

async def generate_page_thumbnails(pdf_path: str, session_id: str) -> bool:
    """
    PDF의 각 페이지를 썸네일 이미지로 변환하여 저장합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        session_id: 세션 ID
        
    Returns:
        bool: 성공 여부
    """
    try:
        doc = fitz.open(pdf_path)
        
        # 썸네일 저장 디렉토리 생성
        thumbnails_dir = os.path.join(DB_DIR, f"textbook_{session_id}", "thumbnails")
        os.makedirs(thumbnails_dir, exist_ok=True)
        logger.info(f"썸네일 디렉토리 생성: {thumbnails_dir}")
        
        # 200 DPI 설정 (A4 기준으로 적절한 크기)
        zoom = 200 / 72  # 기본 72 DPI에서 200 DPI로 변환
        mat = fitz.Matrix(zoom, zoom)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # 페이지를 이미지로 렌더링
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # PIL Image로 변환
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # RGB로 변환 (JPEG 저장을 위해)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 썸네일 크기로 리사이즈 (비율 유지)
            # A4 비율(210:297) 유지하며 너비 300px로 설정
            thumbnail_width = 300
            aspect_ratio = img.height / img.width
            thumbnail_height = int(thumbnail_width * aspect_ratio)
            
            img_thumbnail = img.resize((thumbnail_width, thumbnail_height), Image.Resampling.LANCZOS)
            
            # 썸네일 저장
            thumbnail_path = os.path.join(thumbnails_dir, f"page_{page_num + 1}.jpg")
            img_thumbnail.save(thumbnail_path, "JPEG", quality=90, optimize=True)
            
            logger.info(f"페이지 {page_num + 1} 썸네일 생성 완료: {thumbnail_path}")
            
            # 10페이지마다 잠시 대기 (시스템 부하 방지)
            if (page_num + 1) % 10 == 0:
                await asyncio.sleep(0.1)
        
        total_pages = len(doc)
        doc.close()
        logger.info(f"모든 페이지 썸네일 생성 완료: 총 {total_pages} 페이지")
        return True
        
    except Exception as e:
        logger.error(f"썸네일 생성 중 오류: {e}")
        return False

def create_textbook_collection(filename: str, session_id: str, collection_metadata: Dict) -> chromadb.Collection:
    """
    ChromaDB 컬렉션을 생성합니다. (세션별 교재)
    
    Args:
        filename: 원본 파일명 (메타데이터용)
        session_id: 세션 ID
        collection_metadata: 컬렉션에 대한 메타데이터
    
    Returns:
        ChromaDB Collection 객체
    """
    try:
        # 세션별 DB 디렉토리 (최종 경로에서 직접 작업)
        collection_db_dir = os.path.join(DB_DIR, f"textbook_{session_id}")
        
        # 기본 DB 디렉토리 생성 (없을 경우)
        os.makedirs(DB_DIR, exist_ok=True)
        logger.info(f"기본 DB 디렉토리 확인/생성: {DB_DIR}")
        
        # 기존 디렉토리가 있으면 완전히 삭제 후 재생성 (세션별)
        if os.path.exists(collection_db_dir):
            import shutil
            import time
            import stat
            import subprocess
            
            logger.info(f"세션 {session_id}의 기존 교재 데이터 삭제 시작...")
            
            # 1. 캐시된 클라이언트 완전 정리
            if collection_db_dir in _chroma_clients:
                try:
                    old_client = _chroma_clients[collection_db_dir]
                    del _chroma_clients[collection_db_dir]
                    logger.info("캐시된 ChromaDB 클라이언트 정리 완료")
                except Exception as cache_e:
                    logger.warning(f"캐시 정리 중 오류 (무시됨): {cache_e}")
            
            # 2. 전체 클라이언트 캐시 초기화 (안전을 위해)
            _chroma_clients.clear()
            logger.info("전체 ChromaDB 클라이언트 캐시 초기화")
            
            # 권한 문제 해결을 위한 강제 삭제 함수
            def force_remove_readonly(func, path, exc):
                """읽기 전용 파일도 강제로 삭제"""
                if os.path.exists(path):
                    try:
                        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                        func(path)
                    except Exception:
                        pass
            
            # 3. 디렉토리 삭제 재시도 메커니즘 (다양한 방법 시도)
            max_retries = 3
            deleted = False
            
            for attempt in range(max_retries):
                try:
                    # 방법 1: Python shutil 사용
                    if attempt == 0:
                        logger.info(f"삭제 시도 {attempt + 1}: Python shutil 사용")
                        # 디렉토리 내 모든 파일의 권한을 쓰기 가능으로 변경
                        for root, dirs, files in os.walk(collection_db_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)
                                except Exception:
                                    pass
                        shutil.rmtree(collection_db_dir, onerror=force_remove_readonly)
                    
                    # 방법 2: 시스템 명령어 사용 (Linux/Unix)
                    elif attempt == 1:
                        logger.info(f"삭제 시도 {attempt + 1}: 시스템 명령어 사용")
                        result = subprocess.run(['rm', '-rf', collection_db_dir], 
                                              capture_output=True, text=True, timeout=30)
                        if result.returncode != 0:
                            raise Exception(f"rm 명령어 실패: {result.stderr}")
                    
                    # 방법 3: 백업 이동
                    else:
                        logger.info(f"삭제 시도 {attempt + 1}: 백업 디렉토리로 이동")
                        backup_dir = f"{collection_db_dir}_backup_{int(time.time())}"
                        os.rename(collection_db_dir, backup_dir)
                        logger.warning(f"삭제 실패로 백업 디렉토리로 이동: {backup_dir}")
                    
                    # 삭제 성공 확인
                    if not os.path.exists(collection_db_dir):
                        logger.info(f"기존 교재 데이터 삭제 완료")
                        deleted = True
                        break
                    
                except Exception as e:
                    logger.warning(f"삭제 시도 {attempt + 1} 실패: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # 1초 대기 후 재시도
                    else:
                        logger.error(f"모든 삭제 시도 실패: {e}")
            
            if not deleted and os.path.exists(collection_db_dir):
                raise Exception(f"기존 교재 데이터 정리 실패 - 수동으로 디렉토리를 삭제해주세요: {collection_db_dir}")
            
            # 삭제 후 안정화를 위한 잠시 대기
            time.sleep(0.5)
        
        # 새 디렉토리 생성
        os.makedirs(collection_db_dir, exist_ok=True)
        logger.info(f"DB 디렉토리 생성: {collection_db_dir}")
        
        # 디렉토리 권한 설정
        try:
            os.chmod(collection_db_dir, 0o755)  # 읽기/쓰기/실행 권한 설정
        except Exception as e:
            logger.warning(f"디렉토리 권한 설정 실패 (무시됨): {e}")
        
        # ChromaDB 클라이언트 초기화 (새로 생성 - 캐시 사용 안함)
        try:
            logger.info("새로운 ChromaDB 클라이언트 생성 중...")
            
            # 새로운 클라이언트 생성 (항상 새로 생성)
            import chromadb.config
            settings = chromadb.config.Settings(
                persist_directory=collection_db_dir,
                anonymized_telemetry=False,  # 텔레메트리 비활성화로 성능 향상
            )
            client = chromadb.PersistentClient(settings=settings)
            
            # 캐시에 저장
            _chroma_clients[collection_db_dir] = client
            logger.info("ChromaDB 클라이언트 초기화 완료")
            
            # 클라이언트 상태 확인
            try:
                heartbeat = client.heartbeat()
                logger.debug(f"ChromaDB 서버 상태: {heartbeat}")
            except Exception as hb_e:
                logger.warning(f"ChromaDB 헬스체크 실패 (무시됨): {hb_e}")
                
        except Exception as e:
            logger.error(f"ChromaDB 클라이언트 초기화 실패: {e}")
            raise Exception(f"ChromaDB 클라이언트 초기화 실패: {e}")
        
        # OpenAI 임베딩 함수 설정
        try:
            embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-large"
            )
            logger.info("OpenAI 임베딩 함수 설정 완료: text-embedding-3-large")
        except Exception as e:
            logger.error(f"OpenAI 임베딩 함수 설정 실패: {e}")
            raise Exception(f"OpenAI 임베딩 함수 설정 실패: {e}")
        
        # 컬렉션 생성 (기존 컬렉션 있으면 먼저 삭제)
        try:
            logger.info("교재 컬렉션 준비 중...")

            # 기존 컬렉션 삭제 시도
            try:
                client.delete_collection("textbook")
                logger.debug("기존 'textbook' 컬렉션 삭제 완료")
            except Exception:
                pass  # 존재하지 않으면 무시

            # 새 컬렉션 생성
            collection = client.create_collection(
                name="textbook",
                embedding_function=embedding_function,
                metadata=collection_metadata
            )
            logger.info("컬렉션 'textbook' 생성 완료")
            
            # 생성된 DB 파일들의 권한도 확인
            import glob
            db_files = glob.glob(os.path.join(collection_db_dir, "*.sqlite*"))
            for db_file in db_files:
                try:
                    os.chmod(db_file, 0o644)  # 읽기/쓰기 권한
                    logger.debug(f"DB 파일 권한 설정: {db_file}")
                except Exception as perm_e:
                    logger.warning(f"DB 파일 권한 설정 실패 (무시됨): {perm_e}")
            
        except Exception as e:
            logger.error(f"컬렉션 생성 중 오류: {e}")
            # 추가 디버깅 정보
            logger.error(f"디렉토리 상태: {os.path.exists(collection_db_dir)}")
            logger.error(f"디렉토리 권한: {oct(os.stat(collection_db_dir).st_mode)[-3:] if os.path.exists(collection_db_dir) else 'N/A'}")
            
            # 모든 컬렉션 리스트 출력
            try:
                all_collections = client.list_collections()
                logger.error(f"현재 존재하는 컬렉션들: {[c.name for c in all_collections]}")
            except Exception:
                logger.error("컬렉션 리스트 조회 실패")
                
            raise Exception(f"컬렉션 생성 실패: {e}")
        
        return collection
        
    except Exception as e:
        logger.error(f"컬렉션 생성 중 오류: {e}")
        raise

async def embed_pages_batch(collection: chromadb.Collection, pages_data: List[Dict]) -> bool:
    """
    페이지 데이터를 배치로 임베딩하고 저장합니다.
    
    Args:
        collection: ChromaDB 컬렉션
        pages_data: 페이지별 텍스트 데이터
    
    Returns:
        bool: 성공 여부
    """
    try:
        total_pages = len(pages_data)
        logger.info(f"총 {total_pages}개 페이지 배치 임베딩 시작...")
        
        # 배치로 분할하여 처리
        for i in range(0, total_pages, BATCH_SIZE):
            batch = pages_data[i:i+BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (total_pages - 1) // BATCH_SIZE + 1
            
            logger.info(f"배치 {batch_num}/{total_batches} 처리 중: {len(batch)}개 페이지")
            
            # 배치 데이터 준비
            ids = [f"page_{page['page_number']}" for page in batch]
            texts = [page['text'] for page in batch]
            metadatas = [page['metadata'] for page in batch]
            
            try:
                # ChromaDB에 배치 추가
                collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                logger.info(f"배치 {batch_num} 임베딩 완료: {len(batch)}개 페이지")
                
                # 배치 간 잠시 대기 (API 제한 고려)
                if i + BATCH_SIZE < total_pages:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"배치 {batch_num} 임베딩 중 오류: {e}")
                # 개별 페이지로 재시도
                logger.info("개별 페이지로 재시도...")
                for page_id, text, metadata in zip(ids, texts, metadatas):
                    try:
                        collection.add(
                            ids=[page_id],
                            documents=[text],
                            metadatas=[metadata]
                        )
                        logger.info(f"페이지 {page_id} 개별 임베딩 완료")
                    except Exception as e2:
                        logger.error(f"페이지 {page_id} 임베딩 실패: {e2}")
        
        final_count = collection.count()
        logger.info(f"임베딩 완료. 총 {final_count}개 페이지가 데이터베이스에 저장됨")
        return True
        
    except Exception as e:
        logger.error(f"배치 임베딩 중 오류: {e}")
        return False

async def process_pdf_to_vectordb(pdf_path: str, filename: str, session_id: str, title: str) -> Dict:
    """PDF 파일을 처리하여 세션별 ChromaDB 컬렉션으로 변환합니다.

    Args:
        pdf_path: PDF 파일 경로
        filename: 업로드된 원본 PDF 파일명
        session_id: 세션 ID (세션별 DB 디렉토리)
        title: 사용자가 입력한 교재 제목
    """
    try:
        # API 키 확인
        if not OPENAI_API_KEY:
            logger.error("OpenAI API 키가 설정되지 않았습니다.")
            raise Exception("OpenAI API 키가 설정되지 않았습니다.")
        logger.info(f"OpenAI API 키 확인 완료: {OPENAI_API_KEY[:10]}...")
        
        logger.info(f"PDF 처리 시작: {filename} (세션: {session_id})")
        
        # PDF에서 텍스트 추출
        logger.info("PDF 텍스트 추출 중...")
        pages_data = extract_text_from_pdf(pdf_path)
        
        if not pages_data:
            raise Exception("PDF에서 텍스트를 추출할 수 없습니다.")
        
        total_pages = len(pages_data)

        # 목차 생성용 첫 5페이지 텍스트 추출
        first_pages_text = "\n".join([p["text"] for p in pages_data[:5]]) if pages_data else ""
        contents_index = await generate_contents_index(first_pages_text) if first_pages_text else ""

        collection_meta = {
            "title": title,
            "n_page": total_pages,
            "contents_index": contents_index,
            "filename": filename,
            "source": "uploaded_pdf",
        }

        # ChromaDB 컬렉션 생성 (세션별 교재)
        logger.info(f"세션 {session_id}용 ChromaDB 컬렉션 생성 중...")
        collection = create_textbook_collection(filename, session_id, collection_meta)
        
        # 페이지 데이터 임베딩 및 저장
        logger.info("페이지 데이터 임베딩 중...")
        success = await embed_pages_batch(collection, pages_data)
        
        if not success:
            raise Exception("임베딩 처리 중 오류 발생")
        
        # 페이지 썸네일 생성
        logger.info("페이지 썸네일 생성 중...")
        thumbnail_success = await generate_page_thumbnails(pdf_path, session_id)
        
        if not thumbnail_success:
            logger.warning("썸네일 생성 실패 - 미리보기 기능이 제한될 수 있습니다.")
        
        result = {
            "success": True,
            "message": f"PDF 처리 완료: {len(pages_data)}개 페이지 임베딩",
            "collection_name": "textbook",
            "filename": filename,
            "total_pages": len(pages_data),
            "db_path": os.path.join(DB_DIR, f"textbook_{session_id}"),
            "session_id": session_id
        }
        
        logger.info(f"PDF 처리 완료: {result}")
        return result
        
    except Exception as e:
        logger.error(f"PDF 처리 중 오류: {e}")
        return {
            "success": False,
            "message": f"PDF 처리 실패: {str(e)}",
            "collection_name": None,
            "filename": None,
            "total_pages": 0,
            "db_path": None
        }

def get_current_textbook(session_id: str) -> Dict:
    """현재 업로드된 문제집 정보를 반환합니다."""
    try:
        textbook_dir = os.path.join(DB_DIR, f"textbook_{session_id}")
        if not os.path.exists(textbook_dir):
            return None
        
        # ChromaDB 클라이언트로 컬렉션 조회 (캐시 사용)
        if textbook_dir in _chroma_clients:
            # 캐시된 클라이언트 사용
            client = _chroma_clients[textbook_dir]
            logger.debug("캐시된 ChromaDB 클라이언트 사용")
        else:
            # 새 클라이언트 생성 (읽기 전용)
            import chromadb.config
            settings = chromadb.config.Settings(
                persist_directory=textbook_dir,
                anonymized_telemetry=False,
            )
            client = chromadb.PersistentClient(path=textbook_dir, settings=settings)
            _chroma_clients[textbook_dir] = client
            logger.debug("새 ChromaDB 클라이언트 생성 (읽기 전용)")
        
        collection = client.get_collection("textbook")
        
        # 메타데이터에서 파일명 조회
        metadata = collection.metadata
        filename = metadata.get("filename", "알 수 없는 문제집")
        page_count = collection.count()
        
        return {
            "filename": filename,
            "page_count": page_count,
            "collection_name": "textbook"
        }
    except Exception as e:
        logger.error(f"문제집 정보 조회 오류: {e}")
        return None

async def generate_contents_index(first_pages_text: str) -> str:
    """Gemini를 이용해 목차(JSON 문자열) 생성"""
    if gemini_client is None:
        return ""

    system_prompt = (
        "너는 교재 편집 전문가다. 주어진 교재의 첫 부분을 읽고"
        " '제목:페이지' 형태의 목차를 최대 15항목 JSON 배열로 생성하라."
        " 형식 예시: [{\"title\":\"1. 세포의 구조\", \"page\":1}, ...]"
    )
    user_prompt = first_pages_text[:8000]  # 모델 토큰 제한 대비

    try:
        completion = await gemini_client.chat.completions.create(
            model="gemini-2.5-flash-preview-05-20",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        if completion.choices and completion.choices[0].message:
            return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Gemini contents index 생성 실패: {e}")
    return "" 