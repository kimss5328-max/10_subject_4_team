from fastapi import FastAPI

from contextlib import asynccontextmanager
from ai_service.models_loader import load_model_bundle, is_ready, get_bundle


@asynccontextmanager
async def lifespan(app: FastAPI): # 비동기 컨텍스트 매니저   lifspan = 시작과 끝을 묶은 객체를 받도록
  load_model_bundle()             # ai_service.py 확인할 것 서버 시작 시 1번만 실행
  yield                           # 이 아래는 서버 종료 시 실행 (지금은 없음)
  
app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
  if not is_ready():            # ai_service.py 확인할 것
    return {"status": "loading"}
  bundle = get_bundle()         # ai_service.py 확인할 것
  return{
    "status" : "ok",
    "classes" : bundle.classes,
    "sequence_steps" : bundle.sequence_steps
  }