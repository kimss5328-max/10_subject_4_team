from dataclasses import dataclass
import os
import json
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent            # models 폴더 찾기
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
if not MODEL_DIR.is_absolute():
  MODEL_DIR = BASE_DIR / MODEL_DIR
  
FRAME_FEATURES = 38             # 정규화 좌표 34 + 관절각도 4

@dataclass                      # ModelBundle 미리 보관하는 상자 생성
class ModelBundle:
  model: object
  feature_columns: list[str]
  info: dict
  classes: list[str]
  sequence_steps: int
  feature_count: int
  min_valid_frames: int
  
_bundle: ModelBundle | None = None    # ModelBundle | None 이 변수에 상자 아니면 None이 들어간다 = None은 처음에는 빈 상태로 시작한다는 뜻

def _read_json(path: Path):           # Json 파일을 utf-8로 읽고 파이썬 객채(dict/list)로 바꿔서 리턴
  return json.loads(path.read_text(encoding="utf-8"))

def load_model_bundle() -> ModelBundle:   # load_model_bundle 함수가 ModelBundle을 만들어 돌려준다
  global _bundle
  model = joblib.load(MODEL_DIR / "rf_action_team.joblib")    
  columns = _read_json(MODEL_DIR / "rf_feature_columns.json")
  info = _read_json(MODEL_DIR / "rf_training_info.json")  
  steps = int(info["sequence_steps"])     # rf_training_info.json에서 sequence_steps 값을 가져온다 int()로 감싸는 이유는 정수형으로 맞추기 위해
  expected = steps * FRAME_FEATURES       # 모델이 받아야 할 입력 숫자의 총 개수를 계산
  
  if not(model.n_features_in_ == len(columns) == int(info["feature_count"]) == expected):   #다 같아야지 넘어감
    raise RuntimeError(   # 하나라도 다르면 오류 메세지
      f"입력 차원 불일치 : model = {model.n_features_in_}, columns = {len(columns)}, "
      f"info = {info['feature_count']}, stepsx38 = {expected}"
    ) 
  classes = [str(c) for c in model.classes_]
  if set(classes) != set(map(str, info["classes"])):
    raise RuntimeError(f"Class 불일치 : model = {classes}, info = {info['classes']}")
  
  _bundle = ModelBundle(
    model = model,
    feature_columns = list(columns),
    info = info,
    classes = classes,
    sequence_steps = steps,
    feature_count = expected,
    min_valid_frames = int(info.get("min_valid_frames",10))
    )
  return _bundle

def get_bundle() -> ModelBundle:
  if _bundle is None:
    raise RuntimeError("모델이 아직 로드되지 않았습니다.")
  return _bundle

def is_ready() -> bool:
  return _bundle is not None