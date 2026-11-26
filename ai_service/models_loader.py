from dataclasses import dataclass
import os
import json
from pathlib import Path
import joblib

# from src.sequenc_utils import sequence_feature 보류

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
if not MODEL_DIR.is_absolute():
  MODEL_DIR = BASE_DIR / MODEL_DIR
  
FRAME_FEATURES = 38             # 정규화 좌표 34 + 관절각도 4

@dataclass
class ModelBundle:
  model: object
  feature_columns: list[str]
  info: dict
  classes: list[str]
  sequence_steps: int
  feature_count: int
  min_valid_frames: int
  
_bundle: ModelBundle | None = None

def _read_json(path: Path):
  return json.loads(path.read_text(encoding="utf-8"))

def load_model_bundle() -> ModelBundle:
  global _bundle
  model = joblib.load(MODEL_DIR / "rf_action_team.joblib")    
  columns = _read_json(MODEL_DIR / "rf_feature_columns.json")
  info = _read_json(MODEL_DIR / "rf_training_info.json")
  steps = int(info["sequence_steps"])
  expected = steps * FRAME_FEATURES
  
  if not(model.n_features_in_ == len(columns) == int(info["feature_count"]) == expected):   #다 같아야지 넘어감
    raise RuntimeError(   # 하나라도 다르면 오류 메세지
      f"입력 차원 불일치 : model = {model.n_features_in_}, columns = {len(columns)}, "
      f"info = {info['feature_count']}, stepsx38 = {expected}"
    )
  # if len(features) != expected:  pipeline에서 값 가져오기
  #   raise ValueError (f"입력 길이 {len()}")
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