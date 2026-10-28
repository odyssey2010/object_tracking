# YOLO Instance Segmentation / OBB / Depth 카메라 샘플

OpenCV로 카메라 프레임을 읽고 Ultralytics YOLO로 여러 객체를 검출하여,
객체별 픽셀 마스크와 클래스, 신뢰도를 실시간으로 표시합니다.
기본 모델은 COCO 데이터셋으로 사전 학습된 **YOLO26 Nano Segmentation
(`yolo26n-seg.pt`)**입니다.
사용자 정의 객체를 구분하려면 해당 객체로 학습한 세그멘테이션 가중치가 필요합니다.

## 구성

- [yolo_seg.py](yolo_seg.py): 카메라 입력, 추론, 시각화, 객체별 이미지 저장
- [yolo_obb.py](yolo_obb.py): 카메라 기반 회전 바운딩 박스(OBB) 검출 및 좌표 저장
- [yolo_depth.py](yolo_depth.py): YOLO26 Depth / Depth Anything V2 카메라 깊이 추정 비교
- [run_depth.bat](run_depth.bat): 깊이 추정 실행
- [run_seg.bat](run_seg.bat) / [run_obb.bat](run_obb.bat): 세그멘테이션 / OBB 실행
- [models](models): 다운로드한 `.pt` 가중치 보관 폴더 (Git 추적 제외)
- [requirements.txt](requirements.txt): Ultralytics, OpenCV GUI, NumPy, Transformers 의존성
- [test_yolo_seg.py](test_yolo_seg.py): 카메라 없이 실행하는 회귀 테스트
- [test_yolo_obb.py](test_yolo_obb.py): OBB 좌표 저장 및 카메라 루프 회귀 테스트
- [test_yolo_depth.py](test_yolo_depth.py): 깊이 추정 변환, 저장 및 카메라 루프 회귀 테스트

## 설치 (Windows PowerShell)

Python과 GUI를 사용할 수 있는 Windows 환경, 연결된 카메라가 필요합니다.
PyTorch를 포함한 의존성은 가상환경에 설치합니다.

```powershell
cd C:\dev\object_tracking\yolo_segmentation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

YOLO26 실행을 확인한 Ultralytics 8.4.191을 최소 의존성 버전으로 지정했습니다.
기존 가상환경도 위 `pip install -r requirements.txt` 명령으로 갱신할 수 있습니다.

가상환경 활성화 없이 위처럼 실행 파일을 직접 사용하면 PowerShell 실행 정책의
영향을 받지 않습니다. 최초 설치는 PyTorch 등으로 인해 다운로드 용량이 클 수 있습니다.
기본 설치로 GPU가 자동 보장되지는 않습니다. NVIDIA GPU를 사용하려면
[PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라
드라이버/CUDA에 맞는 PyTorch를 같은 가상환경에 설치하세요.

## 실행

설치 후 [run_seg.bat](run_seg.bat)을 더블 클릭하거나 PowerShell에서 실행할 수 있습니다.
배치 파일은 자신의 폴더로 이동하여 가상환경의 Python을 실행하고,
추가 인자를 그대로 전달합니다.

```powershell
.\run_seg.bat
.\run_seg.bat --camera 1 --device cpu
.\run_seg.bat --help
```

Python을 직접 실행하려면 다음 명령을 사용하세요.

```powershell
# 기본 카메라(index 0), 기본 세그멘테이션 모델
.\.venv\Scripts\python.exe .\yolo_seg.py

# 다른 카메라 및 CPU 사용
.\.venv\Scripts\python.exe .\yolo_seg.py --camera 1 --conf 0.4 --device cpu

# 사용자 가중치와 저장 위치 지정 (CUDA 지원 PyTorch 필요)
.\.venv\Scripts\python.exe .\yolo_seg.py --model .\models\best-seg.pt --device 0 --save-dir .\captures

# 전체 옵션 확인
.\.venv\Scripts\python.exe .\yolo_seg.py --help
```

기본 가중치가 없으면 Ultralytics가 최초 실행 시 다운로드하므로 인터넷이 필요합니다.
오프라인에서는 미리 준비한 로컬 가중치 경로를 `--model`에 지정하세요.
Detection 전용 모델이 아닌 `*-seg.pt` 등 **세그멘테이션 모델**을 사용해야 합니다.

이 작업 환경에는 `models\yolo26n-seg.pt`가 있습니다(6,719,965 bytes).
기본 모델 경로는 스크립트 위치 기준의 `models` 폴더로 결정되므로
다른 작업 폴더에서 Python을 실행해도 같은 기본 가중치를 사용합니다.
`--model`로 직접 지정하는 상대 경로는 기존처럼 현재 작업 폴더 기준입니다.
공식 모델을 추가로 받을 때도 `--model .\models\모델명.pt`처럼 경로를 지정하세요.
가중치 파일은 `.gitignore`로 Git 추적에서 제외되므로 다른 PC에서는 다시 받아야 합니다.
카메라 실행 없이 기본 가중치만 다운로드하려면 다음 명령을 사용하세요.

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO(r'models\yolo26n-seg.pt')"
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--camera` | `0` | OpenCV 카메라 인덱스 (0 이상) |
| `--model` | 스크립트 기준 `models\yolo26n-seg.pt` | 세그멘테이션 가중치 이름 또는 경로 |
| `--conf` | `0.5` | 검출 신뢰도 임계값 (0 초과, 1 이하) |
| `--device` | 자동 선택 | `cpu`, `0` 등의 추론 장치 |
| `--save-dir` | `captures` | 저장 폴더 (현재 작업 폴더 기준) |

### 조작 및 저장 결과

OpenCV 창에 포커스를 둔 상태에서 키를 누릅니다.

- **Q / ESC** 또는 창 닫기: 종료 및 카메라 자원 해제
- **S**: 현재 프레임을 날짜·시간별 하위 폴더에 저장
  - `frame.png`: 주석 없는 원본 BGR 프레임
  - `segmentation.png`: 마스크, 박스, 클래스, 신뢰도가 표시된 이미지
  - `object_000_class_0_mask.png`: 객체별 단일 채널 이진 마스크 (0 또는 255)
  - `object_000_class_0_isolated.png`: 원본에서 객체만 남긴 이미지 (배경은 검정)

객체별 이미지는 원본 프레임 크기를 유지하며, 클래스 번호는 모델의 클래스 목록에 대응합니다.
객체가 없으면 원본과 시각화 이미지만 저장합니다.
저장 실패, 카메라 열기/읽기 실패는 예외로 알리며, 실행 중 오류가 나도 카메라를 해제합니다.

## OBB (Oriented Bounding Box) 검출

[yolo_obb.py](yolo_obb.py)는 객체의 회전 방향에 맞춘 사각형을 표시합니다.
픽셀 마스크를 생성하는 세그멘테이션과 별도 기능이며, 기존 `run_seg.bat`의 동작은 유지됩니다.
추가 패키지 설치 없이 동일한 가상환경과 의존성을 사용합니다.

```powershell
# 기본 카메라, YOLO26 Nano OBB
.\run_obb.bat

# CPU 또는 다른 카메라 사용, 추론 입력 크기 조정
.\run_obb.bat --camera 1 --device cpu --conf 0.4 --imgsz 640

# 더 큰 공식 모델 또는 직접 학습한 OBB 모델
.\run_obb.bat --model .\models\yolo26s-obb.pt --device 0
.\run_obb.bat --model .\models\best-obb.pt --save-dir .\captures\custom_obb

# Python 직접 실행 / 옵션 확인
.\.venv\Scripts\python.exe .\yolo_obb.py --device cpu
.\run_obb.bat --help

# 카메라 없이 가중치 다운로드
New-Item -ItemType Directory -Force .\models | Out-Null
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO(r'models\yolo26n-obb.pt')"
```

기본 모델 경로는 스크립트 기준 **`models\yolo26n-obb.pt`**이며 없으면 다운로드합니다.
현재 작업 환경에는 가중치 다운로드를 완료했습니다(5,907,357 bytes).
가중치는 Git 추적에서 제외되므로 다른 PC에서는 다시 다운로드해야 합니다.
공식 YOLO26 OBB 가중치는 `yolo26n-obb.pt`, `yolo26s-obb.pt`,
`yolo26m-obb.pt`, `yolo26l-obb.pt`, `yolo26x-obb.pt`입니다.
사용자 학습 파일은 실제 task가 `obb`여야 하며, 세그멘테이션·일반 검출 모델은
카메라를 열기 전에 오류로 거부합니다.

**주의: 기본 OBB 모델은 COCO가 아니라 항공 영상 데이터셋 DOTAv1의 15개 클래스로
사전 학습되어 있습니다.** 비행기, 선박, 소형/대형 차량, 저장 탱크, 다리,
항구, 스포츠 시설 등의 항공 영상 검출에 적합합니다.
일반 실내 웹캠에서 사람·컵·제품이 검출되지 않아도 카메라 오류를 의미하지 않습니다.
제품/부품의 회전 박스를 얻으려면 해당 객체를 OBB 라벨로 학습한 모델이 필요합니다.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--camera` | `0` | 카메라 인덱스 (0 이상) |
| `--model` | 스크립트 기준 `models\yolo26n-obb.pt` | OBB 가중치 |
| `--conf` | `0.5` | 신뢰도 (0 초과, 1 이하) |
| `--device` | 자동 선택 | `cpu`, `0` 등 |
| `--imgsz` | `1024` | 양의 정수 추론 입력 크기. 속도가 느리면 `640` 등으로 비교 |
| `--save-dir` | `captures\obb` | 저장 폴더 |

`Q` / `ESC` 또는 창 닫기로 종료합니다. `S`를 누르면 날짜·시간별 폴더에 저장합니다.

- `frame.png`: 주석 없는 원본
- `obb.png`: 회전 박스, 클래스, 신뢰도 시각화
- `detections.json`: 이미지 크기, 좌표/각도 단위, 객체 목록
  - `class_id`, `class_name`, `confidence`
  - `xywhr`: `[중심 x, 중심 y, 너비, 높이, 회전각]`
  - `corners_xy`: `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`

좌표는 추론 입력 크기가 아닌 **원본 이미지 픽셀**, 회전각은 **라디안**입니다.
결과는 `result.boxes`가 아닌 **`result.obb`**에서 가져옵니다.
회전각은 객체의 3D 자세나 진행 방향이 아니며, 박스의 180도 회전은 같은 형상입니다.
예측의 너비가 항상 긴 변인 것은 아니므로 각도를 고정 범위로 가정하지 말고,
기하학적 처리는 네 꼭짓점 좌표를 사용하는 것이 안전합니다.
검출이 없으면 JSON의 `detections`는 빈 배열이며 이미지도 저장합니다.
OBB 출력 자체가 없는 잘못된 결과 및 저장 실패는 오류로 알립니다.

공식 참고: [OBB 작업 및 사전 학습 모델](https://docs.ultralytics.com/tasks/obb/),
[DOTAv1 데이터셋](https://docs.ultralytics.com/datasets/obb/dota-v1/).

2026-10-28 검증: 세그멘테이션·OBB 회귀 테스트 총 17개 통과,
실제 카메라 3프레임의 OBB CPU 추론·GUI 표시·정상 종료 확인.
[공식 선박 예제 이미지](https://ultralytics.com/images/boats.jpg)에서
`conf=0.5`, `imgsz=1024`로 OBB 161개를 검출하고 JSON 좌표가 실제 모델 출력과
일치하는지 확인했습니다. 검출 수는 가중치·라이브러리·설정에 따라 달라질 수 있습니다.

## 사용할 수 있는 세그멘테이션 가중치 (`-seg.pt`)

조사 기준일: **2026-10-28**. 아래 목록은 Ultralytics 공식 문서에 공개된
가중치 기준이며, 임의의 제3자 체크포인트를 모두 포함하는 목록은 아닙니다.
공식 지원과 이 PC에서 실제 실행한 결과를 구분합니다.
**실제 카메라 추론까지 확인한 모델은 기본값인 `yolo26n-seg.pt`와
이전 기본값인 `yolo11n-seg.pt`입니다.**
나머지 모델은 공식 지원 목록에 근거하며 이 프로젝트에서 개별 실행 검증하지 않았습니다.

### 1. 현재 샘플의 `--model`로 선택할 모델

다음 모델들은 일반 객체의 **인스턴스 세그멘테이션**용입니다.
COCO 사전 학습 모델은 사람, 자동차, 버스, 고양이, 개, 병, 컵 등
80개 클래스의 객체를 구분하고 객체마다 별도 마스크를 생성합니다.
모델이 커진다고 학습된 클래스 종류가 늘어나는 것은 아닙니다.

| 계열 | 공개된 가중치 파일명 | 용도 및 선택 기준 |
| --- | --- | --- |
| YOLO26 | `yolo26n-seg.pt`, `yolo26s-seg.pt`, `yolo26m-seg.pt`, `yolo26l-seg.pt`, `yolo26x-seg.pt` | 현재 샘플의 기본 계열. 공식 문서에서 신규 프로젝트에 권장하는 최신 계열로 속도·마스크 품질 비교에 사용. 최신 계열을 지원하는 Ultralytics 필요 |
| YOLO11 | `yolo11n-seg.pt`, `yolo11s-seg.pt`, `yolo11m-seg.pt`, `yolo11l-seg.pt`, `yolo11x-seg.pt` | 이전 기본 계열. 기존 결과 재현 및 YOLO26과의 성능 비교에 사용 |
| YOLOv8 | `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt` | 기존 YOLOv8 프로젝트의 결과 재현, 기존 가중치·배포 환경과 비교 |
| YOLOv9 | `yolov9c-seg.pt`, `yolov9e-seg.pt` | YOLOv9 아키텍처와 기존 실험 비교. `c`/`e`는 이 계열의 모델 변형 이름이며 `n`/`s`로 임의 치환하지 않음 |

위 네 계열의 공식 가중치는 **총 17개**입니다.
기본 모델을 변경하지 않고 실행할 때만 `--model`을 지정하면 됩니다.
오래된 Ultralytics에서는 최신 모델을 로드하지 못할 수 있으므로,
로드 오류가 발생하면 공식 지원 버전과 현재 설치 버전을 확인하세요.
파일 이름에 `-seg`가 있다고 해서 모든 출처의 파일이 호환되는 것은 아닙니다.

### 2. 모델 크기별 용도

아래는 같은 계열 안에서의 일반적인 선택 지침이며, FPS나 정확도 보장값은 아닙니다.

| 크기 | 의미 | 권장 용도 | 고려사항 |
| --- | --- | --- | --- |
| `n` | Nano | CPU 카메라 데모, 저전력 장치, 빠른 동작 확인 | 먼저 시도할 크기. 복잡한 장면의 마스크 품질은 상위 크기와 비교 필요 |
| `s` | Small | 가벼운 GPU 환경, 속도와 품질의 절충 | `n`의 품질이 부족할 때 다음 후보 |
| `m` | Medium | GPU 기반 일반 응용, 복잡한 장면 비교 | 메모리·지연시간 증가를 측정 |
| `l` | Large | 마스크 품질을 우선하는 GPU 작업 | 실시간 목표 FPS 달성 여부 확인 |
| `x` | Extra Large | 충분한 GPU 자원이 있는 정밀 분석, 오프라인 처리 | 같은 계열에서 대체로 연산량과 메모리 요구가 가장 큼 |

공식 성능표를 비교할 때는 **Detection의 box mAP가 아니라 Segmentation의 mask mAP**를
확인하세요. CPU ONNX / GPU TensorRT 벤치마크 속도를 이 샘플의 PyTorch `.pt`
추론 FPS로 해석하면 안 됩니다. 카메라 해상도, GPU, 전처리·화면 표시 및
`retina_masks=True` 비용을 포함하여 실제 환경에서 측정해야 합니다.

```powershell
# 현재 검증된 기본 모델: CPU 카메라 샘플
.\run_seg.bat --device cpu

# 같은 계열에서 품질을 비교할 Small 모델
.\run_seg.bat --model .\models\yolo26s-seg.pt --device 0

# 이전 기본 모델과 비교
.\run_seg.bat --model .\models\yolo11n-seg.pt --device cpu

# 기존 YOLOv8 또는 YOLOv9 환경과 비교
.\run_seg.bat --model .\models\yolov8n-seg.pt --device cpu
.\run_seg.bat --model .\models\yolov9c-seg.pt --device 0
```

`--device 0`은 CUDA를 지원하는 PyTorch와 NVIDIA GPU가 있을 때 사용합니다.
공식 가중치는 최초 사용 시 다운로드되며, 전체 모델을 미리 받을 필요는 없습니다.
직접 파일을 구할 때는 [Ultralytics 공식 가중치 릴리스](https://github.com/ultralytics/assets/releases)를
이용하고, 오프라인 실행에는 내려받은 로컬 파일 경로를 지정하세요.

### 3. 별도 코드 또는 학습이 필요한 모델

| 모델/파일 | 용도 | 현재 샘플에서의 주의사항 |
| --- | --- | --- |
| 사용자 학습 `best.pt`, `last.pt` | 부품, 제품, 결함 등 COCO 외의 특정 객체 분리 | Ultralytics 세그멘테이션 학습 결과라면 `--model .\models\best.pt`로 지정 가능. 이름에 `-seg`가 없어도 실제 task가 `segment`이면 사용 가능 |
| YOLOE-26: `yoloe-26n-seg.pt`, `yoloe-26s-seg.pt`, `yoloe-26m-seg.pt`, `yoloe-26l-seg.pt`, `yoloe-26x-seg.pt` | 텍스트 또는 이미지 예시로 대상 클래스를 지정하는 오픈 보캐뷸러 세그멘테이션 | 이 문서의 일반 COCO 모델과 별도 사용 흐름. 공식 `YOLOE` API 및 프롬프트 설정 코드가 필요하며 현재 샘플에서 이름만 바꿔 사용하는 대상으로 취급하지 않음 |
| YOLOE-11: `yoloe-11s-seg.pt`, `yoloe-11m-seg.pt`, `yoloe-11l-seg.pt` | YOLO11 기반 텍스트·시각 프롬프트 세그멘테이션 | `YOLOE` API 및 프롬프트 설정 필요 |
| YOLOE-v8: `yoloe-v8s-seg.pt`, `yoloe-v8m-seg.pt`, `yoloe-v8l-seg.pt` | YOLOv8 기반 텍스트·시각 프롬프트 세그멘테이션 | `YOLOE` API 및 프롬프트 설정 필요 |
| YOLOE의 `*-seg-pf.pt` | 내장 어휘를 사용하는 프롬프트 없는 객체 분리 | `*-seg.pt`와 다른 체크포인트이며 `set_classes()`를 지원하지 않음. 별도 YOLOE 사용 흐름 확인 |
| YOLOv5의 `yolov5n-seg.pt`, `yolov5s-seg.pt`, `yolov5m-seg.pt`, `yolov5l-seg.pt`, `yolov5x-seg.pt` | 기존 YOLOv5 저장소 기반 세그멘테이션 프로젝트 | 원본 YOLOv5 저장소의 가중치는 현재 `ultralytics` 라이브러리와 호환되지 않으므로 이 샘플에 직접 지정하지 않음 |
| YOLO12-seg | 연구용 아키텍처를 직접 학습하여 사용 | 공식 문서상 세그멘테이션 YAML은 있지만 공식 사전 학습 `yolo12*-seg.pt`는 공개되지 않음. 파일명을 추측하여 다운로드하지 않음 |

YOLOE 텍스트 프롬프트는 추가 텍스트 인코더와 의존성 다운로드가 필요할 수 있습니다.
또한 오픈 보캐뷸러라고 해도 모든 특수 제품·결함을 정확하게 분리한다고 보장되지 않습니다.
산업용 특정 객체는 라벨링과 사용자 정의 학습·평가가 필요할 수 있습니다.

`yolo11n.pt` 같은 Detection 가중치, `*-cls.pt`, `*-pose.pt`, `*-obb.pt`,
`*-sem.pt`는 현재 샘플의 인스턴스 마스크 입력 모델이 아닙니다.
특히 Semantic Segmentation은 클래스별 영역 분류이며 객체별 인스턴스 분리와 다릅니다.

### 4. 공식 출처

- [Instance Segmentation: 가중치 다운로드, COCO 클래스, 마스크 성능표](https://docs.ultralytics.com/tasks/segment/)
- [YOLO26: 지원 작업과 가중치 파일명](https://docs.ultralytics.com/models/yolo26/)
- [YOLO11: 지원 작업과 가중치 파일명](https://docs.ultralytics.com/models/yolo11/)
- [YOLOv8: 지원 작업과 가중치 파일명](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv9: `c`/`e` 세그멘테이션 가중치](https://docs.ultralytics.com/models/yolov9/)
- [YOLOE: 공개 가중치와 프롬프트 사용 방식](https://docs.ultralytics.com/models/yoloe/)
- [YOLOv5: 원본 저장소 가중치 호환성 제한](https://docs.ultralytics.com/models/yolov5/)
- [YOLOv5 원본 저장소: 세그멘테이션 모델 목록](https://github.com/ultralytics/yolov5#segmentation)
- [YOLO12: 사전 학습 가중치 공개 범위](https://docs.ultralytics.com/models/yolo12/)

## 처리 흐름 및 마스크 의미

1. `cv2.VideoCapture`와 `read()`로 BGR 프레임을 가져옵니다.
2. `model.predict(..., retina_masks=True)`로 원본 해상도 마스크를 요청합니다.
3. `result.plot()`으로 마스크와 클래스 정보가 포함된 별도 이미지를 표시합니다.
4. 저장 시 `result.masks.data`를 CPU NumPy 배열로 변환하고 0/255 마스크를 만듭니다.
5. 주석 없는 원본 프레임에 `cv2.bitwise_and`를 적용하여 객체 이미지를 추출합니다.

`result.masks`는 검출 객체가 없으면 `None`일 수 있습니다.
`masks.data`는 `[객체 수, 높이, 너비]` 형태이며, 원본 해상도와 항상 같지는 않습니다.
이 샘플은 `retina_masks=True`를 사용하고 저장 전에 크기도 확인합니다.
`masks.xy`는 원본 좌표계의 폴리곤으로, 외곽선 표시에는 유용하지만
픽셀 마스크를 폴리곤으로 다시 채우면 구멍이나 분리된 영역의 표현이 달라질 수 있습니다.

이 샘플은 프레임별 **추론(predict)** 예제이므로 객체 번호는 지속적인 추적 ID가 아닙니다.
추적이 필요하면 `model.track(..., persist=True)`와 `boxes.id`를 활용하는 별도 처리가 필요합니다.
추적도 가림이나 재등장 상황에서 동일 ID를 항상 보장하지는 않습니다.

## 깊이 추정 (YOLO26 / Depth Anything V2)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 원본 | YOLO26 Nano Depth | Depth Anything V2 Small 비교
.\run_depth.bat
.\run_depth.bat --camera 1 --device cpu

# 한 모델만 실행
.\run_depth.bat --backend yolo --imgsz 768
.\run_depth.bat --backend depth-anything

# Python 직접 실행 / Depth Anything V2 Base로 변경
.\.venv\Scripts\python.exe .\yolo_depth.py --da-model depth-anything/Depth-Anything-V2-Base-hf
.\run_depth.bat --help
```

기본 모델은 `models\yolo26n-depth.pt`와
`depth-anything/Depth-Anything-V2-Small-hf`입니다. 여기서 Small은
카메라 샘플의 기본 경량 모델이며, Base 크기 모델과는 다릅니다.
`--model`로 YOLO 가중치 경로를, `--da-model`로 Hugging Face 모델 ID 또는
로컬 모델 디렉터리를 바꿀 수 있습니다.
YOLO의 기본 추론 크기는 768이고, Depth Anything은 해당 이미지 프로세서의
기본 설정을 사용합니다. 두 모델은 같은 프레임에 순차적으로 추론하므로
CPU에서는 표시 속도가 낮을 수 있습니다.

최초 실행 시 인터넷 연결이 필요하며 가중치는 자동 다운로드됩니다.
YOLO는 `models`에, Depth Anything의 Transformers 모델/프로세서는
`models\huggingface` 캐시에 보관합니다. 후자는 Git 추적에서 제외합니다.
`--backend yolo`는 Transformers 모델을 로딩하거나 다운로드하지 않습니다.

- **Q / ESC**: 종료, **S**: 현재 원본 프레임 및 활성 모델별 결과 저장
- 저장 위치: `captures\depth\<타임스탬프>`
- `frame.png`, `comparison.png`: 원본과 비교 화면
- `yolo_depth.npy`, `depth_anything_depth.npy`: 원본 해상도 `(H, W)`의
  정규화하지 않은 float32 추론값 (선택한 모델만 저장)
- `yolo_color.png`, `depth_anything_color.png`: 표시용 컬러맵

따뜻하고 밝은 색이 가까운 영역을 나타내도록 방향을 맞췄습니다.
YOLO는 값이 작을수록 가깝고, Depth Anything V2 기본 상대 깊이 모델은
값이 클수록 가깝습니다. 각 맵을 매 프레임 독립적으로 min/max 정규화하므로
**동일한 색이 두 모델 또는 다른 프레임에서 동일한 거리를 뜻하지 않습니다.**
Depth Anything V2 기본 모델은 상대 깊이를 출력하며,
YOLO26 Depth도 실제 거리로 사용하려면 장면/카메라에 맞는 스케일 보정이나
미세 조정이 필요합니다. 이 샘플은 값을 보정된 미터 단위로 표시하지 않습니다.
NPY는 `np.load(path, allow_pickle=False)`로 불러올 수 있습니다.

참고: [Ultralytics Depth](https://docs.ultralytics.com/tasks/depth/),
[Depth Anything V2 Small 모델 카드](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).
다른 크기/체크포인트로 바꿀 때는 해당 모델의 라이선스도 확인하세요.

## 검증

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_yolo_seg test_yolo_obb test_yolo_depth
```

회귀 테스트는 임시 폴더와 모의 카메라/모델을 사용합니다.
실제 카메라 연결, GUI 표시, 실제 가중치 추론은 위 실행 명령으로 별도 확인해야 합니다.

## 문제 해결

- 카메라가 열리지 않으면 `--camera 1` 등 다른 인덱스를 시도하고,
  Windows의 카메라 개인정보 설정 및 다른 프로그램의 카메라 점유를 확인하세요.
- 화면 표시에는 `opencv-python`이 필요합니다. `opencv-python-headless`와
  함께 설치하지 말고, GUI가 있는 로컬 데스크톱 세션에서 실행하세요.
- 처리 속도가 느리면 작은 모델(`yolo26n-seg.pt`)을 사용하거나 적절한 GPU를 설정하세요.
  원본 해상도 마스크는 추가 메모리와 연산이 필요합니다.
- 모델 사용/배포 전 [Ultralytics 라이선스](https://www.ultralytics.com/license)를 확인하세요.
