# YOLO Instance Segmentation 카메라 샘플

OpenCV로 카메라 프레임을 읽고 Ultralytics YOLO로 여러 객체를 검출하여,
객체별 픽셀 마스크와 클래스, 신뢰도를 실시간으로 표시합니다.
기본 모델은 COCO 데이터셋으로 사전 학습된 `yolo11n-seg.pt`입니다.
사용자 정의 객체를 구분하려면 해당 객체로 학습한 세그멘테이션 가중치가 필요합니다.

## 구성

- [yolo_seg.py](yolo_seg.py): 카메라 입력, 추론, 시각화, 객체별 이미지 저장
- [requirements.txt](requirements.txt): Ultralytics, OpenCV GUI 패키지, NumPy 의존성
- [test_yolo_seg.py](test_yolo_seg.py): 카메라 없이 실행하는 회귀 테스트

## 설치 (Windows PowerShell)

Python과 GUI를 사용할 수 있는 Windows 환경, 연결된 카메라가 필요합니다.
PyTorch를 포함한 의존성은 가상환경에 설치합니다.

```powershell
cd C:\dev\object_tracking\yolo_segmentation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

가상환경 활성화 없이 위처럼 실행 파일을 직접 사용하면 PowerShell 실행 정책의
영향을 받지 않습니다. 최초 설치는 PyTorch 등으로 인해 다운로드 용량이 클 수 있습니다.
기본 설치로 GPU가 자동 보장되지는 않습니다. NVIDIA GPU를 사용하려면
[PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라
드라이버/CUDA에 맞는 PyTorch를 같은 가상환경에 설치하세요.

## 실행

설치 후 [run.bat](run.bat)을 더블 클릭하거나 PowerShell에서 실행할 수 있습니다.
배치 파일은 자신의 폴더로 이동하여 가상환경의 Python을 실행하고,
추가 인자를 그대로 전달합니다.

```powershell
.\run.bat
.\run.bat --camera 1 --device cpu
.\run.bat --help
```

Python을 직접 실행하려면 다음 명령을 사용하세요.

```powershell
# 기본 카메라(index 0), 기본 세그멘테이션 모델
.\.venv\Scripts\python.exe .\yolo_seg.py

# 다른 카메라 및 CPU 사용
.\.venv\Scripts\python.exe .\yolo_seg.py --camera 1 --conf 0.4 --device cpu

# 사용자 가중치와 저장 위치 지정 (CUDA 지원 PyTorch 필요)
.\.venv\Scripts\python.exe .\yolo_seg.py --model .\best-seg.pt --device 0 --save-dir .\captures

# 전체 옵션 확인
.\.venv\Scripts\python.exe .\yolo_seg.py --help
```

기본 가중치가 없으면 Ultralytics가 최초 실행 시 다운로드하므로 인터넷이 필요합니다.
오프라인에서는 미리 준비한 로컬 가중치 경로를 `--model`에 지정하세요.
Detection 전용 모델이 아닌 `*-seg.pt` 등 **세그멘테이션 모델**을 사용해야 합니다.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--camera` | `0` | OpenCV 카메라 인덱스 (0 이상) |
| `--model` | `yolo11n-seg.pt` | 세그멘테이션 가중치 이름 또는 경로 |
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

## 검증

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_yolo_seg
```

회귀 테스트는 임시 폴더와 모의 카메라/모델을 사용합니다.
실제 카메라 연결, GUI 표시, 실제 가중치 추론은 위 실행 명령으로 별도 확인해야 합니다.

## 문제 해결

- 카메라가 열리지 않으면 `--camera 1` 등 다른 인덱스를 시도하고,
  Windows의 카메라 개인정보 설정 및 다른 프로그램의 카메라 점유를 확인하세요.
- 화면 표시에는 `opencv-python`이 필요합니다. `opencv-python-headless`와
  함께 설치하지 말고, GUI가 있는 로컬 데스크톱 세션에서 실행하세요.
- 처리 속도가 느리면 작은 모델(`yolo11n-seg.pt`)을 사용하거나 적절한 GPU를 설정하세요.
  원본 해상도 마스크는 추가 메모리와 연산이 필요합니다.
- 모델 사용/배포 전 [Ultralytics 라이선스](https://www.ultralytics.com/license)를 확인하세요.
