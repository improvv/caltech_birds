![Title_image](docs/birds_classification_title_image.png)

# 새 이미지 분류 (Image Classification of birds)

[English](README.md) | **한국어**

**최신 이미지 분류 기법으로 딥러닝 새 분류기를 만들고, 그 모델을 이해하는 과정을 빠르게 훑어보는 프로젝트**


[Caltech UCSD Birds 데이터셋 (CUB-200-2011)](http://www.vision.caltech.edu/visipedia/CUB-200.html)을 사용해, PyTorch CNN으로 새의 종(species)을 분류하는 문제를 다룹니다.

PyTorch, Torchvision 및 여러 서드파티 패키지를 사용해 합성곱 신경망(CNN)을 설정·구축·학습·평가하는 방법을 보여주는 노트북과 도구 모음입니다. 클래스 간 차이가 미세하고(fine-grained) 클래스별 데이터 수가 고르지 않은(long-tailed) 분류 문제에서 최고 수준의 성능을 내는 것이 목표입니다. 예제 노트북은 다음 워크플로를 다룹니다:

![workflow](docs/birds_roadmap.png)

예제 노트북은 네트워크를 학습하고 평가하는 데 필요한 모든 과정을 단계별로 안내합니다. 또한 뉴런·레이어·공간 활성화 해석([Lucent](https://github.com/greentfrapp/lucent))과 이미지 특징 기여도 분석([Captum](https://captum.ai/))을 통해, 네트워크가 **무엇을 보고 어떻게** 판단하는지도 살펴봅니다.

마지막으로 PyTorch로 학습한 모델을 [ONNX (Open Neural Network eXchange)](https://onnx.ai/) 형식으로 변환하는 방법을 보여줍니다. 이렇게 하면 새 분류기를 클라우드나 개인 기기 등 다양한 플랫폼에 배포할 수 있습니다. ONNX 버전 모델을 ***ONNX Runtime***으로 테스트하고 PyTorch 버전의 예측과 비교하여, 추론 성능이 동일함을 확인합니다.

# 변경 이력 (Change Log)

## v2.5 cub_tools 패키지 설치 및 환경 생성 방식 업데이트

### cub_tools 패키지 설치

`pkg/`에 있는 cub_tools가 [PYPI에 배포](https://pypi.org/project/cub-tools/)되어, 어떤 환경에서든 아래 명령으로 쉽게 설치할 수 있습니다:

```shell
pip install cub-tools
```

최신(head) 버전은 이 저장소를 클론한 뒤 다음처럼 설치할 수 있습니다:

```shell
cd pkg/

pip install .
```

### Python 학습 환경 설치

실행 환경을 설치하기 위한 환경 파일도 제공됩니다.

**CUDA 드라이버가 설치된 머신에서 실행해야 합니다(작성 시점 기준 11.1). CUDA 버전을 바꾸려면 설치된 버전에 맞는 cudatoolkit 버전인지, 그리고 PyTorch 버전도 맞는지 확인하세요.**

PyTorch는 Conda로 설치하는 것을 권장합니다. 작성 시점에 PYPI 버전(v1.8.1)은 백엔드 라이브러리 링크 오류로 계산 오류가 발생했지만, Conda로 설치한 경우에는 문제가 없었습니다.

conda 의존성 파일로 환경을 설치하려면:

```shell
conda env create -f conda_dependencies.yml
```


## V2.0 YACS YAML 설정 파일을 사용하도록 학습 스크립트 리팩토링

학습 스크립트가 새로운 Trainer 클래스 구현을 사용하도록 전면 리팩토링되었습니다.

### YACS YAML 설정 파일

모델 학습에 필요한 모든 설정과 파라미터는 이제 YAML 설정 파일로 지정합니다.

_cub_tools_ 패키지가 지원하는 3개 모델 라이브러리에 대한 예시는 _scripts/configs_ 에 있습니다.

설정 가능한 전체 필드 목록은 _pkg/cub_tools/config.py_ 를 참고하세요.

```yaml
# my_project/config.py

MODEL:
    MODEL_LIBRARY: 'timm'
    MODEL_NAME: 'resnext101_32x8d'
DIRS:
    ROOT_DIR: '/home/edmorris/projects/image_classification/caltech_birds'
TRAIN:
    NUM_EPOCHS: 40
```

> 💡 `ROOT_DIR`은 본인 환경의 저장소 경로로 바꿔야 합니다.

### 새 학습 스크립트

학습 스크립트를 실행할 때 설정 파일을 지정하여 모델을 학습합니다.

TIMM 라이브러리의 ResNeXt101_32x8d를 학습하려면 scripts 디렉터리에서 다음을 실행합니다:

```shell
python train_pytorch_caltech_birds.py --config configs/timm/resnext101_32x8d_config.yaml
```

이번 릴리스에는 PyTorch Ignite 학습 프레임워크가 추가되어 더 많은 학습 정보를 얻을 수 있습니다. 구현된 기능은 다음과 같습니다.
   1. **Tensorboard 로깅**. models/ 디렉터리를 로그 디렉터리로 지정해 Tensorboard를 실행하면 모니터링할 수 있습니다.
   2. **모델 체크포인트**. 매 반복마다 가장 좋은 모델을 저장합니다.
   3. **조기 종료(Early Stopping)**. 지정한 반복 횟수 동안 손실(loss)이 개선되지 않으면 학습을 종료합니다.

PyTorch Ignite의 추가 기능을 사용하려면 다음을 실행합니다:

```shell
python train_pytorch_ignite_caltech_birds.py --config configs/timm/resnext101_32x8d_config.yaml
```

오픈소스 MLOps 시스템인 ClearML을 사용하면 머신러닝 실험을 기록하고 원격으로 실행할 수 있습니다. 코드 상태(저장소 기준), 환경 의존성, 데이터·모델 버전, 학습 지표와 결과물까지 모델 학습의 모든 것을 웹 GUI에서 쉽게 확인할 수 있습니다. ClearML로 학습을 배포할 수 있는 버전의 학습 스크립트도 있습니다. 이 버전은 Ignite를 사용하는데, Ignite가 ClearML과 연동되어 학습 지표를 바로 기록해주기 때문입니다.

```shell
python train_clearml_pytorch_ignite_caltech_birds.py --config configs/timm/resnext101_32x8d_config.yaml
```

## V1.0 최초 릴리스

모델별 학습 스크립트가 포함된 워크숍 자료 릴리스.


# 데이터셋 정보 및 데이터

## 데이터셋

Caltech-UCSD Birds-200-2011 (CUB-200-2011)은 CUB-200 데이터셋의 확장판으로, 클래스당 이미지 수가 약 2배로 늘었고 새로운 부위(part) 위치 주석이 추가되었습니다. 자세한 내용은 아래 링크된 기술 보고서를 참고하세요.



클래스(종) 수: 200

이미지 수: 11,788

이미지당 주석: 부위 위치 15개, 이진 속성 312개, 바운딩 박스 1개

관련 데이터셋으로는 Caltech-256, Oxford Flower Dataset, Animals with Attributes 등이 있습니다. 더 많은 데이터셋은 Caltech Vision Dataset Archive에서 볼 수 있습니다.

## 데이터 파일

아래 링크에서 데이터를 다운로드하세요:

   1. 이미지 및 주석 [CalTech Visipedia Website - CUB_200_2011.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

   **이미지 압축 파일은 저장소 루트에 다운로드한 뒤, 아래 구조처럼 data/images_orig 하위 디렉터리에 압축을 풀어야 합니다.**


   2. 세그멘테이션 (선택 사항, 이 프로젝트에는 필요 없음) [CalTech Visipedia Website - segmentations.tgz](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz)

파일은 클론한 caltech_birds 저장소의 루트에 둡니다.

다운로드한 압축 파일을 풀어서 아래와 같은 프로젝트 구조를 만듭니다:

    caltech_birds-|
        data-|   ** CALTECH VISIPEDIA 웹사이트에서 별도 다운로드 **
            attributes-|
            images_orig-|   ** 압축을 푼 images 폴더 이름을 images_orig로 변경 **
               ##class_name##-|
                  class001-image001.jpg
                  class001-image002.jpg
            images-|   ** 첫 번째 워크플로 단계(데이터 정리 노트북)에서 새로 생성되는 폴더 **
               train-|
                  ##class_name##-|
                     class###-image001.jpg
               test-|
                  ##class_name##-|
                     class###-image035.jpg
            parts-|
            attributes.txt
            bounding_boxes.txt
            classes.txt
            image_class_labels.txt
            images.txt
            README
            train_test_split.txt
        example-notebooks-|   ** 워크숍 자료가 여기 있음 **
        models-|   ** RELEASES에서 별도 다운로드 **
            classification-|
                #modelname1#-|
                #modelname2#-|
        notebooks-|      ** 새 분류 문제에 대한 다양한 CNN 아키텍처 실험 노트북 **
        pkg-|            ** CUB_TOOLS 파이썬 패키지. 여기서 설치 **
        scripts-|        ** PYTORCH CNN 모델 학습 스크립트 **

> 💡 `data/`와 `models/`는 용량이 커서 `.gitignore`에 포함되어 있습니다. GitHub에는 올라가지 않으므로 각자 다운로드해서 위치시켜야 합니다.


# 워크숍 따라하기 노트북 시리즈

**example_notebooks 디렉터리에 따라하기용 워크숍 노트북이 있습니다.** ResNet152 신경망 아키텍처를 사용해 딥러닝 새 분류기를 만드는 전체 워크플로를 따라갈 수 있습니다.

노트북 설치 및 사용 요구사항, 그리고 CUB-200-2011 데이터셋으로 더 많은 CNN 아키텍처를 실험한 추가 폴더들에 대한 설명은 아래를 참고하세요.


# 설치

설치 방법은 [패키지 설치 안내](https://github.com/ecm200/caltech_birds/blob/master/pkg/readme.md)를 참고하세요.

# 추가 파일 설명

**notebooks** 디렉터리에는 시각화와 상위 수준 코드 대부분이 담긴 Jupyter 노트북이 있습니다.

**scripts** 디렉터리에는 여러 딥 CNN 아키텍처의 학습 스크립트가 있습니다. 계산량이 많아 (GPU에서도) 오래 걸리므로 파이썬 스크립트로 실행하는 편이 낫습니다. 클라우드에서 실행한다면, 원격 호스트에서 로그아웃해도 프로세스가 죽지 않도록 터미널 세션을 유지하는 방법을 쓰는 것을 추천합니다. 정해진 에포크를 다 돌리는 데 보통 몇 시간에서 하루 정도 걸립니다. 저자는 [TMUX](https://github.com/tmux/tmux/wiki/Getting-Started)를 선호합니다. TMUX는 별도의 터미널 세션을 유지해주는 리눅스 도구로, 세션을 붙였다(attach) 뗐다(detach) 할 수 있습니다. TMUX 세션 안에서 긴 학습 스크립트를 실행하고 세션을 뗀 뒤 터미널을 닫아도 프로세스는 계속 돌아갑니다. 나중에 새 터미널에서 실행 중인 TMUX 세션을 다시 붙이면 터미널 출력 기록을 포함해 진행 상황을 확인할 수 있습니다.

**pkg** 디렉터리에는 CNN 모델의 데이터 처리, 시각화, 학습, 평가, 결과 후처리를 위해 개발된 유틸리티 함수가 모두 들어 있습니다. 파이썬 패키지로 만들어져 있어 **pkg** 디렉터리에서 *pip install -e .* 를 실행하면 로컬 환경에 설치됩니다. 설치 후 ***import cub_tools*** 로 함수를 사용할 수 있습니다.

**models** 디렉터리에는 모델 학습 결과와, 모델 예측·네트워크 특징 맵 등 평가 과정의 결과물이 저장됩니다. **예제 노트북에서 사용하는 모든 모델 결과물은 GitHub 저장소의 release에서 다운로드할 수 있습니다.** models zip 파일을 저장소 루트에 두고 압축을 풀면 ResNet152 결과가 담긴 models 디렉터리가 만들어집니다. PNASNET, Inception V3·V4, GoogLeNet, ResNeXt 계열 등 다른 모델 결과도 있습니다.
