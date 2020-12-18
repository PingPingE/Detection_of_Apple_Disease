# Mask R-CNN 기반 사과 병해 진단 서비스 개발
![main](https://user-images.githubusercontent.com/47843060/100581940-ce23e000-332b-11eb-9efb-d31ae0c35b9a.JPG)
<br><br>
동영상: https://drive.google.com/file/d/1VY796aO1YnD7Uuc9CdH8h6o7nz-wRmcv/view?usp=sharing

## 파일 정보
- mrcnn 디렉토리: Mask R-CNN base code파일 포함
- apple_wandb.py: custom 모델 정의 코드
- concat_json.py: 각 병해 class의 json파일(어노테이션 정보)을 합치고 오류 검사를 위한 코드
<br><br>

## 개발 환경
- OS: Ubuntu 18.04.5 LTS
- GPU: GeForce GTX 1660
- GPU memory: 6GB
- CPU: Intel(R) Core(TM) i5-8500 CPU @ 3.00GHz
- Nvidia driver: 450.51.06
- CUDA 11.1 
- Tensorflow-gpu 1.15.0
- Keras 2.3.1
- Vgg Image Annotator
- Python 3.6.8
- Docker
- 협업: Weights & Biases / Notion / Colab
- 웹: Django
- 안드로이드: Android studio

<br><br>
## 개발 기간
- 2020.09 ~ 2020.11(2개월)

<br><br>
## 개발 인원 및 역할
- 4명
- 역할 분담
  - 모바일 개발: 2명
  - 웹 개발: 2명 
  - 모델 개발: 4명

<br><br>
## 알고리즘
- Mask R-CNN
![image](https://user-images.githubusercontent.com/47843060/101361305-276ab100-38e2-11eb-979e-e2741cd3465b.png)

<br><br>
![image](https://user-images.githubusercontent.com/47843060/101361136-f1c5c800-38e1-11eb-9f16-e5d50b558026.png)

<br><br>
## 역할
- 데이터 수집, 가공
- Mask R-CNN기반 모델 개발
- 웹 개발

<br><br>
## 학습 상세
### 데이터
- Train : 205개
- Validation : 58개
- 학습 class
  - Sooty blotch: 그을음병
  - Fly speck: 그을음점무늬병
  - Marssonia blotch: 갈반병
  - Bitter rot: 탄저병
  - White rot: 겹무늬썩음병
  - Brown rot: 잿빛무늬병
  - Normal: 정상<br>
 *Sooty와 Fly speck는 Sooty/Fly로 동시에 진단

- 학습 하이퍼 파라미터
  - Epoch: 96/300
  - Batch_size: 1
  - Backbone: ResNet 50(layers = heads)
  - Detection_min_confidence: 0.9
  - Learning_rate: 0.001->0.0001
  - Optimizer: SGD


<br><br>
## 결과

![image](https://user-images.githubusercontent.com/47843060/100582270-5609ea00-332c-11eb-9d6a-63d2eafa0d04.png)

### mAP: 0.48 
*IoU threshold:0.5


<br><br>
## 웹

### 메인
<br><br>
![main](https://user-images.githubusercontent.com/47843060/100581940-ce23e000-332b-11eb-9efb-d31ae0c35b9a.JPG)
<br><br><br>

### 진단 결과
<br><br>
![result](https://user-images.githubusercontent.com/47843060/100581890-bfd5c400-332b-11eb-8eee-be12be4713ea.JPG)
<br><br><br>


### 병해 정보
<br><br>
![info](https://user-images.githubusercontent.com/47843060/100582448-95d0d180-332c-11eb-8b9e-0d43a8571628.JPG)
<br><br><br>


### 농약사 찾기
<br><br>
![map](https://user-images.githubusercontent.com/47843060/100582385-80f43e00-332c-11eb-8288-e1d5c219513c.JPG)
<br><br><br>


### 농약 검색
<br><br>
![search](https://user-images.githubusercontent.com/47843060/100582425-8f425a00-332c-11eb-9f2c-57072c7a3abf.JPG)
<br><br><br>

