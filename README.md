# sex_age_recognize_from_picture

학부 졸업 프로젝트로 진행한 기계학습 모델링 프로젝트

사진에서 사람의 얼굴을 인식 한 뒤 그 얼굴이 남성인지 여성인지, 나이대가 20~30대, 40~50대, 60~대 인지를 파악하는 모델.

이 프로젝트에는 tensorflow를 사용한 모델들이 업로드 되어있다.

현재 진행중...

## 현재상황

성별인식모델 정확도: 86%
나이대인식모델 정확도: 76%

## 해야 할 것

1. ~~기본모델생성~~ **완료**
2. ~~주름영역을 추출하여 조합한 이미지를 데이터로 사용하기~~ **완료** -> 나이대인식에 더 적합하여 그곳에만 적용
4. ~~validation set생성~~ **완료**
5. ~~앙상블적용~~ **완료**
6. 모델을 사용하기 위해 weights 저장
7. 저장된 weights를 불러와 사진 입력, 결과 출력 프로그램에서 사용

## 참고자료
tensorflow 홈페이지 튜토리얼: <https://tensorflow.org/get_started/>
tensorflow 홈페이지 프로그래머 가이드: <https://tensorflow.org/programmers_guide/>
기초 강의: <https://hunkim.github.io/ml/>
