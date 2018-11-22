# 알파고 제로 + 오목

## 구조
#### Game State
![state](https://github.com/Jhyeok-lee/alphago/blob/develop/img/state.png)
- 학습 또는 네트워크 계산에 필요한 Input State
- 게임 보드 N X N의 matrix를 중첩한 구조
	- 현재 플레이어
	- White의 최근 K 개의 상태
	- Black의 최근 K 개의 상태

#### Agent
- Collecting Game Data
	- Self-Play를 한 결과로 Game State, Winner, MCTS-Propability를 모음
	- Augmenting data : 회전, 뒤집기 적용
	- Data queue : Augmenting한 data를 저장
- Training
	- 새로운 데이터 5120 개가 생겨날 때 마다 학습
	- Batch size : 256
	- Training Rate : 0.001
	- Training Loop : 5회
- Validation Test
	- Training이 끝나고 Best Model과 Current Model간의 대결 테스트

#### Model
![model](https://github.com/Jhyeok-lee/alphago/blob/develop/img/model.png)
- Input Layer:
	- 64개의 채널의 3x3 커널크기를 가진 Convolution Layer
	- Batch Normalization
	- A rectifier linear unit

- Residual Block:
	- 64개의 채널의 3x3 커널크기를 가진 Convolution Layer
	- Batch Normalization
	- A rectifier linear unit
	- 64개의 채널의 3x3 커널크기를 가진 Convolution Layer
	- Batch Normalization
	- Residual Layer의 Input 추가
	- A rectifier linear unit

- Policy Head
	- 2개의 채널의 1x1 커널크기를 가진 Convolution Layer
	- Batch Normalization
	- A rectifier linear unit
	- 보드의 너비 X 높이 개의 output을 가지는 Fully Connected Layer

- Value Head
	- 1개의 채널의 1x1 커널크기를 가진 Convolution Layer
	- Batch Normalization
	- A rectifier linear unit
	- 64 개의 output을 가지는 Fully Connected Layer
	- A rectifier linear unit
	- [-1, 1]의 결과를 output을 가지는 tanh

- Loss Function
![loss](https://github.com/Jhyeok-lee/alphago/blob/develop/img/cost_function.png)

## 프로젝트 일지
[여기로](https://github.com/Jhyeok-lee/alphago/blob/develop/memo.md)

## 6x6 4목 2번째 시도
![66_2](https://github.com/Jhyeok-lee/alphago/blob/develop/img/66_2.gif)

