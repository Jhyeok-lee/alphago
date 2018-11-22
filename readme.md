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

## 프로젝트 일지
[여기로](https://github.com/Jhyeok-lee/alphago/blob/develop/memo.md)

## 6x6 4목 2번째 시도
![66_2](https://github.com/Jhyeok-lee/alphago/blob/develop/img/66_2.gif)

## 구조

## Reference