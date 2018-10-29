# 1. 6x6 4목으로 시작
- 시간이 너무 오래 걸림 -> MCTS가 시간을 대부분 잡아먹음
- MCTS 부분 C++로 바꿔야하는건 아닌지?
- 게임을 축소시켜서 확인
- deepcopy가 시간을 많이 잡아먹는 줄 알고 State 내부 변수만 각각 복사했더니 시간 더 걸림

## 모델 변화
- MSE가 너무 안떨어져서 Network를 통째로 바꿈 -> Alphago Zero와 완전 같음
- Input 데이터 차원 변경

## Residual Layer 개수 정하기 & 기타 파라메터 정하기
- 1개부터 늘려봄
- Overfit이 일어나도록 해서 Value MSE와 Policy Entropy가 어디까지 떨어지는지 확인
- 5개 일 때, MSE가 제일 낮음
- Batch Size가 되는데로 training loop 5로 잡아서 돌려봄
- 설마해서 게임 확인하니 의외로 방어도 하고 공격도 함
- 근데 인간한테 안됨
```
game_count 3, training_step 5
loss 3.57405, value 0.28180, entropy 3.21928
game_count 4, training_step 10
loss 3.42219, value 0.32349, entropy 3.02554
game_count 5, training_step 15
loss 3.22020, value 0.27907, entropy 2.86780
game_count 6, training_step 20
loss 3.15879, value 0.18118, entropy 2.90412
game_count 8, training_step 30
loss 3.06090, value 0.16736, entropy 2.81975
game_count 9, training_step 35
loss 2.94440, value 0.13941, entropy 2.73105
game_count 11, training_step 45
loss 2.91567, value 0.13676, entropy 2.70471
game_count 13, training_step 55
loss 2.83253, value 0.08056, entropy 2.67755
game_count 15, training_step 65
loss 2.78326, value 0.13716, entropy 2.57146
game_count 17, training_step 75
loss 2.77955, value 0.11594, entropy 2.58877
```

## 일단 돌려보기
- game data queue가 가득 채울때 까지 게임 돌림 -> 대략 135게임 돌리면 10000개 나옴
- training loop = 5, batch size = 512 -> 랜덤으로 2500개 정도 학습(전체의 25%)
- 새로운 게임 데이터가 25% 만들어지면 학습 -> 대략 40게임 돌리면 2500개 나옴
- 알파고 제로의 경우 : 500,000 게임 마다 2048 batch size 1000 loop
	- 게임 데이터 다 갈아치우고 한 게임당 200개 나오면 4% 학습함