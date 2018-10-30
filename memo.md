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
- learning rate 점점 감소하게 함 0.01 -> 0.001 -> 0.0001
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
```
game_count 135, training_step 5
loss 4.66822, value 0.98730, entropy 3.60800
game_count 172, training_step 10
loss 4.61086, value 0.96736, entropy 3.57046
game_count 210, training_step 15
loss 4.61010, value 0.99647, entropy 3.54050
game_count 247, training_step 20
loss 4.54531, value 0.95911, entropy 3.51300
game_count 285, training_step 25
loss 4.48798, value 0.95131, entropy 3.46343
game_count 324, training_step 30
loss 4.29920, value 0.85364, entropy 3.37230
game_count 364, training_step 35
loss 4.09460, value 0.70327, entropy 3.31805
game_count 404, training_step 40
loss 3.97343, value 0.64113, entropy 3.25900
game_count 447, training_step 45
loss 3.91378, value 0.72358, entropy 3.11687
game_count 534, training_step 55
loss 3.85677, value 0.80301, entropy 2.98036
game_count 696, training_step 75
loss 3.85158, value 0.83880, entropy 2.93920
game_count 738, training_step 80
loss 3.77227, value 0.80660, entropy 2.89205
game_count 777, training_step 85
loss 3.74161, value 0.80983, entropy 2.85811
game_count 819, training_step 90
loss 3.73825, value 0.81922, entropy 2.84532
game_count 863, training_step 95
loss 3.71636, value 0.81110, entropy 2.83151
game_count 906, training_step 100
loss 3.66868, value 0.80030, entropy 2.79459
game_count 994, training_step 110
loss 3.62700, value 0.90001, entropy 2.65311
game_count 1035, training_step 115
loss 3.55078, value 0.80585, entropy 2.67100
game_count 1072, training_step 120
loss 3.53114, value 0.79295, entropy 2.66422
game_count 1150, training_step 130
loss 3.46996, value 0.73280, entropy 2.66310
game_count 1792, training_step 210
loss 3.45234, value 0.78075, entropy 2.59680
game_count 1836, training_step 215
loss 3.35293, value 0.75857, entropy 2.51951
game_count 1919, training_step 225
loss 3.34285, value 0.82701, entropy 2.44089
game_count 2003, training_step 235
loss 3.27935, value 0.72771, entropy 2.47658
game_count 2048, training_step 240
loss 3.23658, value 0.75521, entropy 2.40624
game_count 2093, training_step 245
loss 3.21838, value 0.72481, entropy 2.41839
game_count 2130, training_step 250
loss 3.18399, value 0.77112, entropy 2.33764
game_count 2174, training_step 255
loss 3.11733, value 0.71651, entropy 2.32554
game_count 2303, training_step 270
loss 3.11182, value 0.69857, entropy 2.33779
game_count 2342, training_step 275
loss 3.09625, value 0.65679, entropy 2.36396
game_count 2669, training_step 315
loss 3.07570, value 0.65557, entropy 2.34423
```
![66_1_loss](https://github.com/Jhyeok-lee/alphago/blob/develop/img/66_1_loss.png)
![66_1_entropy](https://github.com/Jhyeok-lee/alphago/blob/develop/img/66_1_entropy.png)
![66_1_mse](https://github.com/Jhyeok-lee/alphago/blob/develop/img/66_1_mse.png)
- 총 걸린시간 : 16시간 23분
- 총 Training Step : 375
- 총 Game Count : 3150 게임
- Entropy는 지속적으로 떨어지지만 MSE는 너무 요동친다
- MSE를 weight를 주어서 다시

## 혹시나 싶어서 해봤더니....
- residual_layer = 1
- width = 6
- height = 6
- max_state_size = 3
- win_contition = 4
- batch_size = 128
- max_game_count = 300000
- max_data_size = 1280
- max_training_loop_count = 1
- learning_rate = 0.001
- simulation_count = 400
- c_puct = 0.96
- data queue 사이즈, batch 사이즈 줄임
- loop도 한번씩 -> 꽤 괜찮은 모델 나옴
- 걸린시간 : 3시간
- 총 Training Setp : 246
- Game Count : 620게임

![66_2_loss](https://github.com/Jhyeok-lee/alphago/blob/develop/img/66_2_loss.png)
![66_2_entropy](https://github.com/Jhyeok-lee/alphago/blob/develop/img/66_2_entropy.png)
![66_2_mse](https://github.com/Jhyeok-lee/alphago/blob/develop/img/66_2_mse.png)
```
game_count 18, training_step 1
loss 5.29170, value 1.25461, entropy 4.01535
game_count 20, training_step 2
loss 4.91588, value 1.06485, entropy 3.82929
game_count 22, training_step 3
loss 4.82607, value 0.94318, entropy 3.86114
game_count 24, training_step 4
loss 4.80979, value 1.00080, entropy 3.78724
game_count 28, training_step 6
loss 4.69615, value 1.02100, entropy 3.65340
game_count 30, training_step 7
loss 4.64573, value 0.94957, entropy 3.67441
game_count 40, training_step 12
loss 4.63818, value 0.97761, entropy 3.63879
game_count 60, training_step 21
loss 4.56001, value 0.98285, entropy 3.55534
game_count 64, training_step 23
loss 4.53869, value 0.99215, entropy 3.52472
game_count 84, training_step 32
loss 4.51727, value 1.00328, entropy 3.49214
game_count 86, training_step 33
loss 4.44219, value 0.95407, entropy 3.46626
game_count 102, training_step 39
loss 4.43562, value 0.98405, entropy 3.42969
game_count 110, training_step 42
loss 4.34380, value 0.90906, entropy 3.41286
game_count 113, training_step 43
loss 4.33314, value 0.89521, entropy 3.41605
game_count 115, training_step 44
loss 4.31875, value 0.89770, entropy 3.39917
game_count 117, training_step 45
loss 4.22920, value 0.78872, entropy 3.41859
game_count 120, training_step 46
loss 4.18167, value 0.78941, entropy 3.37036
game_count 123, training_step 47
loss 4.14080, value 0.76289, entropy 3.35601
game_count 126, training_step 48
loss 4.00387, value 0.69417, entropy 3.28781
game_count 131, training_step 50
loss 3.97380, value 0.63557, entropy 3.31633
game_count 161, training_step 62
loss 3.95317, value 0.83337, entropy 3.09785
game_count 334, training_step 131
loss 3.88946, value 0.76969, entropy 3.09763
game_count 336, training_step 132
loss 3.87968, value 0.81872, entropy 3.03881
game_count 343, training_step 135
loss 3.87664, value 0.87083, entropy 2.98366
game_count 440, training_step 173
loss 3.81153, value 0.70117, entropy 3.08818
game_count 549, training_step 213
loss 3.67016, value 0.71327, entropy 2.93468
game_count 616, training_step 238
loss 3.66435, value 0.66526, entropy 2.97682
game_count 623, training_step 241
loss 3.61568, value 0.64963, entropy 2.94379
```