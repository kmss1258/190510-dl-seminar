# 190510-dl-seminar day 2

GPU 2장으로 학습?
-------------

> Layer를 병렬로 처리 가능하다고 하는데 잘 모르겠음.

ensemble method on test dataset
-------------

> 네트워크를 여러개를 실행 시키면 못 푸는 애들이 생기기 마련.    
> 따라서, 못 푸는 문제들의 가중치에 대해서 더 크게 주면 다음에 그 문제들을 풀려고 더 많이 달려든다고 함.    
> 요즘은 근데 잘 쓰지 않는 추세라고 함.

현재 메타
--------

> Skip Connection / Residual Connection    


현재 DL 네트워크
------

> Deep reinforcement learning (DRL)    
> 변화하는 환경을 풀기위한 분야

> GAN    
> 데이터가 제한적인 환경에서 Fake 데이터를 주며 학습

> AutoML    
> 자기의 구조를 탐색하면서 효율적인 네트워크를 재구성하는 네트워크

> Explainable artificial intelligence (xAI)    
> 블랙박스인 DL 네트워크 은닉층을 이해하려는 연구

# 190510 김태영 대표님 강의 Day 2

CNN Handwriting 데이터 인식 + Image Generator
---------

```
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# 랜덤시드 고정시키기
np.random.seed(3)

# 1. 데이터 생성하기
train_datagen = ImageDataGenerator(rescale=1./255) 
 # 이렇게 0~255로 분포하는 픽셀 값을 Resizing 해주는 모습을 자주 볼 수 있는데,
 # 이는 Deep Learning이 처리하기 제일 좋은 데이터 값으로 표현하기 위함이라고 한다.
 # 
train_generator = train_datagen.flow_from_directory(
        'handwriting_shape/train',
        target_size=(24, 24),
        batch_size=3,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255) 

test_generator = test_datagen.flow_from_directory(
        'handwriting_shape/test',
        target_size=(24, 24),    
        batch_size=3,
        class_mode='categorical') # add 이런 식으로 데이터를 정의할게.

# 2. 모델 구성하기
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24,24,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit_generator(
        train_generator,
        steps_per_epoch=15,
        epochs=50,
        validation_data=test_generator,
        validation_steps=5) # add fit 이 아니라 fit_generator를 넣고, 테스트 이미지에 generator를 넣어준다고 함.

# 5. 모델 평가하기
print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 6. 모델 사용하기
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)
```

Keras Callback?
------

> Keras.callbacks.ModelCheckPoint 과 같이 정의되어 있는 모델이 존재하는데, 


Paperswithcode.com
-----------

> 논문에 있는거를 코드로 구현한 것. 와우....    
> dataset도 다량 존재.    
> 데이터셋도 



교차검증 in Keras?
-------

> k-fold cross validation 검색 키워드.


미리 학습해둔 모델 load
--------

> top_model.load_weights..... 외국 keras 블로그에 있던 듯.
> 사이트 추천


LSTM
----------

> 동영상 타입에 대해서는 RNN 계열에서 큰 빛을 발함.



강화학습
--------

> 알파고도 강화학습이다.    
> 강화학습을 시킬 때, 보상을 잘 주어야지 원하는 방향으로 학습이 된다.

추천 사이트
-------

> Inspace


강조하시는 부분
-----------

> 출력이 여러 경우의 숫자이다?    
> activation func. --> 출력이 0~1    

> 28 * 28 이미지를 출력?    


GAN 네트워크 요약
------

> A 네트워크와 이와 대응되는 GAN 네트워크가 존재한다고 가정.    
> 1. A 네트워크는 진짜 이미지를 보고 정답(0 or 1)을 말할 수 있음
> 2. GAN 네트워크는 노이즈를 입력으로 받아 가짜 이미지를 출력을 해줌
> 3. 그 출력을 주면 모델에 대한 답도 0을 줌. A 네트워크는 점점 학습을 하면서 출력을 0을 주며, 가짜 이미지를 분류 할 수 있게 됨
> 4. 그다음, A 모델을 GAN 네트워크에 (concanate)붙이고 A 네트워크가 바뀔 수 없도록 자물쇠로 잠궈놓음.
> 5. 그리고 똑같이 GAN 네트워크에 노이즈를 입력을 줘서 GAN - A 를 거쳐 답을 출력함. 출제자는 문제의 답을 1로 줌.
> 6. 어? A 는 0 출력 되도록 만든건데 왜 1이 출력돼?
> 7. 그러면 A네트워크는 자물쇠를 걸어놔 변화할 수 없으므로 GAN 네트워크가 학습됨.
> 8. 이 과정에서 GAN 네트워크는 좀 더 가짜 이미지를 잘 만들도록 학습되어짐. 와우.

```

# MNIST data를 받아다가 

```
