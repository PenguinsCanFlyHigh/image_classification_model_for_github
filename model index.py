import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 데이터셋 경로 설정
data_dir = "C:/programming/Image_Classification_Model/new_train_data"

# 이미지 전처리 및 데이터 로더 설정
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,  # 80% 학습, 20% 검증 데이터로 나누기
    rotation_range=30,  # 이미지 회전 범위
    width_shift_range=0.2,  # 가로 방향으로 이동 범위
    height_shift_range=0.2,  # 세로 방향으로 이동 범위
    shear_range=0.2,  # 전단 변환 범위
    zoom_range=0.2,  # 이미지 확대 범위
    horizontal_flip=True,  # 수평 반전
    fill_mode='nearest'  # 빈 공간을 채우는 방식
)

# 검증 데이터 전처리 (증강 없이, 단순히 리스케일)
validation_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(224, 224),  # ResNet50의 입력 크기
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 클래스 웨이트 계산
class_labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
class_weights_dict = dict(enumerate(class_weights))

print(train_generator.class_indices)
