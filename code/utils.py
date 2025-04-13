"""
公共工具模块（需新建）
"""
import numpy as np
from model import ThreeLayerNet

def load_data():

    def unpickle(file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    # 加载训练数据（前4个batch）
    train_data = []
    train_labels = []
    for i in range(1, 5):
        data_dict = unpickle(f'data/cifar-10-batches-py/data_batch_{i}')
        train_data.append(data_dict[b'data'])
        train_labels += data_dict[b'labels']
    
    # 加载验证数据（第5个batch）
    val_data_dict = unpickle('data/cifar-10-batches-py/data_batch_5')
    X_val = val_data_dict[b'data'].astype(np.float32)
    y_val = np.array(val_data_dict[b'labels'])

    # 合并训练数据
    X_train = np.concatenate(train_data, axis=0).astype(np.float32)
    y_train = np.array(train_labels)

    # 数据预处理
    def preprocess(x):
        # 归一化到0-1范围
        x /= 255.0
        # 标准化（均值为0，标准差为1）
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        return (x - mean) / (std + 1e-7)
    
    # 应用预处理
    X_train = preprocess(X_train)
    X_val = preprocess(X_val)

    # 将数据形状转换为 (N, 3072)
    X_train = X_train.reshape(-1, 32 * 32 * 3)
    X_val = X_val.reshape(-1, 32 * 32 * 3)

    return X_train, y_train, X_val, y_val


def compute_accuracy(model, X, y):
    scores = model.forward(X)[0]
    preds = np.argmax(scores, axis=1)
    return np.mean(preds == y)