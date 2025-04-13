import numpy as np
import matplotlib.pyplot as plt
from model import ThreeLayerNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# 数据加载和预处理
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

def train():
    # 超参数设置
    hidden_size = 256
    lr = 1e-3
    reg = 0.01
    epochs = 20
    batch_size = 64
    
    # 加载并预处理数据
    X_train, y_train, X_val, y_val = load_data()  # 需要实现数据加载
    
    # 初始化模型
    model = ThreeLayerNet(input_size=3072, hidden_size=hidden_size, output_size=10, activation='relu')
    
    # 训练循环
    best_acc = 0.0
    train_loss = []
    val_loss = []
    val_acc = []
    
    for epoch in range(epochs):
        # Mini-batch 训练
        for i in range(0, X_train.shape[0], batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # 前向传播
            scores, h1 = model.forward(batch_X)
            
            # 计算交叉熵损失
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            loss = -np.log(probs[range(batch_X.shape[0]), batch_y]).mean()
            
            # 反向传播
            dscores = probs.copy()
            dscores[range(batch_X.shape[0]), batch_y] -= 1
            dscores /= batch_X.shape[0]
            
            grads = model.backward(batch_X, h1, dscores)
            
            # 参数更新（SGD）
            for param in model.params:
                model.params[param] -= lr * grads[param]
                # L2正则化
                if 'W' in param:
                    model.params[param] -= lr * reg * model.params[param]
        
        # 每个epoch结束后验证
        # 训练集损失
        scores, h1 = model.forward(X_train)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = -np.log(probs[range(X_train.shape[0]), y_train]).mean()
        train_loss.append(loss)
        # 验证集损失+准确率
        scores, h1 = model.forward(X_val)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = -np.log(probs[range(X_val.shape[0]), y_val]).mean()
        val_loss.append(loss)

        val_pred = scores.argmax(axis=1)
        acc = np.mean(val_pred == y_val)
        val_acc.append(acc)
        
        # 保存最佳模型y_train
        if acc > best_acc:
            best_acc = acc
            with open('weights/best_model.pkl', 'wb') as f:
                pickle.dump(model.params, f)
        
        print(f"Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}")
    
    # 绘制训练曲线
    plt.figure()
    plt.plot(val_acc)
    plt.title('Validation Accuracy')
    plt.savefig('reports/val_acc.png')

    plt.figure()
    plt.plot(val_loss)
    plt.title('Validation Loss')
    plt.savefig('reports/val_loss.png')

    plt.figure()
    plt.plot(train_loss)
    plt.title('Training Loss')
    plt.savefig('reports/train_loss.png')
    
if __name__ == '__main__':
    train()