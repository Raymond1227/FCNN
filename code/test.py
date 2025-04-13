import numpy as np
import pickle
from model import ThreeLayerNet
from train import load_data

def test():
    # 加载测试数据
    _, _, X_test, y_test = load_data()
    
    # 加载模型
    with open('weights/best_model.pkl', 'rb') as f:
        params = pickle.load(f)
    
    model = ThreeLayerNet(3072, 128, 10)
    model.params = params
    
    # 计算测试集准确率
    scores = model.forward(X_test)[0]
    preds = np.argmax(scores, axis=1)
    acc = np.mean(preds == y_test)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == '__main__':
    test()