import numpy as np
import itertools
import json
from model import ThreeLayerNet
from train import load_data  # 复用数据加载

# 超参数搜索空间（根据实验需要调整范围）
HYPERPARAMS = {
    "hidden_size": [64, 128, 256],
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "reg_strength": [0.001, 0.01, 0.1],
    "activation": ["relu"],
    "batch_size": [64]
}
# HYPERPARAMS = {
#     "hidden_size": [128, 256, 512],
#     "learning_rate": [3e-4, 1e-4, 5e-5],
#     "reg_strength": [0.0005, 0.001, 0.005],
#     "activation": ["relu", "sigmoid"]
# }
# HYPERPARAMS = {
#     "hidden_size": [128],
#     "learning_rate": [1e-3],
#     "reg_strength": [ 0.001],
#     "activation": ["relu"],
#     "batch_size": [64] 
# }

def run_experiment(config):
    """单个超参数组合的训练实验"""
    model = ThreeLayerNet(
        input_size=3072,
        hidden_size=config["hidden_size"],
        output_size=10,
        activation=config["activation"]
    )
    
    X_train, y_train, X_val, y_val = load_data()
    
    batch_size = config["batch_size"]
    best_acc = 0.0
    epochs = 50
    lr=config['learning_rate']
    reg=config['reg_strength']
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
        val_acc = []
        # 每个epoch结束后验证
        val_pred = model.forward(X_val)[0].argmax(axis=1)
        acc = np.mean(val_pred == y_val)
        val_acc.append(acc)
        
        if acc > best_acc:
            best_acc = acc

        
        print(f"Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}")
    
    return {"best_val_acc": best_acc, "config": config}

def grid_search():
    """网格搜索主函数"""
    results = []
    keys = HYPERPARAMS.keys()
    values = itertools.product(*HYPERPARAMS.values())
    tmp = list(enumerate(values))

    for idx, params in tmp:
        config = dict(zip(keys, params))
        print(f"\nRunning experiment {idx+1}/{len(list(tmp))}")
        print("Current config:", config)
        result = run_experiment(config)
        results.append(result)
        
        # 实时保存结果
        with open("reports/hyper_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # 找出最佳配置
    best = max(results, key=lambda x: x["best_val_acc"])
    print("\n=== Best Configuration ===")
    print(json.dumps(best, indent=2))
    return best

if __name__ == "__main__":
    grid_search()