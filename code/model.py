import numpy as np

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.params = {}
        self.params['W1'] = np.sqrt(2. / input_size) * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.sqrt(2. / hidden_size) * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.activation = activation.lower()
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        # 第一层
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        if self.activation == 'relu':
            h1 = self.relu(a1)
        elif self.activation == 'sigmoid':
            h1 = 1 / (1 + np.exp(-a1))
        
        # 第二层（输出层）
        scores = np.dot(h1, self.params['W2']) + self.params['b2']
        return scores, h1
    
    def backward(self, x, h1, grad_output):
        grads = {}
        # 输出层梯度
        grads['W2'] = np.dot(h1.T, grad_output)
        grads['b2'] = np.sum(grad_output, axis=0)
        
        # 隐藏层梯度
        dh1 = np.dot(grad_output, self.params['W2'].T)
        if self.activation == 'relu':
            da1 = dh1 * self.relu_deriv(h1)
        elif self.activation == 'sigmoid':
            da1 = dh1 * h1 * (1 - h1)
        
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)
        return grads