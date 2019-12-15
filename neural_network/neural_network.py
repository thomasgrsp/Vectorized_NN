import numpy as np

class NeuralNet():
   
    def __init__(self):
        self.max_size = 3
        self.a = [None] * self.max_size  # layer output (after activation) = a(z) =a(weights * input)
        self.w = [None] * self.max_size  # weights
        self.z = [None] * self.max_size  # weights * input
        self.b = [None] * self.max_size  # biases
        self.a_func = []
        self.num_layer = 0
        self.m = None
   
    def add_layer(self, n, activ, input_dim=(None, None)):
        self.a_func.append(activ)
        if self.w[0] is None:
            self.m = input_dim[0]
            self.w[0] = np.random.randn(n, input_dim[1]) * \
                np.sqrt(1/input_dim[1])
        else:
            self.w[self.num_layer] = np.random.randn(n, self.w[self.num_layer-1].shape[0]) * \
                np.sqrt(1/self.w[self.num_layer-1].shape[0])
        self.b[self.num_layer] = np.ones((n, 1))#np.random.randn(n, 1)
        self.num_layer += 1
       
    def forward_pass(self, X):
        for i in range(self.num_layer):
            self.z[i] = np.dot(self.w[i], self.a[i]) + self.b[i]
            self.a[i+1] = self.a_func[i](self.z[i])

    def compute_cost(self, y):
        # CrossEntropy (2 classes)
        return (-1/self.m) * np.sum(y * np.log(self.a[self.num_layer]))
    
    def backward_pass(self, X, y, lr=0.05):
       
        # Compute gradients wrt. weights, biases
        dw = [None] * len(self.w)
        db = [None] * len(self.b)
        g = self.a[self.num_layer] - y
        for l in reversed(range(self.num_layer)):
            if l != self.num_layer-1:
                g = np.multiply(g, self.a_func[l](self.z[l], derivative=True))
            dw[l] = (1/self.m) * np.dot(g, self.a[l].T)
            db[l] = (1/self.m) * np.sum(g, axis=1, keepdims=True)
            g = np.dot(self.w[l].T, g)

        # SGD (Stochastic Gradient Descent)
        for l in range(self.num_layer):
            self.b[l] = self.b[l] - lr * db[l]
            self.w[l] = self.w[l] - lr * dw[l]

    def train(self, X, y, epochs=10, lr=0.05):
        loss = []
        self.a[0] = X
        for i in range(epochs):
            self.forward_pass(X)
            self.backward_pass(X, y, lr=lr)
            loss.append(self.compute_cost(y))
        return loss
    
    def predict(self, X):
        for i in range(self.num_layer):
            if i==0:
                z = np.dot(self.w[0], X) + self.b[0]
            else:
                z = np.dot(self.w[i], a) + self.b[i]
            a = self.a_func[i](z)
        prediction = np.array([[0, 1] if i<=0.5 else [1, 0] for i, j in a.T])
        return prediction