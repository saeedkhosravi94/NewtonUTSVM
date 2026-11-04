import numpy as np

class NewtonUTSVM:
    def __init__(self, X, y, U, C, eps):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        self.U = np.asarray(U, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.eps = eps
        
    def fit(self):
        np.random.seed(42)
        self.w1 = np.random.normal(0, 0.01, (self.X.shape[1], 1))
        self.b1 = 0.0
        self.w2 = np.random.normal(0, 0.01, (self.X.shape[1], 1))
        self.b2 = 0.0
        
        for _ in range(5):  
            self.w1, self.b1 = self.plane1(self.X, self.y, self.U, 
                                         self.C[0], self.C[1], self.C[2], self.eps)
            self.w2, self.b2 = self.plane2(self.X, self.y, self.U,
                                         self.C[3], self.C[4], self.C[5], self.eps)
        
    def predict(self, x_test):
        x_test = np.asarray(x_test, dtype=np.float64)
        
        dist1 = self._safe_distance(x_test, self.w1, self.b1)
        dist2 = self._safe_distance(x_test, self.w2, self.b2)
        
        y_pred = np.where(dist1 < dist2, 1, -1).reshape(-1, 1)
        self.preds = y_pred
        return y_pred
    
    def _safe_distance(self, X, w, b):
        norm = np.linalg.norm(w)
        if norm < 1e-10:
            return np.full((X.shape[0],), np.inf)
        return np.abs(X @ w + b) / norm
    
    def plane1(self, X, y, U, C1, C2, C3, eps):
        A = X[y[:,0] == 1]
        B = X[y[:,0] == -1]
        
        T1 = np.hstack([A, np.ones((A.shape[0], 1))])
        T2 = np.hstack([B, np.ones((B.shape[0], 1))])
        T3 = np.hstack([U, np.ones((U.shape[0], 1))])
        
        Z = np.random.normal(0, 0.01, (X.shape[1]+1, 1))
        prev_Z = np.zeros_like(Z)
        
        learning_rate = 0.1
        best_loss = float('inf')
        
        for count in range(100):
            e2 = np.ones((B.shape[0], 1))
            eu = np.ones((U.shape[0], 1))
            
            margin_B = e2 + T2 @ Z
            margin_U = (-1 + eps)*eu - T3 @ Z
            
            grad = (T1.T @ (T1 @ Z) + 
                   C1 * T2.T @ self.func(margin_B, 'pf') + 
                   C2 * Z - 
                   C3 * T3.T @ self.func(margin_U, 'pf'))
            
            D1 = self.mat_diag(self.func(margin_B, 'pf') > 0)
            D2 = self.func(margin_U, 'pf') > 0
            hessian = (T1.T @ T1 + 
                      C1 * T2.T @ D1 @ T2 + 
                      C2 * np.eye(Z.shape[0]) + 
                      C3 * T3.T @ np.diag(D2.flatten()) @ T3)
            
            hessian += 1e-4 * np.eye(hessian.shape[0])
            
            delta = np.linalg.solve(hessian, grad)
            Z -= learning_rate * delta
            
            current_loss = np.linalg.norm(grad)
            if current_loss < best_loss:
                best_loss = current_loss
                learning_rate = min(learning_rate * 1.1, 1.0)
            else:
                learning_rate = max(learning_rate * 0.5, 1e-4)
                
            if np.linalg.norm(Z - prev_Z) < self.eps:
                break
            prev_Z = Z.copy()
            
        return Z[:-1], Z[-1][0]
    
    def plane2(self, X, y, U, C4, C5, C6, eps):
        A = X[y[:,0] == 1]
        B = X[y[:,0] == -1]
        
        # Add bias terms
        G1 = np.hstack([B, np.ones((B.shape[0], 1))])
        G2 = np.hstack([A, np.ones((A.shape[0], 1))])
        G3 = np.hstack([U, np.ones((U.shape[0], 1))])
        
        Y = np.random.normal(0, 0.01, (X.shape[1]+1, 1))
        prev_Y = np.zeros_like(Y)
        
        learning_rate = 0.1
        best_loss = float('inf')
        
        for count in range(100):
            e1 = np.ones((A.shape[0], 1))
            eu = np.ones((U.shape[0], 1))
            
            margin_A = e1 - G2 @ Y
            margin_U = (-1 + eps)*eu + G3 @ Y
            
            grad = (G1.T @ (G1 @ Y) - 
                   C4 * G2.T @ self.func(margin_A, 'pf') + 
                   C5 * Y + 
                   C6 * G3.T @ self.func(margin_U, 'pf'))
            
            D3 = self.func(margin_A, 'pf') > 0
            D4 = self.func(margin_U, 'pf') > 0
            hessian = (G1.T @ G1 + 
                      C4 * G2.T @ np.diag(D3.flatten()) @ G2 + 
                      C5 * np.eye(Y.shape[0]) + 
                      C6 * G3.T @ np.diag(D4.flatten()) @ G3)
            
            hessian += 1e-4 * np.eye(hessian.shape[0])
            
            delta = np.linalg.solve(hessian, grad)
            Y -= learning_rate * delta
            
            current_loss = np.linalg.norm(grad)
            if current_loss < best_loss:
                best_loss = current_loss
                learning_rate = min(learning_rate * 1.1, 1.0)
            else:
                learning_rate = max(learning_rate * 0.5, 1e-4)
                
            if np.linalg.norm(Y - prev_Y) < self.eps:
                break
            prev_Y = Y.copy()
            
        return Y[:-1], Y[-1][0]
    
    def func(self, x, type='pf', ro=1e20):
        if type == 'pf':
            return np.maximum(0, x) 
        elif type == 'sm':
            return x + (1/ro)*np.log(1+np.exp(-ro*x))
    
    def mat_diag(self, m):
        return np.diag(m.flatten())
    
    def get_params(self):
        return self.w1, self.b1, self.w2, self.b2
    
    def get_preds(self):
        return self.preds