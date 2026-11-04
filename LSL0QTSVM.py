import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class LS_L0_QTSVM:

    def __init__(self, C, s=None, t=0.001, gm=0.1, eps=1e-8, ro = 0.1):
        self.C = C
        self.s = s
        self.t = t
        self.gm = gm
        self.eps = eps
        self.ro = ro
        self.zp_final = None
        self.cp_final = None
        self.zn_final = None
        self.cn_final = None
        self.preds = None

    def feature_mapping(self, X):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        return poly.fit_transform(X)

    def plane(self, Xp, Xn, yn, ro, u, k):
        if u is not None and u.size > 0:
            p = int(u.shape[0])
        else:
            p = 0
            if Xp.shape[0] > 0:
                p = self.feature_mapping(Xp).shape[1]
            elif Xn.shape[0] > 0:
                p = self.feature_mapping(Xn).shape[1]
            if p == 0:
                raise ValueError("feature augmentation error.")

        if Xp.shape[0] == 0 or Xn.shape[0] == 0:
            return np.zeros((p, 1), dtype=float), 0.0

        R_k = self.feature_mapping(Xp).astype(float, copy=False)
        R_neg_k = self.feature_mapping(Xn).astype(float, copy=False)

        n_k = R_k.shape[0]
        n_neg_k = R_neg_k.shape[0]

        ones_k = np.ones((n_k, 1), dtype=float)
        ones_neg_k = np.ones((n_neg_k, 1), dtype=float)

        y_neg_k_col = yn.reshape(-1, 1).astype(float, copy=False)
        y_neg_k_col = np.sign(y_neg_k_col)
        u = u.reshape(p, 1).astype(float, copy=False)

        A11 = (R_k.T @ R_k) + self.C * (R_neg_k.T @ R_neg_k) + ro * np.eye(p, dtype=float)
        A12 = (R_k.T @ ones_k) + self.C * (R_neg_k.T @ ones_neg_k)
        A21 = A12.T
        A22 = np.array([[n_k + self.C * n_neg_k]], dtype=float)

        A_k = np.block([[A11, A12], [A21, A22]])

        b1 = self.C * (R_neg_k.T @ y_neg_k_col) + ro * u
        b2 = np.array([[self.C * float(k) * n_neg_k]], dtype=float)

        b_k = np.vstack([b1, b2])

        try:
            cond_A = np.linalg.cond(A_k)
            if np.isfinite(cond_A) and cond_A <= (1.0 / self.eps):
                sol = np.linalg.solve(A_k, b_k)
            else:
                A_reg = A_k + self.eps * np.eye(A_k.shape[0], dtype=float)
                sol = np.linalg.solve(A_reg, b_k)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A_k, b_k, rcond=None)

        z = sol[:-1]
        c = float(sol[-1].reshape(-1)[0])
        return z, c

    def update_u(self, u, D, z):
        if self.s is not None and self.s > 0 and self.s < D:
            abs_z = np.abs(z.flatten())
            if len(abs_z) > 0:
                abs_z_sorted = np.sort(abs_z)
                threshold = abs_z_sorted[max(0, len(abs_z_sorted) - self.s)] if len(abs_z_sorted) > 0 else 0.0
                u = z * (abs_z.reshape(-1, 1) >= threshold)
            else:
                u = z
        else:
            u = z
        return u

    def fit(self, X, y):
        if X.shape[0] == 0:
            return
        y = y.reshape(-1, 1)

        Xp = X[y.flatten() == 1]
        yp = y[y.flatten() == 1]
        Np = Xp.shape[0]
        Dp = self.feature_mapping(Xp).shape[1]
        rp = self.ro
        up = np.zeros((Dp, 1))
        zpp = np.zeros((Dp, 1))
        cpp = 0.0

        Xn = X[y.flatten() == -1]
        yn = y[y.flatten() == -1]
        Nn = Xn.shape[0]
        Dn = self.feature_mapping(Xn).shape[1]
        rn = self.ro
        un = np.zeros((Dn, 1))
        znp = np.zeros((Dn, 1))
        cnp = 0.0

        for j in range(20):
            zpp_org, cpp_org = zpp, cpp
            znp_org, cnp_org = znp, cnp

            for l in range(20):
                zp, cp = self.plane(Xp, Xn, yn, rp, up, 1)
                up = self.update_u(up, Dp, zp)
                if (np.linalg.norm(zp - zpp) < self.t and abs(cp - cpp) < self.t):
                    break
                zpp, cpp = zp, cp

            for l in range(20):
                zn, cn = self.plane(Xn, Xp, yp, rn, un, -1)
                un = self.update_u(un, Dn, zn)
                if (np.linalg.norm(zn - znp) < self.t and abs(cn - cnp) < self.t):
                    break
                znp, cnp = zn, cn

            self.zp_final, self.cp_final, self.up = zp, cp, up
            self.zn_final, self.cn_final, self.un = zn, cn, un

            rp *= self.gm
            rn *= self.gm

            if (np.linalg.norm(self.zp_final - zpp_org) < self.t and
                abs(self.cp_final - cpp_org) < self.t and
                np.linalg.norm(self.zn_final - znp_org) < self.t and
                abs(self.cn_final - cnp_org) < self.t):
                break

    def predict(self, X_test):

        if (self.zp_final is None or self.cp_final is None or
            self.zn_final is None or self.cn_final is None):
            raise ValueError("use .fit() first")
        
        if X_test.shape[0] == 0:
            return np.array([]).reshape(-1, 1)

        r_x_test = self.feature_mapping(X_test)

        f1_val = np.dot(r_x_test, self.zp_final) + self.cp_final
        f2_val = np.dot(r_x_test, self.zn_final) + self.cn_final

        y_pred = np.zeros((X_test.shape[0], 1))
        
        for i in range(X_test.shape[0]):
            if np.abs(f1_val[i]) < np.abs(f2_val[i]):
                y_pred[i][0] = 1
            else:
                y_pred[i][0] = -1

        self.preds = y_pred
        return y_pred
    
    def get_params(self):
        return self.zp_final, self.cp_final, self.zn_final, self.cn_final
    
    def get_preds(self):
        return self.preds
    
