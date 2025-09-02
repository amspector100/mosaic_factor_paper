import numpy as np
import cvxpy as cp
from typing import Optional
import scipy.linalg
from tqdm.auto import tqdm

def clip_singular_values(A, r=0.99):
    """
    Return A_svd whose singular values are clipped to <= r.
    Guarantees ||A_svd||_2 <= r, hence ||A_svd^k||_F <= sqrt(n) * r^k.
    """
    U, s, Vh = scipy.linalg.svd(A, full_matrices=False)
    s_clipped = np.minimum(s, r)
    return (U * s_clipped) @ Vh  # U @ diag(s_clipped) @ Vh

class NonnegRegression:

    def __init__(self, X, y, loss='abs'):
        """
        Fits the model y = X @ beta + omega, where beta, omega are nonnegative.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable
        loss : str, optional
            Loss function to use, by default 'abs'
            'abs': absolute loss
            'square': squared loss
            'huber': huber loss
        """
        self.X = X
        self.y = y
        self.loss = loss

    def compute_loss(self, resids: np.array, return_np_object: bool=False):
        if self.loss == 'abs':
            output = cp.mean(cp.abs(resids))
        elif self.loss == 'square':
            output = cp.mean(cp.square(resids))
        elif self.loss == 'huber':
            output = cp.mean(cp.huber(resids))
        else:
            raise ValueError(f"Loss {self.loss} not supported")

        if return_np_object:
            return output.value
        else:
            return output
        
    def _single_fit(
        self, 
        X: Optional[np.array]=None,
        y: Optional[np.array]=None,
        lmda: float=0.1,
    ):
        # defaults
        y = self.y if y is None else y
        X = self.X if X is None else X
        # instantiate variables
        beta = cp.Variable(X.shape[1])
        omega = cp.Variable()
        # predictions
        preds = X @ beta + omega
        objective = self.compute_loss(y - preds) + lmda * cp.sum(beta**2)
        constraints = [beta >= 0, omega >= 0]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        return beta.value, omega.value
    
    def fit(self, lmda: float=0.1):
        self.beta, self.omega = self._single_fit(lmda=lmda)

    def compute_cv_preds(self, lmdas: np.array=[0.01, 0.1, 1.0], n_folds: int=5) -> np.array:
        folds = np.array_split(np.arange(len(self.y)), n_folds)
        cv_preds = np.zeros((len(self.y), len(lmdas)))
        for lmda_no, lmda in enumerate(lmdas):
            for fold in folds:
                # train on neg_fold
                neg_fold = [i for i in np.arange(len(self.y)) if i not in fold]
                beta, omega = self._single_fit(X=self.X[neg_fold], y=self.y[neg_fold], lmda=lmda)
                # predict on fold
                fold_preds = self.X[fold] @ beta + omega
                cv_preds[:, lmda_no][fold] = fold_preds
        return cv_preds

    def fit_cv(self, lmdas: np.array=[0.01, 0.1, 1.0], n_folds: int=5):
        self.lmdas = lmdas
        cv_preds = self.compute_cv_preds(lmdas=lmdas, n_folds=n_folds)
        self.cv_preds = cv_preds
        self.cv_losses = np.array([self.compute_loss(self.y - cv_preds[:, i], return_np_object=True) for i in range(len(lmdas))])
        self.best_lmda = lmdas[np.argmin(self.cv_losses)]
        self.fit(lmda=self.best_lmda)
        return self
    
    def predict(self, X: np.array):
        return X @ self.beta + self.omega