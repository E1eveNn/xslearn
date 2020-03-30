from models.base import *
from models.tree import DecisionTreeClassifier
import copy
import threading
from utils.split import sampling2d



class BaseAdaBoost(BaseModel):
    def __init__(self, base_estimator, loss, n_estimators, learning_rate, verbose=1):
        super(BaseAdaBoost, self).__init__(verbose=verbose)
        self.base_estimator = base_estimator
        self.loss = loss
        self.n_iter = n_estimators
        self.learning_rate = learning_rate



class AdaBoostClassifier(BaseAdaBoost):
    def __init__(self, base_estimator=DecisionTreeClassifier(), loss='linear', n_estimators=50, learning_rate=1):
        super(AdaBoostClassifier, self).__init__(base_estimator, loss, n_estimators, learning_rate)
        self.D = None
        self.base_estimator_list = [copy.deepcopy(self.base_estimator)] * n_estimators
        self.alphas = None



    def initial_params(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        m, n = X.shape
        self.D = np.zeros((self.n_iter, m))
        self.D[0, :] = 1 / m
        self.alphas = np.zeros((self.n_iter, 1))
        return X, y




    def fit(self, X, y):
        X, y = self.initial_params(X, y)
        for i in range(self.n_iter):
            t1 = time.time()
            estimator = self.base_estimator_list[i]
            weights = np.expand_dims(self.D[i], axis=1)
            new_X = X * weights
            new_y = y
            estimator.fit(new_X, new_y)
            pred = estimator.predict(new_X)
            errs = np.mean(pred != new_y)
            t2 = time.time()
            iter_time = t2 - t1
            self.iteration += 1
            acc = self.get_acc(y, pred)
            self.train_verbose(acc, None, iter_time)
            if errs == 0:
                break
            self.alphas[i] = np.log((1 - errs) / errs) * 0.5
            if i != self.n_iter - 1:
                D = weights * np.exp(- self.alphas[i] * new_y * pred)
                weights = D / np.sum(D)
                self.D[i + 1] = weights.T
        if self.verbose == 2:
            print()





    def predict(self, X, is_train=False, *args):
        if X.shape[0] > 1:
            pred = np.zeros((X.shape[0], 1))
            for i, x in enumerate(X):
                res = 0
                for j in range(self.n_iter):
                    res += self.learning_rate * self.alphas[j] * self.base_estimator_list[j].predict(x, is_train)
                pred[i] = res
        else:
            pred = 0
            for j in range(self.n_iter):
                pred += self.learning_rate * self.alphas[j] * self.base_estimator_list[j].predict(X, is_train)
        return pred if is_train else np.sign(pred)




class BaggingThread(threading.Thread):
    def __init__(self, estimator, X, y, bootstrap, max_samples, bootstrap_features, max_features, oob_score):
        threading.Thread.__init__(self)
        self.estimator = estimator
        self.X = X
        self.y = y
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.bootstrap_features = bootstrap_features
        self.max_features = max_features
        self.oob_score = oob_score


    def run(self):
        m, n = self.X.shape
        # sample sampling
        sampled_X, sampled_y, sample_idx = sampling2d(self.X, self.y, self.bootstrap, self.max_samples)
        # feature sampling
        sampled_X, sampled_y, feature_idx = sampling2d(sampled_X, sampled_y, self.bootstrap_features, self.max_features, axis=1)
        self.estimator.fit(sampled_X, sampled_y)
        if self.oob_score:
            valid_idx = [idx for idx in range(m) if idx not in sample_idx]
            valid_X = self.X[valid_idx, :]
            valid_y = self.y[valid_idx]



class BaseBagging(BaseModel):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, n_job=-1,verbose=0):
        super(BaseBagging, self).__init__()
        self.base_estimator = base_estimator
        self.n_iter = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.n_job = 1 if n_job == -1 else n_job
        self.verbose = verbose
        self.base_estimator_list = [copy.deepcopy(base_estimator)] * n_estimators
        self.thread_list = []


    def initial_params(self, X, y):
        assert self.n_job >= 1, 'n_job should be -1 or an integer!'
        X = np.asarray(X)
        y = np.asarray(y)
        m, n = X.shape
        if isinstance(self.max_samples, float):
            self.max_samples = int(self.max_samples * m)
        if isinstance(self.max_features, float):
            self.max_features = int(self.max_features * n)
        for i in range(self.n_iter // self.n_job):
            self.thread_list.append([1] * self.n_job)
        self.thread_list.append([1] * (self.n_iter % self.n_job))
        return X, y


    def fit(self, X, y, *args):
        X, y = self.initial_params(X, y)
        sp = 0
        ep = min(sp + self.n_job, self.n_iter)
        while sp < self.n_iter:
            threads = []
            for i in range(sp, ep):
                t = BaggingThread(self.base_estimator_list[i], X, y, self.bootstrap, self.max_samples, self.bootstrap_features, self.max_features, self.oob_score)
                t.setDaemon(True)
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            sp = ep
            ep = min(sp + self.n_job, self.n_iter)










