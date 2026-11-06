"""训练辅助：早停策略。"""


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best = 0.0
        self.counter = 0

    def step(self, metric):
        if metric > self.best:
            self.best, self.counter = metric, 0
            return False
        self.counter += 1
        return self.counter >= self.patience
