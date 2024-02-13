import torch
from ignite.engine import Engine
from ignite.metrics import PSNR, Precision, Recall


def eval_step(apply_threshold=True, threshold=0.5):
    def eval_step_(engine, batch):
        inputs, targets = batch
        if apply_threshold:
            inputs = torch.where(inputs > threshold, 1., 0.)
            targets = torch.where(targets > threshold, 1., 0.)
        inputs = 1.0 - inputs
        targets = 1.0 - targets

        return inputs, targets

    return eval_step_


class Validator:
    def __init__(self, apply_threshold=True, threshold=0.5):
        self.apply_threshold = apply_threshold
        self._evaluator = Engine(eval_step(self.apply_threshold, threshold))

        self._psnr = PSNR(data_range=1.0)
        self._psnr.attach(self._evaluator, 'psnr')

        if apply_threshold:
            self._precision = Precision()
            self._precision.attach(self._evaluator, 'precision')
            self._recall = Recall()
            self._recall.attach(self._evaluator, 'recall')
            self._precision_value = 0.0
            self._recall_value = 0.0

        self._count = 0
        self._psnr_value = 0.0

    def compute(self, predicts: torch.Tensor, targets: torch.Tensor):
        state = self._evaluator.run([[predicts, targets]])

        self._count += len(predicts)
        self._psnr_value += state.metrics['psnr']
        avg_psnr = state.metrics['psnr'] / len(predicts)
        metrics = {'psnr': avg_psnr}

        if self.apply_threshold:
            self._precision_value += state.metrics['precision']
            self._recall_value += state.metrics['recall']
            avg_precision = 100. * state.metrics['precision'] / len(predicts)
            avg_recall = 100. * state.metrics['recall'] / len(predicts)
            metrics['precision'] = avg_precision
            metrics['recall'] = avg_recall

        return metrics

    def get_metrics(self):
        psnr = self._psnr_value / self._count
        metrics = {'psnr': psnr}

        if self.apply_threshold:
            precision = 100. * self._precision_value / self._count
            recall = 100. * self._recall_value / self._count
            metrics['precision'] = precision
            metrics['recall'] = recall

        return metrics

    def reset(self):
        self._count = 0
        self._psnr_value = 0

        if self.apply_threshold:
            self._precision_value = 0
            self._recall_value = 0
