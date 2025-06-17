import numpy as np
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed by attributes (as well
        as normally).

    >>> obj = AttrDict()
    >>> obj['test'] = 'hi'
    >>> print obj.test
    hi
    >>> del obj.test
    >>> obj.test = 'bye'
    >>> print obj['test']
    bye
    >>> print len(obj)
    1
    >>> obj.clear()
    >>> print len(obj)
    0
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})

def audio_aggregate(aud, aggr_len=5):
    pad = (aggr_len-1) // 2
    aud = np.pad(aud, ((pad,pad),(0,0)))
    aud_expand = []
    for i in range(pad, len(aud)-pad):
        temp = aud[i-pad:i+(pad+1)]
        aud_expand.append(temp)
    aud_expand = np.array(aud_expand)
    return aud_expand
        
class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_after_epoch: int, **kwargs):
        """
        Args:
            start_after_epoch (int): The epoch number after which early stopping will start
                                     accumulating epochs with no improvement.
            **kwargs: Other parameters for EarlyStopping (e.g., monitor, patience, mode, min_delta).
        """
        super().__init__(**kwargs)
        self.start_after_epoch = start_after_epoch

    def _run_early_stopping_check(self, trainer):
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        if trainer.current_epoch > self.start_after_epoch:
            should_stop, reason = self._evaluate_stopping_criteria(current)
    
            # stop every ddp process if any world process decides to stop
            should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
            trainer.should_stop = trainer.should_stop or should_stop
            if should_stop:
                self.stopped_epoch = trainer.current_epoch
            if reason and self.verbose:
                self._log_info(trainer, reason, self.log_rank_zero_only)