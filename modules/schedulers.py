class SchedulerManager:
    def __init__(self):
        self.schedulers = {}
        self.curr_epoch = 0
        self.curr_step = 0

    def add_scheduler(self, name, scheduler):
        self.schedulers[name] = scheduler

    def step(self, *args, **kwargs):
        for sched in self.schedulers.values():
            if hasattr(sched, 'step'):
                sched.step(*args, **kwargs)

    def state_dict(self):
        return {k: getattr(v, 'state_dict', lambda: {})() for k, v in self.schedulers.items()}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.schedulers and hasattr(self.schedulers[k], 'load_state_dict'):
                self.schedulers[k].load_state_dict(v)
