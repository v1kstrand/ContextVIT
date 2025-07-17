import math

class SchedulerManager:
    def __init__(self):
        self.schedulers = {}
        self.curr_step = 0
        self.curr_epoch = 0

    def add_scheduler(self, name, scheduler):
        self.schedulers[name] = scheduler
        self.schedulers[name].update_value()

    def step(self, exp):
        self.curr_step += 1
        for name, scheduler in self.schedulers.items():
            if scheduler.step():
                exp.log_metric(name, scheduler.curr_value, step=self.curr_epoch)

    def state_dict(self):
        state_dicts = {
            name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
        }
        state_dicts["_curr_step"] = self.curr_step
        state_dicts["_curr_epoch"] = self.curr_epoch
        return state_dicts

    def load_state_dict(self, state_dict):
        self.curr_step = state_dict["_curr_step"]
        self.curr_epoch = state_dict["_curr_epoch"]
        for name, scheduler_state in state_dict.items():
            if name in self.schedulers:
                self.schedulers[name].load_state_dict(scheduler_state)
            elif name[0] != "_":
                print(f"Info: Scheduler '{name}' not found")
        self.update()

    def update(self):
        for scheduler in self.schedulers.values():
            scheduler.update_value()


class Scheduler:
    def __init__(
        self,
        param_name,
        start_value,
        end_value,
        max_steps,
        scheduler_type="linear",
        scheduler_target="optimizer",
        target_object=None
    ):
        assert scheduler_type in ["linear", "cosine"]
        assert scheduler_target in ["attr", "optimizer", "attr_dict"]
        self.scheduler_type = scheduler_type
        self.scheduler_target = scheduler_target
        self.target_object = target_object
        self.param_name = param_name
        self.start_value = start_value
        self.end_value = end_value
        self.max_steps = max_steps
        self.curr_step = 0
        self.curr_value = start_value

    def update_value(self, value=None):
        self.curr_value = value or self.curr_value
        if self.scheduler_target == "attr":
            setattr(self.target_object, self.param_name, self.curr_value)
        elif self.scheduler_target == "attr_dict":
            dict_params, param = self.param_name
            getattr(self.target_object, dict_params)[param] = self.curr_value
        else:
            self.target_object.param_groups[0][self.param_name] = self.curr_value

    def step(self):
        if self.curr_step <= self.max_steps and self.scheduler_type == "cosine":
            new_value = self.start_value + 0.5 * (self.end_value - self.start_value) * (
                1 - math.cos(math.pi * self.curr_step / self.max_steps)
            )
        elif self.curr_step <= self.max_steps and self.scheduler_type == "linear":
            new_value = self.start_value + (self.end_value - self.start_value) * (
                self.curr_step / self.max_steps
            )
        else:
            return False
        
        self.update_value(new_value)
        self.curr_step += 1
        return True

    def state_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k[0] != "_" and k != "target_object"
        }

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)