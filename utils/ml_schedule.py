"""
    This script is used for different type of schedule in machine learning
"""
import abc  # library for class abstraction


# define the abstract base class
class Schedule(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_value(self, time):
        pass


# define the constant schedule
class ConstantSchedule(Schedule):
    """ This schedule always returns the same value"""
    def __init__(self, value):
        self._value = value

    def get_value(self, time):
        return self._value


# define the linear schedule
class LinearSchedule(Schedule):
    """ This schedule returns the value linearly"""
    def __init__(self, start_value, end_value, duration):
        self._start_value = start_value
        self._end_value = end_value
        self._duration = duration
        self._schedule_amount = end_value - start_value

    def get_value(self, time):
        return self._start_value + self._schedule_amount * min(1.0, time * 1.0 / self._duration)


# # schedule test
#
# con_schedule = ConstantSchedule(10)
# linear_schedule = LinearSchedule(0, 1, 100)
# for ep in range(100):
#     print("Epoch = {} : value = {}".format(ep, linear_schedule.get_value(ep)))