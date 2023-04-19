import abc 


class BaseModel(object, metaclass = abc.ABCMeta):

    @abc.abstractmethod
    def select_action(self, state):
        pass

    @abc.abstractmethod 
    def train(self):
        pass 
    