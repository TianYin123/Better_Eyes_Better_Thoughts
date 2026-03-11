from abc import abstractmethod

class BaseLLM:
    def __init__(self):
        pass

    @abstractmethod
    def process_messages(self,messages):
        pass

    @abstractmethod
    def generate_output(self,messages):
        pass
    
    @abstractmethod
    def generate_outputs(self,messages_list):
        pass
