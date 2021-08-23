from utils import *
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURR_DIR)
sys.path.append(CURR_DIR + "/ACL19")
# sys.path.append(CURR_DIR + "/ECML PKDD 2020")
sys.path.append(CURR_DIR + "/EMNLP20")

from ACL19 import *
# from ECMLPKDD20 import *
from EMNLP20 import *

class Models():
    def __init__(self, setup):
      super(Models, self).__init__()
      self.ACL19 = ACL19(setup)
      # self.ECMLPKDD20 = ECMLPKDD20(setup)
      self.EMNLP20 = EMNLP20(setup)

    def load_model(self,model_name):
      if model_name == "ACL":  
        return self.ACL19
      # if model_name == "ECMLPKDD20":  
      #   return self.ECMLPKDD20
      if model_name == "EMNLP20":  
        return self.EMNLP20

    def hello(self):
      print("hello")