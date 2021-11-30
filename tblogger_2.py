"""
Author: Steve Paul 
Date: 11/29/21 """


from tensorboard import default
from tensorboard import program


tracking_address = "logger/R1_6"
# tracking_address = "logger/R1_MLP_1"

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None,'--logdir',tracking_address])
    tb.main()