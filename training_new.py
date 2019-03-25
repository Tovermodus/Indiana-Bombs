#import main
from agent_code.test_agent.q_funct import Qf
Q = Qf()
Q.setup()
#for j in range(200):
#    for i in range(15):
#        main.main()
Q.load()
Q.train_new_from_file()
