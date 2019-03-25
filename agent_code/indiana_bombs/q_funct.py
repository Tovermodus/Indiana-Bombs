import numpy as np
from sklearn.neural_network import MLPRegressor as MLP
import pickle


def sigm(v, s = 1):
    return 1/(1+np.exp(-s*v))
class Qf():
    
    def setup(self):
        self.mlp = [MLP(verbose=True, hidden_layer_sizes=(100,), max_iter=400, warm_start=True) for i in range(6)]
        self.learning_rate = 0.3
        self.discount_rate = 0.7
        #self.train_from_file()
    def load(self):
        for i in range(6):
            filename = 'mlp_' + str(i) + '.sav'
            self.mlp[i] = pickle.load(open(filename, 'rb'))
    def save(self):
        for i in range(6):
            filename = 'mlp_' + str(i) + '.sav'
            pickle.dump(self.mlp[i],open(filename, 'wb'))
    def predict(self, dataset):      #dataset = [[aktion1,daten],[aktion2,daten],...]
        return np.array([self.mlp[int(dataset[i,0])].predict(np.array([dataset[i,1:]])) for i in range(len(dataset))])
    def train(self, train_dataset):#dataset = [[reward1,aktion1,daten1],[reward2,aktion2,daten2],...]
        for i in range(6):
            print(i)
            print(train_dataset)
        
            valid_data_indices = np.where(train_dataset[:-1,1] == i)[0]
            valid_data_length = len(valid_data_indices)
            features = train_dataset[valid_data_indices,2:]
            actions = train_dataset[valid_data_indices,1]
            rewards = train_dataset[valid_data_indices,0]
            targets = np.zeros(valid_data_length)
            for j in range(valid_data_length):
                next_state = train_dataset[valid_data_indices[j]+1,2:]
                targets[j] = (1-self.learning_rate)*self.predict(np.array([np.append([i],features[j])])) + self.learning_rate * self.td_error(rewards[j], next_state)
            if(len(features) != 0):
                self.mlp[i].fit(features, targets)
            else:
                features = train_dataset[0:5,2:]
                targets = np.zeros(5)
                self.mlp[i].fit(features, targets)
                
            
            print("fitted",i)
        self.save()
    def train_new(self, train_dataset):#dataset = [[reward1,aktion1,daten1],[reward2,aktion2,daten2],...]
        mlp = [MLP(verbose=True, hidden_layer_sizes=(100,), max_iter=300, warm_start=True, learning_rate='adaptive') for i in range(6)]
        for i in range(6):
            print(i)
            print(train_dataset)
        
            valid_data_indices = np.where(train_dataset[:-1,1] == i)[0]
            valid_data_length = len(valid_data_indices)
            features = train_dataset[valid_data_indices,2:]
            actions = train_dataset[valid_data_indices,1]
            rewards = train_dataset[valid_data_indices,0]
            targets = np.zeros(valid_data_length)
            for j in range(valid_data_length):
                next_state = train_dataset[valid_data_indices[j]+1,2:]
                targets[j] = (1-self.learning_rate)*self.predict(np.array([np.append([i],features[j])])) + self.learning_rate * self.td_error(rewards[j], next_state)
            if(len(features) != 0):
                mlp[i].fit(features, targets)
            else:
                features = train_dataset[0:5,2:]
                targets = np.zeros(5)
                mlp[i].fit(features, targets)
                
            
            print("fitted",i)
        self.mlp = mlp
        self.save()
    def train_from_file(self):
        dataset = np.load("training_data.npy")
        self.train(dataset[-50000:])
    def train_new_from_file(self):
        dataset = np.load("training_data.npy")
        self.train_new(dataset[-200000:])
    def td_error(self,reward, next_state): 
        next_action_states = np.array([np.append([j],next_state) for j in range(6)])
        next_Q = self.predict(next_action_states)
        max_next_Q = np.max(next_Q)
        return reward + self.discount_rate * max_next_Q
