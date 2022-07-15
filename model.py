import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fn
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = fn.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # loss function

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # state is one dimensional, should be (1,x)
            state = torch.unsqueeze(state, 0) # tf.expand_dims(state, axis=0) in tensorflow
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )         # convert done into tuple with one value 
        
        # 1: predicted Q values with the current state
        pred = self.model(state)    # has 3 values [straight, right, left]
        
        # 2: we need to calculate the new Q value 
        # but we need to have in the same format as prediction [0, 0, 0]
        # so we will clone the prediction (pred) 
        # and then set index with the predicted action (argmax(action)) to the new Q value
        # ------------------------------------------------------- #
        # e.g. the action is [1, 0, 0] so with argmax([1, 0, 0]),
        # the index is taken from the maximum value 
        # and will be replaced with the new Q value.
        # ======================================================= #
        # Q_new = r + y * max(next_predicted Q value)   ->  only do this if not done
        # target = pred.clone()                             # clone for same format
        # preds[argmax(action)] = Q_new                     # set index with the predicted action to the new Q value

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad() # we empty the gradient
        loss = self.criterion(target, pred) # calculate loss with the target(qnew) and pred
        loss.backward() # apply backpropagation to update our gradients

        self.optimizer.step()
