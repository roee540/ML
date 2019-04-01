import numpy as np
import random
import time

action_size = 6
state_size = 49
qtable = np.zeros((2, state_size, action_size))
#actions = [ 'right' , 'left' , 'up' , 'down' , 'pick' , 'drop' ]
actions = [0, 1, 2, 3, 4, 5]
global is_picked
is_picked = False

def training(env):
    total_episodes = 3000
    max_steps = 1999

    learning_rate = 0.7
    gamma = 0.618 

    epsilon = 1.0        
    max_epsilon = 1.0        
    min_epsilon = 0.01         
    decay_rate = 0.01
    
    for episode in range(total_episodes):
        global is_picked
        state = get_1d_state(env)
        step = 0 
        done = False
        
        for step in range(max_steps):
            exp_exp_tradeoff = random.uniform(0,1)
            
            if is_picked == True:
                is_picked = 1
            else:
                is_picked = 0
            
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[is_picked, state, :])          

            else:
                action = random.choice(actions)
            
            if episode >= 2900:
                if exp_exp_tradeoff < epsilon:
                    print('random | is picked: {}'.format(is_picked))
                else:
                    print('qtable | is picked: {}'.format(is_picked))
                    
                for i in range(6):
                    qtable[is_picked, state, i] = round(qtable[is_picked, state, i], 2)
                print('Q_value: {}'.format(qtable[is_picked, state]))
                print(env)
                time.sleep(3)    
        
            reward, done, is_picked, first_time_is_picked, first_time_is_droped = rewards(env, action)
            
            if episode >= 2900:
                if action == 0:
                    print('right | reward: {}'.format(reward))
                elif action == 1:
                    print('left | reward: {}'.format(reward))
                elif action == 2:
                    print('up | reward: {}'.format(reward))
                elif action == 3:
                    print('down | reward: {}'.format(reward))
                elif action == 4:
                    print('pick | reward: {}'.format(reward))
                else:
                    print('drop | reward: {}'.format(reward))
            
            new_state = moves(env, action)
            
            if is_picked == True:
                is_picked = 1
            else:
                is_picked = 0
                
                
            if is_picked == True and first_time_is_picked == True:
                is_picked = 0
                
            if is_picked == False and first_time_is_droped == True:
                is_picked = 1

            qtable[is_picked, state, action] = qtable[is_picked, state, action] + learning_rate * (reward + gamma * 
                                    np.max(qtable[is_picked, new_state, :]) - qtable[is_picked, state, action])
                
            
            if is_picked == False and first_time_is_picked == True:
                is_picked = True
                
            if is_picked == True and first_time_is_droped == True:
                is_picked = False
            
            state = new_state
        
            if done == True:
                is_picked = 0
                valid_loc = []
                for i in range(6):
                    if env[i][3] == 0:
                        valid_loc.append(i)
                random_loc = random.choice(valid_loc)
                env[random_loc][3] = 1
                env[6][6] = 3  

        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        
        
def test(env):
    total_test_episodes = 10
    max_steps = 99
    global is_picked
    is_picked = 0
    for episode in range(total_test_episodes):
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(max_steps):
            state = get_1d_state(env)
            action = np.argmax(qtable[is_picked, state,:])
            
            reward, done, is_picked, first_time_is_picked, first_time_is_droped = rewards(env, action)
            moves(env, action)
            
            if is_picked == True:
                is_picked = 1
            else:
                is_picked = 0
        

            if action == 0:
                print('right | reward: {}'.format(reward))
            elif action == 1:
                print('left | reward: {}'.format(reward))
            elif action == 2:
                print('up | reward: {}'.format(reward))
            elif action == 3:
                print('down | reward: {}'.format(reward))
            elif action == 4:
                print('pick | reward: {}'.format(reward))
            else:
                print('drop | reward: {}'.format(reward))
                     
                
            print('qtable | is picked: {}'.format(is_picked))

            print('Q_value: {}'.format(qtable[is_picked, state]))
            print(env)
            time.sleep(3)
        
            if done:
                is_picked = 0

def create_env():
    agent, pick, drop, block = [i for i in range(1, 5)]
    valid_loc = [] 
    env = np.zeros((7, 7))
    env[0,0] = pick
    env[6,6] = drop
    for i in range(7):
        j = random.choice([1,2,3,4,5])
        env[i,j] = block
       
    for i in range(6):
        for j in range(6):
            if env[i,j] != pick and env[i,j] != drop and env[i,j] != block:
                valid_loc.append([i,j])
    i, j = random.choice(valid_loc)
    env[i][j] = agent
    
    return env

def get_1d_state(env):
    env_copy = env.copy()
    env_copy.shape = (49, 1)
    for i in range(len(env_copy)):
        if env_copy[i] == 1:
            return i

def get_2d_state(env):
    for i in range(7):
        for k in range(7):
            if env[i][k] == 1:
                return [i, k]

def is_valid_move(env, action):
    for i in range(7):
        if env[0][i] == 1 and action == 2:
            return False
        elif env[6][i] == 1 and action == 3:
            return False
        elif env[i][0] == 1 and action == 1:
            return False
        elif env[i][6] == 1 and action == 0:
            return False
        else:
            pass
    return True

def round_qtable(qtable):
    for q in range(2):
        for i in range(49):
            for k in range(6):
                qtable[q,i,k] = round(qtable[q,i,k], 2)
    return qtable

def rewards(env, action):
    done = False
    score = 0
    global is_picked
    first_time_is_picked = False
    first_time_is_droped = False
    agent_loc = get_2d_state(env)
    
    if not is_picked:
        if (action == 1 and agent_loc == [0, 1]) or (action == 2 and agent_loc == [1, 0]):
            score = 10
        elif action in (1, 2) and agent_loc == [1, 1]:
            score = 5
        elif (action in (0, 3) and agent_loc == [6, 5]) or (action in (3, 0 ) and agent_loc == [6, 5]):
            score == -10
        elif action == 4 and agent_loc == [0, 0]:
            score = 100
            is_picked = True
            first_time_is_picked = True
            return score, done, is_picked, first_time_is_picked, first_time_is_droped
        else:
            score = -1
            
    if is_picked:
        if action == 5 and agent_loc != [6, 6]:
            is_picked = False
            first_time_is_droped = True
            score = -1
        elif (action == 0 and agent_loc == [0,0] and env[0,1] != 4) or (action == 3 and agent_loc == [0,0] and env[1,0] != 4):
            score = 5
        elif (action == 1 and agent_loc == [0, 1]) or (action == 2 and agent_loc == [1, 0]):
            score = -10    
        elif (action == 0 and agent_loc == [6, 5]) or (action == 3 and agent_loc == [5, 6]):
            score = 200
        elif action == 5 and agent_loc == [6, 6]:
            score = 1000
            done = True
        else:
            score = -1
        
    return score, done, is_picked, first_time_is_picked, first_time_is_droped
    
                   
def moves(env, action):
    agent_loc = get_2d_state(env)
    i, j = agent_loc
    if is_valid_move(env, action):
        if action == 0 and env[i, j+1] in (0, 3):
            env[i,j] = 0
            env[i,j+1] = 1
            if env[0,0] == 0:
                env[0,0] = 2
        elif action == 1 and env[i, j-1] in (0, 2):
            env[i,j] = 0
            env[i,j-1] = 1
            if env[6,6] == 0:
                env[6,6] = 3
        elif action == 2 and env[i-1, j] in (0, 2):
            env[i,j] = 0
            env[i-1,j] = 1
            if env[6,6] == 0:
                env[6,6] = 3
        elif action == 3 and env[i+1, j] in (0, 3):
            env[i,j] = 0
            env[i+1,j] = 1
            if env[0,0] == 0:
                env[0,0] = 2
        else:
            pass
    
    return get_1d_state(env)
    
if __name__ == '__main__':
    env = create_env()
    training(env)
    qtable = round_qtable(qtable)
    test(env)
    