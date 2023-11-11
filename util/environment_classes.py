import os
import time
import csv
import pandas as pd
from matplotlib import pyplot as plt

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from pettingzoo.test import seed_test, parallel_seed_test, test_save_obs

from pettingzoo.sisl import waterworld_v4


class waterworld_ppo():
    ''' 
    Class to store a PPO training agent, along with all of the settings
    and information about the current training etc.

    Makes it easier to store straining status and evaluate etc
    '''
    
    def __init__(self, log_dir:str = './log_dir', seed:int = 42, 
                 n_envs:int = 8, **env_kwargs):
        '''
        Initialization function. Sets up the environment and learning algorithm

        For now it just takes in optional arguments for the environment. The initialization will
        set up the PPO learner and the working directory to save checkpoints, but won't
        train anything

        Args:
            work_dir    :   where to store the checkpoints
            seed        :   seed for the environment, so we start in the same place
            n_envs      :   how many environments to run in parallel
        '''

        # initialize the environment
        env = waterworld_v4.parallel_env(**env_kwargs) # initialize environment
        env.reset(seed) # so long and thanks for all the fish
        possible_agents = env.possible_agents # have to grab this before we wrap it in a vectorized environment

        # train N environments in parallel
        env = ss.pettingzoo_env_to_vec_env_v1(env) # vectorize that stuff
        env = ss.concat_vec_envs_v1(vec_env= env, num_vec_envs= n_envs, num_cpus = 2, base_class='stable_baselines3')

        # initialize the model
        model = PPO(
            policy = PPOMlpPolicy,
            env=env,
            tensorboard_log=os.path.join(log_dir,'tensorboard_logs'),
            learning_rate=1e-3,
            batch_size=256
        )
    
        # initialize the csv reward file as necessary
        # order of csv file: timestamp, training_step_count, game_number, rewards for each agent
        #Create folder if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # insert a header row if needed, otherwise just write the current row
        reward_csv_file = os.path.join(log_dir,'reward_logs.csv')
        # if not os.path.exists(reward_csv_file):
        with open(reward_csv_file,'w') as fid:
            writer = csv.writer(fid, lineterminator='\n')
            # assemble the header row -- adjustable number of agents
            header_row = ['Timestamp','TrainingStepCount','GameNumber']
            agent_list = [f'{agent}' for agent in possible_agents]
            header_row = header_row + agent_list
            # write to csv
            writer.writerow(header_row)

        # store parameters in instance
        self.env = env # environment
        self.model = model # model
        self.log_dir = log_dir # for storing model checkpoints and tensorboard data
        self.reward_csv_file = os.path.join(log_dir,'reward_logs.csv') # to store the rewards for a csv
        self.curr_chk = None # current checkpoint file name -- to allow iterative training
        self.train_stepcount = 0 # how many steps have we trained?
        self.env_kwargs = env_kwargs # store this for creating the eval environment


    def train(self, num_steps:int = 100_000):
        '''
        train the model. Loads an existing model checkpoint if available, otherwise starts
        anew
        
        Args:
            num_steps   :   steps per training round
        '''

        # if self.curr_chk is not None:
        #     model = PPO.load(os.path.join(self.log_dir, self.curr_chk), env=self.env)
        # else:
        #     model = self.model # might be able to just do this, instead of reloading

        self.model.learn(total_timesteps= num_steps) # does this save in-place? seems to return a model

        # save the current model checkpoint
        self.curr_chk = f"{self.env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.model.save(os.path.join(self.log_dir,self.curr_chk)) # save in the current file name
        self.train_stepcount += num_steps # keep track of how many steps we have trained


    def eval(self, num_games:int=100, render_mode:str=None ):
        '''
        run the games, and see how many rewards they get for each game
        the environment is set to run in AEC mode, but it pulls in the settings
        from the training environments

        Args:
            num_games   :   number of games to run
            render_mode :   how/if to display the outputs of the games.
        '''

        # # get the current checkpoint. If there isn't one, it's going to run off the randomly initialized values
        # if self.curr_chk is not None:
        #     model = PPO.load(os.path.join(self.log_dir, self.curr_chk)) # load current checkpoint
        # else:
        #     model = self.model
        model = self.model

        # apparently we're running this through AEC instead of Parallel. Not sure why...
        # Don't need to wrap this in SS since we're only running one environment at a time
        env_kwargs = self.env_kwargs # load in old one
        env = waterworld_v4.env(render_mode=render_mode, **env_kwargs)

        # open the reward csv log
        fid =  open(self.reward_csv_file,'a')
        writer = csv.writer(fid, lineterminator='\n')

        # loop through the N games
        for ii in range(num_games):
            # set a different seed for each iteration
            env.reset(seed=ii)
            # create a dict of agents and rewards for this game
            rewards = {agent:0 for agent in env.possible_agents}

            # play through each turn
            for agent in env.agent_iter():
                # get the action for this iteration
                obs, reward, termination, truncation, info = env.last()

                # get the rewards per agent
                # have to do this separately since we're using AEC instead of parallel
                for agent_r in env.agents:
                    rewards[agent_r] += env.rewards[agent_r]

                # has the game stopped?
                if termination or truncation:
                    break
                else:
                    act = model.predict(obs, deterministic=True)[0]

                env.step(act)
            
            entry = [time.strftime('%Y%m%d_%H%M%S'), self.train_stepcount, ii]
            reward_list = [rewards[agent] for agent in env.possible_agents]
            entry = entry + reward_list
            writer.writerow(entry)

        env.close() # stop the AEC environment
        fid.close() # close the file pointer

        

    def interlace_run(self, num_loops:int=10, num_steps:int=100_000, num_games:int=100):
        ''' 
        run a series of train and eval loops
        
        Args:
            num_loops   :   number of loops to run
            num_steps   :   number of steps per training loop
            num_games   :   number of games to evaluate per loop
        '''

        # start with an eval to see the randomly initialized model
        self.eval(num_games=num_games)

        # loop it
        for loop in range(num_loops):
            self.train(num_steps=num_steps)
            self.eval(num_games=num_games)
            print(f"Finished loop {loop} of {num_loops}")


    def plot_rewards(self):
        '''
        create a plot of the rewards for the evaluation loops
        '''
        
        # going to use this a lot
        agents = self.env.possible_agents

        if not os.path.exists(self.reward_csv_file):
            print('No data found from evaluation loops. Run eval() or interlace_run()')
            return -1

        log_pd = pd.read_csv(self.reward_csv_file) # easier than csv reader
        games = log_pd['GameNumber'].unique()
        log_pd_games = log_pd.groupby('GameNumber').mean()

        # create a stacked plot with the mean from the games
        plt.stackplot(log_pd_games['StepCount'],[log_pd_games[agent] for agent in agents])
        
        fig,axs = plt.subplots(nrows = len(agents))
        
        for ii_agent, agent in enumerate(agents):
            for game in games:
                game_idx = log_pd['GameNumber'].eq(game).index
                axs[ii_agent].plot(log_pd['StepCount'].iloc[game_idx], log_pd[agent].iloc[game_idx])



if __name__ == '__main__':
    print('initializing from command line with default settings')
    x = waterworld_ppo(log_dir='./log_dir', seed=42, n_envs=8)
    x.interlace_run()
