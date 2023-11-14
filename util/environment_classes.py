import os
import time
import csv
import pandas as pd
from matplotlib import pyplot as plt
# import numpy as np
import cv2

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
    
    def __init__(self, log_dir:str = './log_dir', seed:int = None, 
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

        # initialize the environments
        train_env = waterworld_v4.parallel_env(**env_kwargs) # training environment
        eval_env = waterworld_v4.env(**env_kwargs) # eval environment
        train_env.reset() # so long and thanks for all the fish
        possible_agents = train_env.possible_agents # have to grab this before we wrap it in a vectorized environment

        # train N environments in parallel
        train_env = ss.pettingzoo_env_to_vec_env_v1(train_env) # vectorize that stuff
        train_env = ss.concat_vec_envs_v1(vec_env= train_env, num_vec_envs= n_envs, num_cpus = 2, base_class='stable_baselines3')

        # initialize the model
        model = PPO(
            policy = PPOMlpPolicy,
            env=train_env,
            tensorboard_log=os.path.join(log_dir,'tensorboard_logs'),
            learning_rate=1e-3,
            batch_size=256,
            ent_coef = 0.01,
        )
    
        # initialize the csv reward file as necessary
        # order of csv file: timestamp, training_step_count, game_number, rewards for each agent
        #Create folder if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # insert a header row if needed
        reward_csv_file = os.path.join(log_dir,'reward_logs.csv')
        if not os.path.exists(reward_csv_file):
            with open(reward_csv_file,'a') as fid:
                writer = csv.writer(fid, lineterminator='\n')
                # assemble the header row -- adjustable number of agents
                header_row = ['Timestamp','TrainingStepCount','GameNumber']
                agent_list = [f'{agent}' for agent in possible_agents]
                header_row = header_row + agent_list
                # write to csv
                writer.writerow(header_row)

        # store parameters in instance
        self.env = train_env # training environment
        self.eval_env = eval_env # evaluation environment
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

        self.model.learn(total_timesteps= num_steps) # does this save in-place? seems to return a model

        # save the current model checkpoint
        self.curr_chk = f"{self.env.unwrapped.metadata.get('name')}_{self.train_stepcount:09d}"
        self.train_stepcount += num_steps # keep track of how many steps we have trained
        self.model.save(os.path.join(self.log_dir,self.curr_chk)) # save in the current file name


    def eval(self, num_games:int=100, render_mode:str=None ):
        '''
        run the games, and see how many rewards they get for each game
        the environment is set to run in AEC mode, but it pulls in the settings
        from the training environments

        Args:
            num_games   :   number of games to run
            render_mode :   how/if to display the outputs of the games.
        '''

        model = self.model

        # # apparently we're running this through AEC instead of Parallel. Not sure why...
        # # Don't need to wrap this in SS since we're only running one environment at a time
        # env_kwargs = self.env_kwargs # load in old one
        # env = waterworld_v4.env(render_mode=render_mode, **env_kwargs)

        env = self.eval_env

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
            print(f"Finished loop {loop+1} of {num_loops}")


    def plot_rewards(self):
        '''
        create a plot of the rewards for the evaluation loops
        '''

        if not os.path.exists(self.reward_csv_file):
            print('No data found from evaluation loops. Run eval() or interlace_run()')
            return -1

        log_pd = pd.read_csv(self.reward_csv_file) # easier than csv reader
        games = log_pd['GameNumber'].unique()
        agents = [key for key in log_pd.keys() if 'pursuer' in key]
        
        # get the means and the standard deviations
        log_pd_mean = log_pd.groupby('TrainingStepCount')[agents].mean()
        log_pd_sd = log_pd.groupby('TrainingStepCount')[agents].std()

        # put together a colormap so the standard devs will match
        cmap = plt.get_cmap('tab10')

        # create a stacked plot with the mean from all games
        fig_stack_mean, ax_stack_mean = plt.subplots()
        plt.stackplot(log_pd_mean.index,[log_pd_mean[agent] for agent in agents], alpha=0.5)
        ax_stack_mean.set_xlabel('Training Step')
        ax_stack_mean.set_ylabel('Rewards per Game')
        ax_stack_mean.set_title(f'Mean Reward Counts Summed Across Agents')
        ax_stack_mean.legend([agent for agent in agents])

        # line plots with patches for the standard deviations
        fig_sd, ax_sd = plt.subplots()
        for ii_agent,agent in enumerate(agents):
            ax_sd.plot(log_pd_mean.index,log_pd_mean[agent], label = agent, color=cmap(ii_agent))
            above_std = log_pd_mean[agent] + log_pd_sd[agent]
            below_std = log_pd_mean[agent] - log_pd_sd[agent]
            ax_sd.fill_between(log_pd_mean.index, above_std, below_std, color=cmap(ii_agent), alpha=0.25)
        ax_sd.set_xlabel('Training Step')
        ax_sd.set_ylabel('Total Rewards per Game')
        ax_sd.set_title(f'Mean Rewards, with Standard Deviations')
        ax_sd.legend()
        

        # plot all games individual, subplot for each agent
        fig,axs = plt.subplots(nrows = len(agents), squeeze=False, sharex=True)
        axs = axs.flatten() # to get a 1D array no matter the number of agents we have
        for ii_agent, agent in enumerate(agents):
            for game in games:
                game_idx = log_pd['GameNumber'].eq(game)
                axs[ii_agent].plot(log_pd['TrainingStepCount'].loc[game_idx], log_pd[agent].loc[game_idx], label=f'Game {game}')
            
            axs[ii_agent].legend()
            axs[ii_agent].set_ylabel('Total Rewards per Game')
            axs[ii_agent].set_title(f'Rewards for 10 different games, {agent}')
        axs[-1].set_xlabel('Training Step')



    def load_version(self, quality:str = None, checkpoint_name:str = None, step_num:int = None):
        '''
        load a specific checkpoint and save a video.
        The user can either specify the "quality" of the checkpoint (best or worst), 
        the checkpoint name (name of the .zip file) or the training step number.
        
        Default is to render the best mean agent reward.

        Only allows one argument at a time.
        
        Args:
            quality         :   'best' or 'worst' ['best']
            checkpoint_name :   name of .zip checkpoint file
            step_num        :   training step number
        '''
        
        # set to load the best if nothing is loaded
        if not checkpoint_name and not step_num and not quality:
            quality = 'best'
        
        # run either the best or worst
        if (quality is not None):
            # override the other two
            checkpoint_name = None
            step_num = None
            # load the csv log so we can figure out which timepoint is the best
            log_pd = pd.read_csv(self.reward_csv_file)
            log_pd_mean = log_pd.groupby('TrainingStepCount')[[key for key in log_pd.keys() if 'pursuer' in key]].mean()
            # find the best/worst training epoch
            if quality == 'best':
                extrema_ind = log_pd_mean.sum(axis=1).argmax() # training iteration with best total rewards
            if quality == 'worst':
                extrema_ind = log_pd_mean.sum(axis=1).argmin() # training iteration with worst total rewards

            extrema_step = log_pd_mean.index[extrema_ind] # get the stepcount with greatest average rewards
            ck_name = f"{self.env.unwrapped.metadata['name']}_{extrema_step:09d}"
            full_checkpoint = os.path.join(self.log_dir,ck_name) 

        # a specific checkpoint name
        if checkpoint_name is not None:
            full_checkpoint = os.path.join(self.log_dir, checkpoint_name)
            step_num = None
            if not os.path.exists(full_checkpoint+'.zip'):
                print(f'Could not find {checkpoint_name}')
                return -1

        # a specific training epoch
        # should add checking (make sure it's an epoch we have a csv for)
        # but not for now
        if step_num is not None:
            ck_name = f"{self.env.unwrapped.metadata['name']}_{step_num:09d}"
            full_checkpoint = os.path.join(self.log_dir, ck_name)

        # load model with the appropriate environment
        if not os.path.exists(full_checkpoint+'.zip'):
            print(f'{full_checkpoint} does not exist')
            return -1

        # load the model with the environment
        self.model = PPO.load(full_checkpoint, env = self.env) # load model
        
        
    def render_env(self, frame_rate:float = 30, num_frames:float = 500, name_suffix:str = None):
        '''
        render a video in the log_dir 
        
        Args:
            frame_rate  :   frame rate in hertz for rendered video [30]
            num_frames  :   number of frames to render [500]
            name_app    :   suffix for video name. If None, will append current timestamp
        '''

        env_kwargs = self.env_kwargs
        env = waterworld_v4.env(**env_kwargs, max_cycles = num_frames, render_mode = 'rgb_array') # load environment

        # reward and rendering stuff
        env.reset()
        rewards = {agent:0 for agent in env.agents}
        model = self.model

        # filename stuff
        if name_suffix is None: # append the timestamp if the user doesn't include a suffix
            name_suffix = time.strftime('%Y%m%d_%H%M%S')
        filename = self.env.unwrapped.metadata['name']+'_'+name_suffix+'.mp4'
        filename = os.path.join(self.log_dir,filename)
        fps = frame_rate*len(env.agents) # to account for turns

        # open the VideoWriter object, write all of the frames to it
        vid_writer =  cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (750,750))
       
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
            frame = env.render()
            vid_writer.write(frame)

        vid_writer.release()

                




if __name__ == '__main__':
    print('initializing from command line with default settings')
    x = waterworld_ppo(log_dir='./log_dir', seed=42, n_envs=8)
    x.interlace_run()
