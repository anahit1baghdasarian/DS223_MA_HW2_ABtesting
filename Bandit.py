"""
Multi-Armed Bandit (MAB) Algorithms Comparison

This script compares the performance of Epsilon Greedy and Thompson Sampling algorithms in solving the multi-armed bandit problem.
The script defines classes for Epsilon Greedy and Thompson Sampling bandits, performs experiments, and visualizes the results.

Author: Anahit Baghdasaryan
Date: 23 Oct 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

############################### LOGGER
import logging
from abc import ABC, abstractmethod
from logs import *

logging.basicConfig(level=logging.DEBUG, filename='logfile.log', filemode='w')
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

class Bandit(ABC):
    """Abstract base class representing a multi-armed bandit algorithm."""
        ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##
        
    @abstractmethod
    def __init__(self, p):
        """
        Initialize the Bandit with a given reward probability.
        
        Args:
            p (float): The true mean reward probability of the bandit arm.
        """        
        self.p = p

    @abstractmethod
    def __repr__(self):
        """
        Return a string representation of the Bandit object.
        """        
        pass

    @abstractmethod
    def pull(self):
        """
        Simulate pulling the bandit arm and return the reward.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the bandit's internal parameters based on the observed reward.
        """
        pass

    @abstractmethod
    def experiment(self):
        """
        Perform the bandit algorithm's experiment and record results.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Generate a report summarizing the bandit algorithm's performance.
        """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass
    
    @abstractmethod
    def plot_performance(self):
        """
        Plot the performance of the bandit algorithm.
        """
        pass
#--------------------------------------#

class Visualization():
    """Class for visualizing bandit algorithms' performance."""
    def plot1(self, bandits, num_trials, seed, title):
        """
        Visualize the convergence of winning rates for bandit algorithms.
        
        Args:
            bandits (list): List of Bandit objects.
            num_trials (int): Number of trials in the experiment.
            seed (int): Seed for random number generation. 
            title (str): Title for the plot.
        """

        # Visualize the performance of each bandit: linear and log
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(10, 10)

        for bandit in bandits:
            ax1.plot(bandit.cumulative_average_rwrd, label=f'Bandit {bandit.p:.2f}')

        ax1.set_title('Convergence of Winning Rate for Bandits')
        ax1.set_xlabel('Number of Trials')
        ax1.set_ylabel('Estimated Reward')
        ax1.legend()

        for bandit in bandits:
            ax2.semilogx(range(1, num_trials + 1), bandit.cumulative_average_rwrd, label=f'Bandit {bandit.p:.2f}')

        ax2.set_title('Convergence of Winning Rate for Bandits (Log Scale)')
        ax2.set_xlabel('Number of Trials')
        ax2.set_ylabel('Estimated Reward')
        ax2.legend()
        
        if title is not None:
            plt.suptitle(title)  # Set the general title for the entire figure
        plt.show()

    def plot2(self, bandits_eps, bandits_thomp, num_trials):
        """
        Visualize cumulative rewards for Epsilon Greedy and Thompson Sampling algorithms.
        
        Args:
            bandits_eps (list): List of EpsilonGreedy objects.
            bandits_thomp (list): List of ThompsonSampling objects.
            num_trials (int): Number of trials in the experiment.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        
        for bandit in bandits_eps:
            ax1.plot(bandit.cumulative_reward, label=f'Bandit {bandit.p:.2f}')
                     
        ax1.set_title('Cumulative Rewards Epsilon Greedy')
        ax1.set_xlabel('Number of Trials')
        ax1.set_ylabel('Cumulative Reward')
        ax1.legend()
        
        for bandit in bandits_thomp:
            ax2.plot(bandit.cumulative_reward, label=f'Bandit {bandit.p:.2f}')
                     
        ax2.set_title('Cumulative Rewards Thompson Sampling')
        ax2.set_xlabel('Number of Trials')
        ax2.set_ylabel('Cumulative Reward')
        ax2.legend()
                     
    def plot3(self, bandits_eps, bandits_thomp, num_trials):
        """
        Visualize cumulative regrets for Epsilon Greedy and Thompson Sampling algorithms.
        
        Args:
            bandits_eps (list): List of EpsilonGreedy objects.
            bandits_thomp (list): List of ThompsonSampling objects.
            num_trials (int): Number of trials in the experiment.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)

        for bandit in bandits_eps:
            cumulative_regret_dict = bandit.cumulative_regret_dict
            trials = list(cumulative_regret_dict.keys())
            regrets = list(cumulative_regret_dict.values())
            ax1.plot(trials, regrets, label=f'Bandit {bandit.p:.2f}')

        ax1.set_title('Cumulative Regrets Epsilon Greedy')
        ax1.set_xlabel('Number of Trials')
        ax1.set_ylabel('Cumulative Regret')
        ax1.legend()

        for bandit in bandits_thomp:
            cumulative_regret_dict = bandit.cumulative_regret_dict
            trials = list(cumulative_regret_dict.keys())
            regrets = list(cumulative_regret_dict.values())
            ax2.plot(trials, regrets, label=f'Bandit {bandit.p:.2f}')

        ax2.set_title('Cumulative Regrets Thompson Sampling')
        ax2.set_xlabel('Number of Trials')
        ax2.set_ylabel('Cumulative Regret')
        ax2.legend()

        plt.show()
        
#--------------------------------------#

class EpsilonGreedy(Bandit):
    
    """
    Class representing the Epsilon Greedy bandit algorithm.
    
    This class inherits from the abstract base class `Bandit`.
    """

    def __init__(self, p):
        """
        Initialize an Epsilon Greedy bandit instance.
        
        Args:
            p (float): True mean reward probability of the bandit arm.
        """
        super().__init__(p)
        self.p_estimate = 0
        self.N = 0 

    def __repr__(self):
        """
        Return a string representation of the Epsilon Greedy bandit object.
        """
        return 'Epsilon Greedy Bandit'

    def pull(self):
        """
        Simulate pulling the bandit arm and return the sampled reward.
        
        Returns:
            float: Sampled reward from the bandit arm.
        """
        return np.random.randn() + self.p

    def update(self, res):
        """
        Update the bandit's estimate based on the observed reward.
        
        Args:
            res (float): Observed reward from pulling the bandit arm.
        """
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + res) / self.N

    def experiment(self, bandit_rewards=[1, 2, 3, 4], num_trials = 20000, seed = 71, verbose=True):
        """
        Run the Epsilon Greedy algorithm experiment and record results.
        
        Args:
            bandit_rewards (list): List of true mean reward probabilities for bandit arms. (Default: [1, 2, 3, 4])
            num_trials (int): Number of trials in the experiment. (Default: 20000)
            seed (int): Seed for random number generation. (Default: 71)
            verbose (bool): If True, print detailed experiment results. (Default: True)
        """
        bandits = [EpsilonGreedy(reward) for reward in bandit_rewards]
        true_best = np.argmax(bandit_rewards)
        cumulative_regret = 0
        cumulative_regret_dict = {}
        learningrate_lst = []
        rewards_lst = []
        b_lst = []
        res_t = []
      

        for i in range(1, num_trials + 1):
            eps = 1 / i
            p = np.random.random()
            if p < eps:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])

            band_max = np.argmax([b.p_estimate for b in bandits])
            exp_reward = (1 - eps) * bandits[band_max].p_estimate + eps * (
                        (sum([bandit.p_estimate for bandit in bandits]) - bandits[band_max].p_estimate) / (
                            len(bandits) - 1))
            learningrate_lst.append(exp_reward)
            b_lst.append(j)

            res = bandits[j].pull()
            bandits[j].update(res)
            rewards_lst.append(res)
            
            for n in range(len(bandits)):
                res_t.append(bandits[n].pull())
                
            regret = max(res_t[((i - 1) * len(bandits) ):((i - 1) * len(bandits) + len(bandits))]) - res  
            cumulative_regret += regret
            cumulative_regret_dict[i] = cumulative_regret
        
            
        cumulative_average_rwrd = np.cumsum(rewards_lst) / (np.arange(num_trials) + 1)
        cumulative_average_rgrt = np.cumsum(cumulative_regret) / (np.arange(num_trials) + 1)
        
        estimated_avg_rewards = [bandit.p_estimate for bandit in bandits]
        
        
        if verbose:
            logger.info(f'Estimated average rewards: {estimated_avg_rewards}')
            logger.info("--------------------------------------------------")

        self.rewards = rewards_lst
        self.cumulative_average_rwrd = cumulative_average_rwrd
        self.cumulative_regret = cumulative_regret
        self.cumulative_average_rgrt = cumulative_average_rgrt
        self.cumulative_reward = np.cumsum(rewards_lst)
        self.cumulative_regret_dict = cumulative_regret_dict
        self.learning = learningrate_lst
        self.bandit_choices = b_lst
        self.best_bandit_index = np.argmax([b.p_estimate for b in bandits])

    def plot_performance(self):
        """
        Plot the convergence of winning rate for the Epsilon Greedy bandit.
        """
        plt.plot(self.cumulative_average_rwrd, label='Epsilon Greedy')
        plt.title('Convergence of Winning Rate for Epsilon Greedy')
        plt.xlabel('Number of Trials')
        plt.ylabel('Estimated Reward')
        plt.legend()
        plt.show()

    def report(self, verbose=True, df_save_path=None):
        """
        Generate a report summarizing the performance of the Epsilon Greedy bandit algorithm.
        
        Args:
            verbose (bool): If True, print detailed experiment results. (Default: True)
            df_save_path (str): File path for saving results as CSV. (Default: None)
        
        Returns:
            pd.DataFrame: DataFrame containing experiment results.
        """
        if verbose:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.set_size_inches(10, 10)
            ax1.plot(self.learning)
            ax1.set_title('Learning Progress in Epsilon Greedy with epsilon 1/t')
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Expected Reward')
            ax2.plot(self.cumulative_reward)
            ax2.set_title('Accumulated Reward in Epsilon Greedy with epsilon 1/t')
            ax2.set_xlabel('Number of Trials')
            ax2.set_ylabel('Cumulative Reward')
            logger.info(f'Total Cumulative Reward: {np.sum(self.rewards)}')
            logger.info(f'Total Cumulative Regret: {self.cumulative_regret}')
            logger.info(f'Average Rewards: {self.cumulative_average_rwrd}')
            logger.info(f'Average Regrets: {self.cumulative_average_rgrt}')
        df_dict = {'Bandit': self.bandit_choices, 'Reward': self.rewards, 'Algorithm': 'Epsilon-Greedy'}
        df = pd.DataFrame(df_dict)
        if df_save_path != None:
            df.to_csv(df_save_path)
        return df
    
#--------------------------------------#

class ThompsonSampling(Bandit):
    
    """
    Class representing the Thompson Sampling bandit algorithm.
    
    This class inherits from the abstract base class `Bandit`.
    """
    
    def __init__(self, true_mean):
        """
        Initialize a Thompson Sampling bandit instance.
        
        Args:
            true_mean (float): True mean reward probability of the bandit arm.
        """
        super().__init__(true_mean)
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0
        self.cum_reward = 0
        
    def __repr__(self):
        """
        Return a string representation of the Thompson Sampling bandit object.
        """
        return 'Thompson Sampling Bandit'

    def pull(self):
        """
        Simulate pulling the bandit arm and return the sampled reward.
        
        Returns:
            float: Sampled reward from the bandit arm.
        """
        return (np.random.randn() / np.sqrt(self.tau)) + self.p

    def sample(self):
        """
        Sample a reward using the current parameters of the Thompson Sampling bandit.
        
        Returns:
            float: Sampled reward value.
        """
        return (np.random.randn() / np.sqrt(self.lambda_)) + self.m

    def update(self, res):
        """
        Update the bandit's parameters based on the observed reward.
        
        Args:
            res (float): Observed reward from pulling the bandit arm.
        """
        self.lambda_ += self.tau
        self.cum_reward += res
        self.m = (self.tau * self.cum_reward) / self.lambda_
        self.N += 1

    def experiment(self, bandit_rewards = [1,2,3,4], num_trials = 20000, num_samples_plot = [10,50,100,200,500,1000,5000,10000,20000], seed = 71, verbose=True):
        """
        Run the Thompson Sampling algorithm experiment and record results.
        
        Args:
            bandit_rewards (list): List of true mean reward probabilities for bandit arms. (Default: [1, 2, 3, 4])
            num_trials (int): Number of trials in the experiment. (Default: 20000)
            num_samples_plot (list): List of trial numbers for plotting specific samples. 
                (Default: [10,50,100,200,500,1000,5000,10000,20000])
            seed (int): Seed for random number generation. (Default: 71)
            verbose (bool): If True, print detailed experiment results. (Default: True)
        """
        bandits = [ThompsonSampling(mean) for mean in bandit_rewards]
        true_best = np.argmax(bandit_rewards)
        rewards_lst = []
        learningrate_lst = []
        b_lst = []
        res_t = []
        cumulative_regret = 0
        cumulative_regret_dict = {}

        for i in range(1, num_trials + 1):
            j = np.argmax([bandit.sample() for bandit in bandits])
            learningrate_lst.append(bandits[j].m)
            b_lst.append(j)   

            res = bandits[j].pull()
            bandits[j].update(res)
            rewards_lst.append(res)
            
            for n in range(len(bandits)):
                res_t.append(bandits[n].pull())
                
            regret = max(res_t[((i - 1) * len(bandits) ):((i - 1) * len(bandits) + len(bandits))]) - res   
            cumulative_regret += regret
            cumulative_regret_dict[i] = cumulative_regret
        
        
        cumulative_average_rwrd = np.cumsum(rewards_lst) / (np.arange(num_trials) + 1)
        cumulative_average_rgrt = np.cumsum(cumulative_regret) / (np.arange(num_trials) + 1)
        
        self.rewards = rewards_lst 
        self.cumulative_reward = np.cumsum(rewards_lst)
        self.cumulative_average_rwrd = cumulative_average_rwrd
        self.cumulative_average_rgrt = cumulative_average_rgrt
        self.cumulative_regret = cumulative_regret
        self.cumulative_regret_dict = cumulative_regret_dict
        self.learning = learningrate_lst
        self.bandit_choices = b_lst        
        self.best_bandit_index = np.argmax([bandit.sample() for bandit in bandits])

    def plot_performance(self):
        """
        Plot the convergence of winning rate for the Thompson Sampling bandit.
        """
        plt.plot(self.cumulative_average_rwrd, label='Thompson Sampling')
        plt.title('Convergence of Winning Rate for for Thompson Sampling')
        plt.xlabel('Number of Trials')
        plt.ylabel('Estimated Reward')
        plt.legend()
        plt.show()

    def report(self, verbose=True, df_save_path=None):
        """
        Generate a report summarizing the performance of the Thompson Sampling bandit algorithm.
        
        Args:
            verbose (bool): If True, print detailed experiment results. (Default: True)
            df_save_path (str): File path for saving results as CSV. (Default: None)
        
        Returns:
            pd.DataFrame: DataFrame containing experiment results.
        """
        if verbose:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.set_size_inches(10, 10)
            ax1.plot(self.learning)
            ax1.set_title('Learning Progress in Thompson Sampling')
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Expected Reward')
            ax2.plot(self.cumulative_reward)
            ax2.set_title('Accumulated Reward in Thompson Sampling')
            ax2.set_xlabel('Number of Trials')
            ax2.set_ylabel('Cumulative Reward')
            logger.info(f'Total Cumulative Reward: {np.sum(self.rewards)}')
            logger.info(f'Total Cumulative Regret: {self.cumulative_regret}')
            logger.info(f'Average Rewards: {self.cumulative_average_rwrd}')
            logger.info(f'Average Regrets: {self.cumulative_average_rgrt}')
        df_dict = {'Bandit': self.bandit_choices, 'Reward': self.rewards, 'Algorithm': 'Thompson-Sampling'}
        df = pd.DataFrame(df_dict)
        if df_save_path != None:
            df.to_csv(df_save_path)
        return df
    
#--------------------------------------#

def comparison(bandit_rewards, num_trials=20000, seed=71):
    """
    Compare Epsilon Greedy and Thompson Sampling algorithms.
    
    Args:
        bandit_rewards (list): List of true mean reward probabilities for bandit arms.
        num_trials (int): Number of trials in the experiment. (Default: 20000)
        seed (int): Seed for random number generation. (Default: 71)
    """
    epsilon_greedy = EpsilonGreedy(1)
    epsilon_greedy.experiment(bandit_rewards, num_trials=num_trials, seed=seed, verbose=False)
    df_epsilon = epsilon_greedy.report(verbose=False)

    thompson_sampling = ThompsonSampling(1)
    thompson_sampling.experiment(bandit_rewards, num_trials=num_trials, num_samples_plot=[10,50,100,200,500,1000,5000,10000,20000],
                                 seed=seed, verbose=False)
    df_thompson = thompson_sampling.report(verbose=False)

    epsilon_best_reward = bandit_rewards[epsilon_greedy.best_bandit_index]
    thompson_best_reward = bandit_rewards[thompson_sampling.best_bandit_index]

    if thompson_best_reward > epsilon_best_reward:
        logger.info('The Thompson Sampling algorithm performed better as it discovered a more rewarding arm.')
    elif epsilon_best_reward > thompson_best_reward:
        logger.info('The Epsilon Greedy algorithm performed better as it discovered a more rewarding arm.')
    else:
        logger.info('Both algorithms found equally rewarding arms. Further analysis is required.')
        epsilon_rewards = df_epsilon.loc[:, 'Reward'].tolist()
        thompson_rewards = df_thompson.loc[:, 'Reward'].tolist()
        epsilon_cumulative_reward = np.cumsum(epsilon_rewards)
        thompson_cumulative_reward = np.cumsum(thompson_rewards)
        diff_ls = epsilon_cumulative_reward - thompson_cumulative_reward
        plt.plot(diff_ls)
        plt.title('Cumulative Reward Difference (Epsilon Greedy - Thompson Sampling)')
        plt.xlabel('Number of trials')
        plt.ylabel('Cumulative Reward Difference')
        plt.show()

        logger.info('The T-test evaluates whether the mean rewards achieved through Epsilon Greedy are statistically equivalent to those attained through Thompson Sampling.')
        logger.info(ttest_ind(epsilon_rewards, thompson_rewards, alternative='two-sided'))


if __name__ == '__main__':
    bandit_rewards = [1, 2, 3, 4]
    num_trials = 20000
    seed = 71
    
    # Run the comparison between algorithms
    comparison(bandit_rewards, num_trials=num_trials, seed=seed)
    
#--------------------------------------#

    # Create EpsilonGreedy bandits and run the experiment
    epsilon_greedy = EpsilonGreedy(1)
    epsilon_greedy.experiment(bandit_rewards, num_trials=num_trials, seed=seed, verbose=False)
    
    # Specify the file path for saving EpsilonGreedy results
    epsilon_csv_path = 'epsilon_greedy_results.csv'

    # Save EpsilonGreedy results to CSV
    epsilon_greedy.report(df_save_path=epsilon_csv_path)

    # Plot the results using Visualization class
    Visualization().plot1([epsilon_greedy], num_trials, seed, title='Epsilon Greedy')

#--------------------------------------#

    # Create ThompsonSampling bandits and run the experiment
    thompson_sampling = ThompsonSampling(1)
    thompson_sampling.experiment(bandit_rewards, num_trials=num_trials, num_samples_plot=[10,50,100,200,500,1000,5000,10000,20000],
                                 seed=seed, verbose=False)
    
    # Specify the file path for saving ThompsonSampling results
    thompson_csv_path = 'thompson_sampling_results.csv'

    # Save ThompsonSampling results to CSV
    thompson_sampling.report(df_save_path=thompson_csv_path)
    
    # Plot the results using Visualization class
    Visualization().plot1([thompson_sampling], num_trials, seed, title = 'Thompson Sampling')
    
#--------------------------------------#

    # Plot the results using Visualization class
    Visualization().plot2([epsilon_greedy], [thompson_sampling], num_trials) #cumulative reward
    Visualization().plot3([epsilon_greedy], [thompson_sampling], num_trials) #cumulative regrets
    
# if __name__=='__main__':
   
#     logger.debug("debug message")
#     logger.info("info message")
#     logger.warning("warning message")
#     logger.error("error message")
#     logger.critical("critical message")
