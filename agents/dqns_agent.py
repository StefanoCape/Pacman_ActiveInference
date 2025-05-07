import logging
import math
import numpy as np
import random
import scallopy
import torch
import tqdm

from typing import *

from ..pacman.arena import AvoidingArena
from .utils import extract_cell, image_to_torch, Memory, Transition



DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
LOGGER = logging.getLogger(__name__)

torch.set_default_device(DEVICE)



class CellClassifier(torch.nn.Module):
    """
    A convolutional neural network for cell image classification with optional MC Bayesian dropout.

    This model includes convolutional layers followed by fully connected layers and
    uses dropout to enable Monte Carlo (MC) sampling for Bayesian inference.
    """

    def __init__(self, bayesian :bool=False):
        """
        Parameters
        ----------
        bayesian : bool, optional
            Whether to use a Monte-Carlo dropout to approximate a Bayesian CNN, by default ``False``.
        """
        super(CellClassifier, self).__init__()
        self.bayesian = bayesian

        self.c2d_1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4, padding=2)
        self.c2d_2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=4, padding=1)

        self.fc_1 = torch.nn.Linear(in_features=512, out_features=256)
        self.fc_2 = torch.nn.Linear(in_features=256, out_features=4)

        self.dropout = torch.nn.Dropout(p=0.5)
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        return


    def forward(self, x :torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        If Bayesian mode is enabled, dropout is applied in training mode even during inference.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, 3, H, W)
            Input tensor, where H and W are height and width of input images, i.e. cells.

        Returns
        -------
        : torch.Tensor of shape (batch_size, 4)
            Output tensor representing class probabilities.
        """
        if self.bayesian:
            self.dropout.train(True)
        
        # conv layers
        x = self.relu(self.c2d_1(x))  # -> (_, 16, 16, 16)
        x = self.relu(self.c2d_2(x))  # -> (_, 32,  4,  4)

        x = self.flatten(x)  # -> (_, 512)

        # dense layers
        x = self.dropout(x)
        x = self.relu(self.fc_1(x))  # -> (_, 256)
        x = self.dropout(x)
        x = self.softmax(self.fc_2(x))  # -> (_, 4)
        return x


    def mc_forward(self, x: torch.Tensor, n_samples :int=10) -> torch.Tensor:
        """
        Perform a forward pass with Monte Carlo dropout.

        This method applies the forward pass multiple times with dropout enabled to approximate
        a posterior distribution over the predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, H, W), where H and W are height and width of input images.
        n_samples : int, optional
            Number of Monte Carlo samples to draw, by default is ``10``.

        Returns
        -------
        mean : torch.Tensor of shape (batch_size, 4)
            Mean predicted class probabilities.
        std : torch.Tensor of shape (batch_size, 4)
            Standard deviation of the predicted class probabilities.

        Notes
        -----
        Calling this method with ``self.bayesian = False`` produces the same output as the ``forward()`` method,
        but it is slower.
        """
        if not self.bayesian:
            LOGGER.warning("calling Monte Carlo forward method with <self.bayesian> set to False")

        x = x.repeat_interleave(n_samples, axis=0)  # -> (_ * n_samples, 3, H, W)
        preds = self.forward(x)                     # -> (_ * n_samples, 4)
        preds = preds.reshape(-1, n_samples, 4)     # -> (_, n_samples, 4)

        mean, std = preds.mean(axis=1), preds.std(axis=1)  # -> (_, 4)
        return mean, std



class PolicyNet(torch.nn.Module):
    """
    This class implements the agent's policy network.
    It consists of a neural component for feature extraction and 
    a logical part that maps the features to actions.
    """

    def __init__(
        self,
        arena :AvoidingArena,
        bayesian :bool=False,
        n_samples :int=1,
        provenance :str='difftopkproofs',
        edge_penality :float=0.1
    ):
        """
        Parameters
        ----------
        arena : AvoidingArena
            Arena of the game.
        bayesian : bool, optional
            Whether to use a Bayesian cell classifier, by default ``False``.
        n_samples : int, optional
            Number of samples used in Bayesian inference, by default ``1``.
            This parameter is ignored if ``bayesain=False``.
        provenance : str, optional
            Type of provenance used during execution, by default ``'difftopkproofs'``.
        edge_penality : float, optional
            Factor used to penalize longer paths that lead the agent to the target.
            It must have a value between [0.0, 1.0], by default ``0.1``.
        """
        super(PolicyNet, self).__init__()

        self.arena = arena
        self.bayesian = bayesian
        self.n_samples = n_samples
        self.provenance = provenance
        self.edge_penality = edge_penality

        self.num_cells = arena.grid_x * arena.grid_y
        self.nodes = [(i,j) for i in range(arena.grid_x) for j in range(arena.grid_y)]

        # neural component
        self.cell_classifier = CellClassifier(self.bayesian)

        # logical component
        self.path_planner = scallopy.Module(
            program = f"""
                // grid nodes
                type grid_node(x: usize, y: usize)

                // input from neural networks
                type agent(x: usize, y: usize)
                type target(x: usize, y: usize)
                type enemy(x: usize, y: usize)
                
                // safe nodes of the grid
                rel node(x, y) = grid_node(x, y) and not enemy(x, y)
                
                // basic connectivity
                rel edge(x, y, xp, y, {self.arena.Actions.RIGHT.value}) = node(x, y) and node(xp, y) and xp == x + 1
                rel edge(x, y, x, yp, {self.arena.Actions.UP.value})    = node(x, y) and node(x, yp) and yp == y + 1
                rel edge(x, y, xp, y, {self.arena.Actions.LEFT.value})  = node(x, y) and node(xp, y) and xp == x - 1
                rel edge(x, y, x, yp, {self.arena.Actions.DOWN.value})  = node(x, y) and node(x, yp) and yp == y - 1
                
                // path for connectivity conditioned on no enemy on the path
                rel path(x, y, x, y) = node(x, y)
                rel path(x, y, xp, yp) = edge(x, y, xp, yp, _)
                rel path(x, y, xpp, ypp) = path(x, y, xp, yp) and edge(xp, yp, xpp, ypp, _)

                // get the next position
                rel next_position(xp, yp, a) = agent(x, y) and edge(x, y, xp, yp, a)
                rel next_action(a) = next_position(x, y, a) and path(x, y, gx, gy) and target(gx, gy)

                // constraint violation
                rel too_many_goal() = n := count(x, y: target(x, y)), n > 1
                rel too_many_enemy() = n := count(x, y: enemy(x, y)), n > {self.arena.num_enemies}
                rel violation() = too_many_goal() or too_many_enemy()
            """,
            provenance = self.provenance,
            k=1,
            facts = {
                "node": [(torch.tensor(1 - self.edge_penality, requires_grad=False), node) for node in self.nodes]
            },
            input_mappings = {
                "agent": self.nodes,
                "target": self.nodes,
                "enemy": self.nodes
            },
            retain_topk={"agent": 3, "target": 3, "enemy": 7},
            output_mappings = {
                "next_action": list(range(4)),
                "violation": ()
            }
        )
        return


    def forward(self, x :torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input observations and predict the next action probabilities.

        Given a batch of RGB images representing the environment, this method extracts
        spatial features from predefined cells, classifies cell content, and uses a 
        path planning module to predict the agent's next action.

        Parameters
        ----------
        x : torch.Tensor of shape (B, C, H, W)
            A 4D tensor where B is the batch size, C must be 3 (RGB channels) and
            H and W are the height and width of the input images.
            The tensor must have a floating point data type.

        Returns
        -------
        next_action : torch.Tensor of shape (B, A)
            A 2D tensor where A is the number of possible actions.
            Each row contains the softmax-normalized probabilities for the next action.
        violation : torch.Tensor of shape (B,)
            Probabilistic indicator of logic rule violations for each sample in the batch.
            Indicates whether any constraints (e.g., multiple targets) were violated.
        """
        batch_size, n_channel, *_ = x.shape

        if x.ndim != 4 or n_channel != 3:
            LOGGER.error("<x>'s shape should be (B,C,H,W)")
        if not torch.is_floating_point(x):
            LOGGER.error("<x>'s dtype should be a float")

        # split the grids into cells
        cells = torch.stack(
            [
                torch.stack(
                    [
                        extract_cell(x_i, node[0], node[1], self.arena.cell_size)
                        for node in self.nodes
                    ]
                )
                for x_i in x
            ]
        ).reshape(batch_size * self.num_cells, 3, self.arena.cell_size, self.arena.cell_size)

        # extract features
        if self.bayesian:
            mean, _ = self.cell_classifier.mc_forward(cells, self.n_samples)
            features = mean.reshape(batch_size, self.num_cells, 4)
        else:
            features = self.cell_classifier.forward(cells).reshape(batch_size, self.num_cells, 4)
        
        agent_p = features[:, :, 0]
        target_p =  features[:, :, 1]
        enemy_p = features[:, :, 2]
        #empty_p = features[:, :, 3]

        # predict next action probabilities
        results = self.path_planner(agent=agent_p, target=target_p, enemy=enemy_p)
        next_action = torch.softmax(results["next_action"], dim=1)
        violation = results["violation"]
        return next_action, violation



class DQNSAgent():
    """
    Agent, based on Deep Q-Network, that uses an 
    epsilon-greedy policy to solve the Pacaman Maze game.
    """

    def __init__(
        self,
        arena :AvoidingArena,
        batch_size :int=32,
        memory_size :int=1024,
        gamma :float=0.99,
        eps_start :float=0.9,
        eps_end :float=0.05,
        eps_decay :float=1000,
        lr :float=1e-4,
        tau :float=5e-3,
        bayesian_net :bool=False,
        n_samples :int=1,
        provenance :str='difftopkproofs',
        model_weights_path :str=None
    ):
        """
        Parameters
        ----------
        arena : AvoidingArena
            Arena of the game.
        batch_size : int, optional
            Batch size used to train the agent, by default ``32``.
        memory_size : int, optional
            Capacity of the agent's memory in number of transitions, by default ``1024``.
        gamma : float, optional
            Discount factor of the Q-learning algorithm, by default ``0.99``.
        eps_start : float, optional
            Starting value of epsilon, by default ``0.9``.
            It determines the probability of choosing an action at random during training.
        eps_end : float, optional
            Final value of epsilon, by default ``0.05``.
        eps_decay : float, optional
            Controls the rate of exponential decay of epsilon, by default ``1000``.
        lr : float, optional
            Learning rate of the optimizer, by default ``1e-4``.
        tau : float, optional
            Update rate of the target network, by default ``5e-3``.
        bayesian : bool, optional
            Whether to use a Bayesian cell classifier, by default ``False``.
        n_samples : int, optional
            Number of samples used in Bayesian inference, by default ``1``.
            This parameter is ignored if ``bayesain=False``.
        provenance : str, optional
            Type of provenance used during execution, by default ``'difftopkproofs'``.
        model_weights_path : Dict | None, optional
            Path to the file containing the model's initial parameters, default ``None``.
            If ``None``, parameters are randomly initialized.
        """
        self.arena = arena
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr
        self.tau = tau
        self.bayesian_net = bayesian_net
        self.n_samples = n_samples
        self.provenance = provenance

        self.training_steps_done = 0
        self.memory = Memory(self.memory_size)

        self.policy_net = PolicyNet(arena, bayesian=bayesian_net, n_samples=n_samples, provenance=provenance)
        self.target_net = PolicyNet(arena, bayesian=bayesian_net, n_samples=n_samples, provenance=provenance)
        
        if model_weights_path is not None:
            self.policy_net.load_state_dict(torch.load(model_weights_path, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.criterion1 = torch.nn.HuberLoss()
        self.criterion2 = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), self.lr)
        return


    def select_action(self, rgb_array :torch.Tensor|np.ndarray) -> int:
        """
        Select an action based on the input image using the policy network.

        Parameters
        ----------
        rgb_array : torch.Tensor | np.ndarray
            Input image represented as a torch tensor with shape (C, H, W) 
            and values in the range [0.0, 1.0] or as a NumPy array with shape (H, W, C)
            and values in the range [0, 255].

        Returns
        -------
        : int
            The index of the action selected by the policy network.

        Notes
        -----
        - If the input is a NumPy array, the function `image_to_torch` is used to 
        convert NumPy array to the expected torch tensor format.
        """  
        if isinstance(rgb_array, np.ndarray):
            rgb_array = image_to_torch(rgb_array)
        
        # add batch dimension
        rgb_array = rgb_array.unsqueeze(0)

        next_action, _ = self.policy_net(rgb_array)
        action = torch.argmax(next_action, dim=1)
        return int(action)


    def train(self, num_episodes :int, num_epochs :int) -> None:
        """
        Train the agent using the specified number of episodes and epochs.

        Parameters
        ----------
        num_episodes : int
            Number of episodes per epoch.
        num_epochs : int
            Number of epochs.
        """
        self.training_steps_done = 0
        for i in range(1, num_epochs + 1):
            self.train_epoch(num_episodes, i)
            self.test_epoch(num_episodes, i)
        return None


    def train_epoch(self, num_episodes :int, epoch :int=0) -> None:
        """
        Train the agent for an epoch using the specified number of episodes.
        
        Parameters
        ----------
        num_episodes : int
            Number of episodes.
        epoch : int, optional
            Epoch considered, by default ``0``.
        """
        if self.arena.render_mode != "rgb_array":
            LOGGER.warning("<arena>'s render mode should be 'rgb_array'")

        iterator = tqdm.tqdm(range(num_episodes))
        counter_succ, counter_upd, sum_loss = 0, 0, 0.0

        for i in iterator:
            _ = self.arena.reset()
            curr_state_image = image_to_torch(self.arena.render())

            done = False
            while not done:
                # select next action using ε-greedy policy
                if random.random() < self._eps_threshold():
                    action = self.arena.action_space.sample()
                else:
                    action = self.select_action(curr_state_image)
                
                # perform selected action
                _, reward, terminated, truncated, _ = self.arena.step(int(action))
                
                done = terminated or truncated
                next_state_image = None if done else image_to_torch(self.arena.render())
                
                # save transition
                transition = Transition(curr_state_image, action, reward, done, next_state_image)
                self.memory.push(transition)
                
                # update policy network's weights
                loss = self._optimize() if len(self.memory) >= self.batch_size else 0.0

                # soft update of the target network's weights θ′ ← τ θ + (1 − τ) θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                # update counters
                sum_loss += loss
                counter_upd += 1
                counter_succ += 1 if done and reward > 0 else 0

                # update current state
                curr_state_image = next_state_image

            # print training progress
            success_rate = (counter_succ / (i + 1)) * 100.0
            avg_loss = sum_loss / counter_upd
            iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {counter_succ}/{i + 1} ({success_rate:.2f}%)")     
        return None


    def test_epoch(self, num_episodes :int, epoch :int=0) -> None:
        """
        Test the agent for an epoch using the specified number of episodes.
        
        Parameters
        ----------
        num_episodes : int
            Number of episodes.
        epoch : int, optional
            Epoch considered, by default ``0``.
        """
        if self.arena.render_mode != "rgb_array":
            LOGGER.warning("<arena>'s render mode should be 'rgb_array'")
        
        iterator = tqdm.tqdm(range(num_episodes))
        counter_succ = 0

        for episode_i in iterator:
            _ = self.arena.reset()
            state_image = image_to_torch(self.arena.render())

            done = False
            while not done:
                # select and perform the next action using the policy network
                action = self.select_action(state_image)
                _, reward, terminated, truncated, _ = self.arena.step(action)
            
                done = terminated or truncated
                state_image = image_to_torch(self.arena.render())

                # update counter
                counter_succ += 1 if done and reward > 0 else 0

            # print test progress
            success_rate = (counter_succ / (episode_i + 1)) * 100.0
            iterator.set_description(f"[Test Epoch {epoch}] Success {counter_succ}/{episode_i + 1} ({success_rate:.2f}%)")
        return None


    def save_model(self, path :str) -> None:
        """
        Save the state dictionary of the policy network to a file.

        Parameters
        ----------
        path : str
            The file path where the model's state dictionary will be saved.
        """
        torch.save(self.policy_net.state_dict(), path)
        return None


    def _eps_threshold(self) -> float:
        """
        Update epsilon according to the parameters provided during
        the object's creation and the number of training steps performed.
        
        Returns
        -------
        : float
            New value of epsilon.
        """
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.training_steps_done / self.eps_decay)
        self.training_steps_done += 1
        return eps


    def _optimize(self) -> float:
        """
        Update the agent's policy using a batch of transitions.
        
        Returns
        -------
        : float
            Current loss.
        
        Notes
        -----
        - A number of transitions bigger than ``self.batch_size`` must have
        taken place before the agent can actually be updated.
        """
        if len(self.memory) < self.batch_size:
            LOGGER.warning("not enough transactions in memory")
            return 0.0
        
        # draw random transitions
        transitions = self.memory.sample(self.batch_size)
        # from batch-array of transitions to transition of batch-arrays
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(batch.done, dtype=torch.bool).logical_not()
        non_final_next_states = torch.stack([ns for ns in batch.next_state if ns is not None])

        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)

        # compute Q(s_t, a) for the action taken
        next_action, violation = self.policy_net(state_batch)
        state_action_values = next_action.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # compute V(s_{t+1}) for all the next states
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states)[0].max(1).values
        # compute the expected Q(s_t, a)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # compute the loss
        loss1 = self.criterion1(state_action_values, expected_state_action_values)
        loss2 = self.criterion2(violation.detach(), torch.zeros(self.batch_size))
        loss = loss1 + loss2

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()