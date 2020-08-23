import pandas as pd # Load libraries
import numpy as np
import matplotlib.pyplot as plt
import copy # To copy a list / objects
import random # To decide on tie classes
import pickle
from io import BytesIO
import base64

### Loading code
# 1. Load Data
if __name__ == "__main__":
    l_track = pd.read_csv('L-track.txt', skiprows=1, header=None)
    l_track_dim = pd.read_csv('L-track.txt').columns.values

    o_track = pd.read_csv('O-track.txt', skiprows=1, header=None)
    o_track_dim = pd.read_csv('O-track.txt').columns.values
    o_track_dim[1] = '25'

    r_track = pd.read_csv('R-track.txt', skiprows=1, header=None)
    r_track_dim = pd.read_csv('R-track.txt').columns.values

def load_track(track, track_dim):
    """Load the track.txt file and do preprocessing.

    The load_track() function will preprocess the racertack
    so that it can be represented in a numpy format where
    the characters are converted to integers. Boundary states
    of (#) are converted to (0), the starting zone (S) is
    converted to (1), the finishing zone (F) is converted
    to (2), and the racetrack (.) is converted to 3.

    Args:
        track (pandas dataframe): The track after being
            loaded from the textfile.
        track_dim (numpy array): An array of the dimensions
            of the racetrack.
    Returns:
        track_init (numpy matrix): The racetrack after
            being converted to a numpy matrix.
    """
    track_dim = [int(dim) for dim in track_dim]
    track_init = np.zeros(track_dim)
    for row_index in range(track_dim[0]):
        row_array = np.array(list(track.iloc[row_index,:].values[0]))
        row_array[row_array == '#'] = 0
        row_array[row_array == 'S'] = 1
        row_array[row_array == 'F'] = 2
        row_array[row_array == '.'] = 3
        track_init[row_index,] = row_array
        
    return track_init

if __name__ == "__main__":
    l_track_np = load_track(track=l_track, track_dim=l_track_dim)
    o_track_np = load_track(track=o_track, track_dim=o_track_dim)
    r_track_np = load_track(track=r_track, track_dim=r_track_dim)

# 2. Reinforcement Learning
class value_iteration(object):
    """Implement the Value Iteration (VI) algorithm.

    The value_iteration() class allows for implementing the
    VI algorithm on the racetrack problem. There are two
    hyperparameters for the VI algorithm and one hyperparameter
    for the racetrack.

    Args:
        see __init__()
    Returns:
        see test()
    """
    def __init__(self, test_board, discount_rate, bellman_error, crash_type=1):
        """Initialize variables for the class.

        The __init__() function initializes the variables of the model
        so that the algorithm can function. The discount rate and Bellman
        error are related to the VI algorithm. The crash type is related
        to the problem, where 1 represents the vehicle resetting to the
        start zone in a crash and 2 represents the vehicle resetting to
        the nearest racetrack location.

        Args:
            test_board (numpy matrix): The racetrack represented within
                a numpy matrix.
            discount_rate (float): The discount rate between [0,1].
            bellman_error (float): The Bellman error >=0.
            crash_type (integer): The crash type (1 or 2).
        """
        # Initialize VI hyperparameters
        self.error = bellman_error
        self.gamma = discount_rate
        self.crash_type = crash_type
        self.all_states = test_board
        self.counter_t = 0
        self.crash_counter_train = 0
        self.crash_counter_test = 0
        self.stall_counter_train = 0
        self.stall_counter_test = 0
        
        board_dim = [dim for dim in test_board.shape]
        self.n_rows = board_dim[0]
        self.n_cols = board_dim[1]

        self.state_x = 0 # Related to the coordinates
        self.state_y = 0
        self.success_x = 0
        self.success_y = 0
        self.fail_x = 0
        self.fail_y = 0

        start_coords = np.where(test_board == 1) # For initial/reset zone
        self.start_coords = tuple(list(zip(start_coords[0], start_coords[1])))

        possible_states = np.where(test_board != 0) # Allowed zones
        self.possible_states_list = tuple(list(zip(possible_states[0],
                                             possible_states[1])))

        goal_coords = np.where(test_board == 2) # For terminate state
        self.goal_coords = tuple(list(zip(goal_coords[0], goal_coords[1])))
        possible_states = np.where(test_board != 0)

        possible_actions = [-1, 0 ,1] # Allowed actions
        self.possible_actions_list = tuple([[accel_x, accel_y] for\
            accel_x in possible_actions for accel_y in possible_actions])
        self.actions_range = range(len(self.possible_actions_list))

        # For updating the Q-values
        self.temp_Q = [0] * len(self.possible_actions_list)
        self.Q_indexer = tuple(list(range(len(self.temp_Q))))

        self.velocity_list = [] # For velocity updates
        self.velocity_limit = (-5, 5)
        
        # Initialize V-table
        self.table_V = np.zeros(board_dim)

        # Initialize Pi-table
        # <velocity_x, velocity_y, acceleration_x, acceleration_y>
        # (4-dim table for state spaces and possible moves)
        board_dim.append(4)
        self.table_pi = np.zeros(tuple(board_dim))

        # Set boundary positions for V-table and Pi-table to NaN's
        boundary_states = np.where(test_board == 0)
        for coord in list(zip(boundary_states[0], boundary_states[1])):
            x_coord, y_coord = coord[0], coord[1]
            self.table_V[x_coord][y_coord] = np.NaN
            self.table_pi[x_coord][y_coord] = np.NaN

        # Identify all boundary pixels
        boundary_states = np.asarray(boundary_states)
        boundary_states = boundary_states.T
        boundary_states = boundary_states.tolist()
        self.boundary_states = tuple(boundary_states)

    def octants3_4_7_8(self, x_init_local, y_init_local,
                       x_final_local, y_final_local):
        """A helper function to bresenham_line_alg().
        
        The octants3_4_7_8() function calculates the line
        path for the half of the octants where the slope
        is >1.

        Args:
            x_init_local (int): The x-coordinate of the starting pixel.
            y_init_local (int): The y-coordinate of the starting pixel.
            x_final_local (int): The x-coordinate of the ending pixel.
            y_final_local (int): The y-coordinate of the ending pixel. 
        Returns:
            line_path (list): A list of the path taken as a
                series of coordinates.
        """
        # Initialize variables
        delta_x = x_final_local - x_init_local
        delta_y = y_final_local - y_init_local
        y_indexer = 1; line_path = []
        y_temp = y_init_local

        if (delta_y < 0): # Check if slope positive or negative
            y_indexer = -1; delta_y = -delta_y

        coefficient_D = 2*delta_y - delta_x

        # Create forward or backward sequence of integers
        if ((x_init_local - x_final_local) < 0):
            x_range = list(range(x_init_local, x_final_local + 1))
        else:
            x_range = list(reversed(range(x_init_local, x_final_local + 1)))

        for x_idx in x_range: # Loop and create line path
            line_path.append([x_idx, y_temp])
            if (coefficient_D > 0):
                y_temp += y_indexer
                coefficient_D = coefficient_D - 2*delta_x

            coefficient_D = coefficient_D + 2*delta_y

        return line_path
            
    def octants1_2_5_6(self, x_init_local, y_init_local,
                       x_final_local, y_final_local):
        """A helper function to bresenham_line_alg().
        
        The octants1_2_5_6() function calculates the line
        path for the half of the octants where the slope
        is <=1.

        Args:
            x_init_local (int): The x-coordinate of the starting pixel.
            y_init_local (int): The y-coordinate of the starting pixel.
            x_final_local (int): The x-coordinate of the ending pixel.
            y_final_local (int): The y-coordinate of the ending pixel. 
        Returns:
            line_path (list): A list of the path taken as a
                series of coordinates.
        """
        # Initialize variables
        delta_x = x_final_local - x_init_local
        delta_y = y_final_local - y_init_local
        x_indexer = 1; line_path = []
        x_temp = x_init_local

        if (delta_x < 0): # Check if slope positive or negative
            x_indexer = -1; delta_x = -delta_x

        coefficient_D = 2*delta_x - delta_y

        # Create forward or backward sequence of integers
        if ((y_init_local - y_final_local) < 0):
            y_range = list(range(y_init_local, y_final_local + 1))
        else:
            y_range = list(reversed(range(y_init_local, y_final_local + 1)))

        for y_idx in y_range: # Loop and create line path
            line_path.append([x_temp, y_idx])
            if (coefficient_D > 0):
                x_temp += x_indexer
                coefficient_D = coefficient_D - 2*delta_y

            coefficient_D = coefficient_D + 2*delta_x

        return line_path
            
    def bresenham_line_alg(self, x_init, y_init, x_final, y_final):
        """Determine the path taken using the Bresenham
        line algorithm.
        
        The bresenham_line_alg() function determines the path
        taken between to coordinates by using the formula for
        the Bresenham line algorithm. The code is based off
        pseudocode from Wikipedia. Permission is given by the
        professor to do so.

        Args:
            x_init (int): The x-coordinate of the starting pixel.
            y_init (int): The y-coordinate of the starting pixel.
            x_final (int): The x-coordinate of the ending pixel.
            y_final (int): The y-coordinate of the ending pixel.
        Returns:
            line_path (list): A list of the path taken as a
                series of coordinates.
        
        Helper functions: octants1_2_5_6(), octants3_4_7_8()
        https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
        """
        # When the slope < 1
        if (abs(y_final - y_init) < abs(x_final - x_init)):
            if (x_init > x_final): # octant 4
                line_path = self.octants3_4_7_8(\
                    x_init_local=x_final, y_init_local=y_final,
                    x_final_local=x_init, y_final_local=y_init)
            else: # octants 3, 7, 8
                line_path = self.octants3_4_7_8(\
                    x_init_local=x_init, y_init_local=y_init,
                    x_final_local=x_final, y_final_local=y_final)
        else: # When the slope >= 1
            if (y_init > y_final): # octant 6
                line_path = self.octants1_2_5_6(\
                    x_init_local=x_final, y_init_local=y_final,
                    x_final_local=x_init, y_final_local=y_init)
            else: # octants 2, 5, 7
                line_path = self.octants1_2_5_6(\
                    x_init_local=x_init, y_init_local=y_init,
                    x_final_local=x_final, y_final_local=y_final)

        # Check if line order has been reversed
        if (line_path[0] != [x_init, y_init]):
            line_path.reverse()

        return line_path
            
    def expected_value(self):
        """Calculate the expected value for the Q function.

        The expected_value() function will calculate the
        expected value of choosing action (a) while in
        state (s), or Q(s,a) for a state-action pair.
        
        Returns:
            exp_value (float): The expected value for the
                state-action pair.
        """
        # If agent crosses finish zone
        if (self.success_x, self.success_y) in self.goal_coords:
            return 0
        else: # If agent doesn't cross finish zone
            exp_value = -1 + self.gamma *\
                (0.8 * self.old_table_V[self.success_x][self.success_y] +\
                 0.2 * self.old_table_V[self.fail_x][self.fail_y])
            return exp_value

    def velocity_update(self, vel_local, accel_local):
        """A helper function to next_state().

        The velocity_update() function will perform the
        velocity update for next_state(). It checks first
        if the current velocity is already either the max
        or min and will update it accordingly.

        Args:
            vel_local (int): The current velocity.
            accel_local (int): The current acceleration.
        Returns:
            vel_local (int): If already max or min.
            vel_local + accel_local (int): The normal
                velocity update.
        """
        # Max velocity
        if ((vel_local == self.velocity_limit[1]) and (accel_local > 0)):
            return vel_local
        # Min velocity
        elif ((vel_local == self.velocity_limit[0]) and (accel_local < 0)):
            return vel_local
        else: # Normal velocity update
            return vel_local + accel_local
        
    def next_state(self, local_action, success=True, testing_stage=False):
        """Determine the next state from a given state.
        
        The next_state() function will find the next state from
        a given state. The VI algorithm requires at various steps
        for a next state to be determined. This is used during
        both training and testing. There are also conditions
        such as whether or not acceleration has failed or not.

        Args:
            local_action (int): The action to be taken at the
                current state.
            success (bool): Indicate if acceleration has worked.
            testing_stage (bool): Indicate if the stage is
                testing or training.
        """
        # Initialize acceleration and velocity variables
        if testing_stage == False: # Training initialization
            vel_x = self.table_pi[self.state_x][self.state_y][0]
            vel_y = self.table_pi[self.state_x][self.state_y][1]
            accel_x = local_action[0]; accel_y = local_action[1]
        else: # Testing initialization
            vel_x = self.pi_state[0]; vel_y = self.pi_state[1]
            accel_x = self.pi_state[2]; accel_y = self.pi_state[3]
        
        # Potential next state
        if success: # If acceleration works
            vel_x = self.velocity_update(vel_local=vel_x, accel_local=accel_x)
            vel_y = self.velocity_update(vel_local=vel_y, accel_local=accel_y)
            new_x_coord = int(self.state_x + vel_x)
            new_y_coord = int(self.state_y + vel_y)
        else: # If acceleration fails
            new_x_coord = int(self.state_x + vel_x)
            new_y_coord = int(self.state_y + vel_y)
        
        line_path = self.bresenham_line_alg(\
            x_init=self.state_x, y_init=self.state_y,
            x_final=new_x_coord, y_final=new_y_coord) # Line path

        # Check for either finishing or crashing
        for path_coord_idx in list(range(len(line_path))):
            # Check for crashes first
            if (line_path[path_coord_idx] in self.boundary_states):
                if testing_stage: # Update crash counter
                    self.crash_counter_test += 1
                else:
                    self.crash_counter_train += 1
                # Crash case 1, reset to start zone
                if (self.crash_type == 1):
                    reset_coords = random.choice(self.start_coords)
                    new_x_coord = reset_coords[0]
                    new_y_coord = reset_coords[1]
                    vel_x, vel_y = 0, 0
                    break
                else: # Crash case 2, reset to nearest coords
                    reset_coords = line_path[path_coord_idx - 1]
                    new_x_coord = reset_coords[0]
                    new_y_coord = reset_coords[1]
                    vel_x, vel_y = 0, 0
                    break
            # Check if finished without crashing
            elif (tuple(line_path[path_coord_idx]) in self.goal_coords):
                new_x_coord = line_path[path_coord_idx][0]
                new_y_coord = line_path[path_coord_idx][1]
                # Update flag if taking action
                if (testing_stage):
                    self.finished = True
                break

        if testing_stage == False: # Velocity list update
            self.velocity_list.append([vel_x, vel_y])

        # Update next state coordinates
        if success: # if acceleration works
            self.success_x = new_x_coord
            self.success_y = new_y_coord
        else: # if acceleration fails
            self.fail_x = new_x_coord
            self.fail_y = new_y_coord

    def value_policy_update(self):
        """Update the value and policy tables.

        The value_policy_update() function will update
        the value and policy table for a given state.
        """
        # Choose the max Q_t(s,a) for all (a) for current (s)
        max_exp_value = max(self.temp_Q)
        max_indices = [max_idx for max_idx in self.Q_indexer if\
                       self.temp_Q[max_idx] == max_exp_value]
 
        # Randomly select one of the max Q_t(s,a) actions
        max_idx_choice = random.choice(max_indices)
        action_choice = self.possible_actions_list[max_idx_choice]
        pi_update = [int(vel) for vel in self.velocity_list[max_idx_choice]]
        pi_update.extend(action_choice)

        # Update the pi_t(s) to only include directions for argmax
        self.table_pi[self.state_x][self.state_y] = np.array(pi_update)

        # Update the value for state s, given the action taken
        # by policy pi_t(s)
        self.table_V[self.state_x][self.state_y] = max_exp_value

    def train(self):
        """Train the agent on the racetrack.

        The train() function will train the agent on the
        racetrack given the initialized hyperparameters.

        Returns:
            self.pi_table_df (pandas dataframe): The resulting
                state-action determined after training.
        """
        while True: # Repeat until convergence
            self.counter_t += 1 # Update counter

            # Copy the current table for reference
            self.old_table_V = copy.deepcopy(self.table_V)

            # Loop through all states, for all s in S
            for possible_state in self.possible_states_list:
                # Initialize temp variables for selected state
                self.state_x, self.state_y =\
                    possible_state[0], possible_state[1]
                self.velocity_list.clear() # Reset velocity list

                # Loop through all actions, for all a in A
                for action_index in self.actions_range:
                    # Current action
                    action = self.possible_actions_list[action_index]

                    # Calculate 2 possible states, acceleration success/fail
                    self.next_state(local_action=action, success=True)
                    self.next_state(local_action=action, success=False)

                    # Calculate Q_t(s,a)
                    self.temp_Q[action_index] = self.expected_value()

                # Calculate pi_t(s), V_t(s)
                self.value_policy_update()

            # Until max_{s in S} |V_t(s) - V_{t-1}(s)| < error
            if (np.nanmax(abs(self.table_V - self.old_table_V)) < self.error):
                break

        # Make pi-table presentable
        self.pi_table_df = pd.DataFrame(copy.deepcopy(self.table_V))
        for possible_state in self.possible_states_list: # for all s in S
            s_row, s_col = possible_state[0], possible_state[1]
            self.pi_table_df.iloc[s_row, s_col] = ','.join([str(policy) for\
                policy in self.table_pi[s_row][s_col].tolist()])                

        return self.pi_table_df

    def find_pi_state(self, xy_coords):
        """A helper function to test().

        The find_pi_state() function will determine the policy
        pi for a given state.

        Args:
            xy_coords (tuple): The x and y coordinates from
                a state.
        """
        pi_state = self.pi_track[xy_coords[0]][xy_coords[1]].split(',')
        pi_state = [int(float(pi)) for pi in pi_state]
        self.pi_state = pi_state
    
    def test(self, print_results=False):
        """Test the agent on the racetrack after training.

        The test() function will test the agent on the
        racetrack by initializing it at the start coordinates
        and having it follow the policy determined at each possible
        state. The same rules of acceleration failing also apply.

        Args:
            print_results (bool): Indicate whether results should
                be output.
        """
        # Intialize variables
        self.stall_counter_test = 0; self.crash_counter_test = 0
        self.pi_track = copy.deepcopy(self.pi_table_df.to_numpy())
        coords_update = random.choice(self.start_coords)
        self.visited_coords = []
        self.visited_coords.append(coords_update)
        self.finished = False; self.timer = 0

        while True: # Race the agent until it finishes
            self.timer += 1 # Update timer

            # Update policy for a new coordinate
            self.find_pi_state(xy_coords=coords_update)
            action = (self.pi_state[2], self.pi_state[3])
            self.state_x = coords_update[0]; self.state_y = coords_update[1]

            # Test for acceleration failure
            unif_num = random.uniform(0, 1)
            if (unif_num <= 0.8): # Successful accceleration
                self.next_state(local_action=action,
                                success=True, testing_stage=True)
                coords_update = (self.success_x, self.success_y)

            else: # Failed acceleration
                self.stall_counter_test += 1
                self.next_state(local_action=action,
                                success=False, testing_stage=True)
                coords_update = (self.fail_x, self.fail_y)

            # Save coords
            self.visited_coords.append(coords_update)

            if self.finished: # Check if agent crosses finish line
                if print_results:
                    print("Finished:", "coords",
                          self.state_x, self.state_y,
                          "action", action, "update", coords_update)
                break

## 2.2 Q-learning
class q_learning(object):
    """Implement the Q-Learning (QL) algorithm.

    The q_learning() class allows for implementing the
    QL algorithm on the racetrack problem. There are four
    hyperparameters for the QL algorithm and one hyperparameter
    for the racetrack.

    Args:
        see __init__()
    Returns:
        see test()
    """
    def __init__(self, test_board, discount_rate, epsilon, learning_rate,
                 max_episodes, crash_type=1):
        """Initialize variables for the class.

        The __init__() function initializes the variables of the model
        so that the algorithm can function. The discount rate, epsilon,
        learning rate, and max episodes are related to the QL algorithm.
        The crash type is related to the problem, where 1 represents the
        vehicle resetting to the start zone in a crash and 2 represents
        the vehicle resetting to the nearest racetrack location.

        Args:
            test_board (numpy matrix): The racetrack represented within
                a numpy matrix.
            discount_rate (float): The discount rate between [0,1].
            epsilon (float): The epsilon-greedy probability between [0,1].
            learning_rate (float): The learning rate between [0,1].
            max_episodes (int): The number of maximum episodes.
            crash_type (integer): The crash type (1 or 2).
        """
        # Initialize VI hyperparameters
        self.gamma = discount_rate
        self.epsilon = epsilon
        self.eta = learning_rate
        self.tau = max_episodes
        self.crash_type = crash_type
        self.crash_counter_train = 0
        self.crash_counter_test = 0
        self.stall_counter_train = 0
        self.stall_counter_test = 0

        self.all_states = test_board # Related to the board
        board_dim = [dim for dim in test_board.shape]
        self.n_rows = board_dim[0]
        self.n_cols = board_dim[1]

        self.success_x = 0 # Rleated to coordinates
        self.success_y = 0
        self.fail_x = 0
        self.fail_y = 0

        self.curr_action = (0, 0) # Related to state-actions
        self.max_action = (0, 0)
        self.curr_state = (0, 0)
        self.update_state = (0, 0)
        self.reward = -1 # Reward

        start_coords = np.where(test_board == 1) # For initial/reset zone
        self.start_coords = tuple(list(zip(start_coords[0], start_coords[1])))

        possible_states = np.where(test_board != 0) # Possible state coords
        self.possible_states_list = tuple(list(zip(possible_states[0],
                                             possible_states[1])))

        goal_coords = np.where(test_board == 2) # For terminate state
        self.goal_coords = tuple(list(zip(goal_coords[0], goal_coords[1])))
        possible_states = np.where(test_board != 0)

        possible_actions = [-1, 0 ,1] # Possible actions
        self.possible_actions_list = tuple([[accel_x, accel_y] for\
            accel_x in possible_actions for accel_y in possible_actions])
        self.actions_range = range(len(self.possible_actions_list))

        # To update the Q-values
        self.temp_Q = [0] * len(self.possible_actions_list)
        self.Q_indexer = tuple(list(range(len(self.temp_Q))))
        self.velocity_limit = tuple([-5, 5])

        # Initialize Q-table n_col X n_row X (state-action-Q) (5)
        # (state-action-Q) = <velocity_x, velocity_y,
        # acceleration_x, acceleration_y, Q((x,y),a)>
        # (4-dim table for state spaces and possible moves)
        board_dim.append(5)
        self.table_Q = np.zeros(tuple(board_dim))

        # Set boundary positions for Q-table to NaN's
        boundary_states = np.where(test_board == 0)
        for coord in list(zip(boundary_states[0], boundary_states[1])):
            x_coord, y_coord = coord[0], coord[1]
            self.table_Q[x_coord][y_coord] = np.NaN

        # Set boundary states
        boundary_states = np.asarray(boundary_states)
        boundary_states = boundary_states.T
        boundary_states = boundary_states.tolist()
        self.boundary_states = tuple(boundary_states)

    def octants3_4_7_8(self, x_init_local, y_init_local,
                       x_final_local, y_final_local):
        """A helper function to bresenham_line_alg().
        
        The octants3_4_7_8() function calculates the line
        path for the half of the octants where the slope
        is >1.

        Args:
            x_init_local (int): The x-coordinate of the starting pixel.
            y_init_local (int): The y-coordinate of the starting pixel.
            x_final_local (int): The x-coordinate of the ending pixel.
            y_final_local (int): The y-coordinate of the ending pixel. 
        Returns:
            line_path (list): A list of the path taken as a
                series of coordinates.
        """
        # Initialize variables
        delta_x = x_final_local - x_init_local
        delta_y = y_final_local - y_init_local
        y_indexer = 1; line_path = []
        y_temp = y_init_local

        if (delta_y < 0): # Check if slope positive or negative
            y_indexer = -1; delta_y = -delta_y

        coefficient_D = 2*delta_y - delta_x

        # Create forward or backward sequence of integers
        if ((x_init_local - x_final_local) < 0):
            x_range = list(range(x_init_local, x_final_local + 1))
        else:
            x_range = list(reversed(range(x_init_local, x_final_local + 1)))

        for x_idx in x_range: # Loop and create line path
            line_path.append([x_idx, y_temp])
            if (coefficient_D > 0):
                y_temp += y_indexer
                coefficient_D = coefficient_D - 2*delta_x

            coefficient_D = coefficient_D + 2*delta_y

        return line_path
            
    def octants1_2_5_6(self, x_init_local, y_init_local,
                       x_final_local, y_final_local):
        """A helper function to bresenham_line_alg().
        
        The octants1_2_5_6() function calculates the line
        path for the half of the octants where the slope
        is <=1.

        Args:
            x_init_local (int): The x-coordinate of the starting pixel.
            y_init_local (int): The y-coordinate of the starting pixel.
            x_final_local (int): The x-coordinate of the ending pixel.
            y_final_local (int): The y-coordinate of the ending pixel. 
        Returns:
            line_path (list): A list of the path taken as a
                series of coordinates.
        """
        # Initialize variables
        delta_x = x_final_local - x_init_local
        delta_y = y_final_local - y_init_local
        x_indexer = 1; line_path = []
        x_temp = x_init_local

        if (delta_x < 0): # Check if slope positive or negative
            x_indexer = -1; delta_x = -delta_x

        coefficient_D = 2*delta_x - delta_y

        # Create forward or backward sequence of integers
        if ((y_init_local - y_final_local) < 0):
            y_range = list(range(y_init_local, y_final_local + 1))
        else:
            y_range = list(reversed(range(y_init_local, y_final_local + 1)))

        for y_idx in y_range: # Loop and create line path
            line_path.append([x_temp, y_idx])
            if (coefficient_D > 0):
                x_temp += x_indexer
                coefficient_D = coefficient_D - 2*delta_y

            coefficient_D = coefficient_D + 2*delta_x

        return line_path
            
    def bresenham_line_alg(self, x_init, y_init, x_final, y_final):
        """Determine the path taken using the Bresenham
        line algorithm.
        
        The bresenham_line_alg() function determines the path
        taken between to coordinates by using the formula for
        the Bresenham line algorithm. The code is based off
        pseudocode from Wikipedia. Permission is given by the
        professor to do so.

        Args:
            x_init (int): The x-coordinate of the starting pixel.
            y_init (int): The y-coordinate of the starting pixel.
            x_final (int): The x-coordinate of the ending pixel.
            y_final (int): The y-coordinate of the ending pixel.
        Returns:
            line_path (list): A list of the path taken as a
                series of coordinates.
        
        Helper functions: octants1_2_5_6(), octants3_4_7_8()
        https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
        """
        # When slope < 1
        if (abs(y_final - y_init) < abs(x_final - x_init)):
            if (x_init > x_final): # octant 4
                line_path = self.octants3_4_7_8(\
                    x_init_local=x_final, y_init_local=y_final,
                    x_final_local=x_init, y_final_local=y_init)
            else: # octants 3, 7, 8
                line_path = self.octants3_4_7_8(\
                    x_init_local=x_init, y_init_local=y_init,
                    x_final_local=x_final, y_final_local=y_final)
        else: # When slope >= 1
            if (y_init > y_final): # octant 6
                line_path = self.octants1_2_5_6(\
                    x_init_local=x_final, y_init_local=y_final,
                    x_final_local=x_init, y_final_local=y_init)
            else: # octants 2, 5, 7
                line_path = self.octants1_2_5_6(\
                    x_init_local=x_init, y_init_local=y_init,
                    x_final_local=x_final, y_final_local=y_final)

        # Check if line order has been reversed
        if (line_path[0] != [x_init, y_init]):
            line_path.reverse()

        return line_path

    def velocity_update(self, vel_local, accel_local):
        """A helper function to next_state().

        The velocity_update() function will perform the
        velocity update for next_state(). It checks first
        if the current velocity is already either the max
        or min and will update it accordingly.

        Args:
            vel_local (int): The current velocity.
            accel_local (int): The current acceleration.
        Returns:
            vel_local (int): If already max or min.
            vel_local + accel_local (int): The normal
                velocity update.
        """
        # Max velocity
        if ((vel_local == self.velocity_limit[1]) and (accel_local > 0)):
            return vel_local
        # Min velocity
        elif ((vel_local == self.velocity_limit[0]) and (accel_local < 0)):
            return vel_local
        else: # Normal velocity update
            return vel_local + accel_local
        
    def next_state(self, local_action, success=True,
        taking_action=False, testing_stage=False):
        """Determine the next state from a given state.
        
        The next_state() function will find the next state from
        a given state. The algorithm requires at various steps
        for a next state to be determined. This is used during
        both training and testing. There are also conditions
        such as whether or not acceleration has failed or not
        and if action is being taken.

        Args:
            local_action (int): The action to be taken at the
                current state.
            success (bool): Indicate if acceleration has worked.
            taking_action (bool): Indicate if action is being
                taken.
            testing_stage (bool): Indicate if the stage is
                testing or training.
        """
        # Initialize acceleration and velocity variables
        if testing_stage == False: # Training initialization
            state_x, state_y = self.curr_state[0], self.curr_state[1]
            vel_x = self.table_Q[state_x, state_y, 0]
            vel_y = self.table_Q[state_x, state_y, 1]
            accel_x = local_action[0]; accel_y = local_action[1]
        else: # Testing initialization
            state_x, state_y = self.curr_state[0], self.curr_state[1]
            vel_x = self.pi_state[0]; vel_y = self.pi_state[1]
            accel_x = self.pi_state[2]; accel_y = self.pi_state[3]
        
        # Potential next state
        if success: # If acceleration works
            vel_x = self.velocity_update(vel_local=vel_x, accel_local=accel_x)
            vel_y = self.velocity_update(vel_local=vel_y, accel_local=accel_y)
            new_x_coord = int(state_x + vel_x)
            new_y_coord = int(state_y + vel_y)
        else: # If acceleration fails
            new_x_coord = int(state_x + vel_x)
            new_y_coord = int(state_y + vel_y)
        
        line_path = self.bresenham_line_alg(\
            x_init=state_x, y_init=state_y,
            x_final=new_x_coord, y_final=new_y_coord) # Line path

        # Check for either finishing or crashing
        for path_coord_idx in list(range(len(line_path))):
            # Crash case 1, reset to start zone
            if (line_path[path_coord_idx] in self.boundary_states):
                if testing_stage: # Update crash counter
                    self.crash_counter_test += 1
                else:
                    self.crash_counter_train += 1
                if (self.crash_type == 1): # Crash case 1
                    reset_coords = random.choice(self.start_coords)
                    new_x_coord = reset_coords[0]
                    new_y_coord = reset_coords[1]
                    vel_x, vel_y = 0, 0
                    break
                else: # Crash case 2, reset to nearest coords
                    reset_coords = line_path[path_coord_idx - 1]
                    new_x_coord = reset_coords[0]
                    new_y_coord = reset_coords[1]
                    vel_x, vel_y = 0, 0
                    break
            # Check if finish without crashing
            elif (tuple(line_path[path_coord_idx]) in self.goal_coords):
                new_x_coord = line_path[path_coord_idx][0]
                new_y_coord = line_path[path_coord_idx][1]
                # Update flag if taking action or testing
                if (taking_action or testing_stage):
                    self.finished = True
                break

        if (taking_action == True): # Acceleration / velocity update
            # Update current acceleration
            self.table_Q[state_x, state_y, 2] = int(accel_x)
            self.table_Q[state_x, state_y, 3] = int(accel_y)

            # Update next state's velocity
            self.table_Q[new_x_coord, new_y_coord, 0] = int(vel_x)
            self.table_Q[new_x_coord, new_y_coord, 1] = int(vel_y)

        # Update next state coordinates
        if success: # If acceleration works
            self.success_x = new_x_coord
            self.success_y = new_y_coord
        else: # If acceleration fails
            self.fail_x = new_x_coord
            self.fail_y = new_y_coord

    def table_Q_update(self):
        """Update the Q-table for a given state along
        with updating the state.

        The table_Q_update() function will update the
        Q-table for a given state-action pair. It will
        also update the current state (s) to the next
        state (s').
        """
        # Initialize variables
        curr_x, curr_y = self.curr_state[0], self.curr_state[1]

        # Update state (s) to (s')
        self.curr_state = self.update_state

        # Identify current and max Q-values
        curr_Q = copy.deepcopy(self.table_Q[curr_x, curr_y, 4])
        max_Q = self.argmax_Q(update_Q=True)

        # Update the Q-table
        self.table_Q[curr_x, curr_y, 4] = curr_Q + self.eta *\
            (self.reward + self.gamma * max_Q - curr_Q)

    def argmax_Q(self, update_Q=False):
        """Find either the max Q(s,a) or the max action
        for a state (s).

        The argmax_Q() function is used in different
        parts of the algorithm, where it either finds
        the best action to take while at a state (s)
        (where best implies the largest resulting Q-value),
        or it finds the maximum Q-value that can be
        obtained by taking an action (a') while in a
        state (s).

        Args:
            update_Q (bool): Indicate if the current
                step is to update the Q-table.
        Returns:
            max_exp_value (float): The maximum expected
                value that can be found by taking an
                action (a') while in state (s).
            argmax_action (int): The action to be
                taken.
        """
        # Find argmax_{a in A} Q(s,a)
        for action_index in self.actions_range: # for all a in A
            # current action
            action = self.possible_actions_list[action_index]
            self.next_state(local_action=action)

            # calculate Q_t(s,a)
            self.temp_Q[action_index] =\
                self.table_Q[self.success_x, self.success_y, 4]

        max_exp_value = max(self.temp_Q) # Find the max Q-value

        if (update_Q == True): # Return the max Q(s',a') for Q-update
            return max_exp_value

        # Identify the actions that led to the max Q-value
        max_indices = [max_idx for max_idx in self.Q_indexer if\
                       self.temp_Q[max_idx] == max_exp_value]
 
        # Randomly select one of the max Q_t(s,a) actions
        max_idx_choice = random.choice(max_indices)
        argmax_action = self.possible_actions_list[max_idx_choice]

        return argmax_action
        
    def epislon_greedy(self):
        """Apply the epsilon-greedy method of exploration.

        The epislon_greedy() function is used to balance
        exploration vs. exploitation. With probability
        (1 - epsilon), the agent will exploit the space
        and with probability epsilon, it will explore the
        space.
        """
        unif_num = random.uniform(0, 1) # Random number generator
        if (unif_num <= (1 - self.epsilon)): # argmax Q(s,a)
            # Determine the best action (exploitation)
            greedy_action = self.argmax_Q()
        else: # Take a random action (exploration)
            greedy_action = random.choice(self.possible_actions_list)

        self.curr_action = greedy_action # Update (a)

    def take_action(self):
        """Take action (a) while in state (s).

        The take_action() function will take the current
        action (a) while in the current state (s) and
        observe the reward (r) and the update state (s').
        """
        unif_num = random.uniform(0, 1) # Random number
        if (unif_num <= 0.8): # Action works
            self.next_state(local_action=self.curr_action,
                            success=True, taking_action=True)
            self.update_state = [self.success_x, self.success_y]
        else: # Action fails
            self.stall_counter_train += 1
            self.next_state(local_action=self.curr_action,
                            success=False, taking_action=True)
            self.update_state = [self.fail_x, self.fail_y]

        if self.finished == True: # Update (r)
            self.reward = 0
            
    def train(self):
        """Train the agent on the racetrack.

        The train() function will train the agent on the
        racetrack given the initialized hyperparameters.

        Returns:
            self.table_Q_df (pandas dataframe): The resulting
                state-action determined after training.
        """
        for _ in range(self.tau): # Loop through episodes
            # Re-initialize variables
            self.curr_state = random.choice(self.start_coords) # set (s)
            self.finished = False; self.reward = -1
            
            while True: # Repeat until agent crosses finish
                # Choose (a) using policy derived from Q (epsilon-greedy)
                self.epislon_greedy() # (a) self.curr_action

                # Take action (a) while in state (s), observe (r) and (s')
                self.take_action() # self.reward, self.update_state

                # Update Q(s,a) and state (s) to (s')
                self.table_Q_update() # (s) self.curr_state

                # Until (s) is terminal state
                if self.finished:
                    break

        # Represent the Q-table as a pandas dataframe
        self.table_Q_df = pd.DataFrame(np.zeros([self.n_rows, self.n_cols]))
        for possible_state in self.possible_states_list: # for all s in S
            s_row, s_col = possible_state[0], possible_state[1]
            self.table_Q_df.iloc[s_row, s_col] = ','.join([str(q_data) for\
                q_data in self.table_Q[s_row, s_col,:].tolist()])

        return self.table_Q_df
    
    def find_pi_state(self, xy_coords):
        """A helper function to test().

        The find_pi_state() function will determine the policy
        pi for a given state.

        Args:
            xy_coords (tuple): The x and y coordinates from
                a state.
        """
        # Find the policy to take at a given state
        pi_state = copy.deepcopy(\
            self.pi_track[xy_coords[0], xy_coords[1]].split(','))
        pi_state = tuple([int(float(pi)) for pi in pi_state[:-1]])
        self.pi_state = pi_state
    
    def test(self, print_results=False):
        """Test the agent on the racetrack after training.

        The test() function will test the agent on the
        racetrack by initializing it at the start coordinates
        and having it follow the policy determined at each possible
        state. The same rules of acceleration failing also apply.

        Args:
            print_results (bool): Indicate whether results should
                be output.
        """
        # Intialize variables
        self.stall_counter_test = 0; self.crash_counter_test = 0
        self.pi_track = copy.deepcopy(self.table_Q_df.to_numpy())
        self.update_state = random.choice(self.start_coords)
        self.visited_coords = []
        self.visited_coords.append(self.update_state)
        self.finished = False; self.timer = 0

        while True: # Race the agent until it finishes
            self.timer += 1 # Update timer

            # Update policy for a new coordinate
            self.find_pi_state(xy_coords=self.update_state)
            action = tuple([self.pi_state[2], self.pi_state[3]])
            self.curr_state = self.update_state

            unif_num = random.uniform(0, 1) # Random number
            if (unif_num <= 0.8): # Successful accceleration
                self.next_state(local_action=action,
                                success=True, testing_stage=True)
                self.update_state = tuple([self.success_x, self.success_y])
                
            else: # Failed acceleration
                self.stall_counter_test += 1
                self.next_state(local_action=action,
                                success=False, testing_stage=True)
                self.update_state = tuple([self.fail_x, self.fail_y])

            # Save coords
            self.visited_coords.append(self.update_state)

            if self.finished: # Check if agent crosses finish line
                break

## 2.3 SARSA
class sarsa(object):
    """Implement the SARSA (SA) algorithm.

    The sarsa() class allows for implementing the
    SA algorithm on the racetrack problem. There are four
    hyperparameters for the SA algorithm and one hyperparameter
    for the racetrack.

    Args:
        see __init__()
    Returns:
        see test()
    """
    def __init__(self, test_board, discount_rate, epsilon,
                 learning_rate, max_episodes, crash_type=1):
        """Initialize variables for the class.

        The __init__() function initializes the variables of the model
        so that the algorithm can function. The discount rate, epsilon,
        learning rate, and max episodes are related to the QL algorithm.
        The crash type is related to the problem, where 1 represents the
        vehicle resetting to the start zone in a crash and 2 represents
        the vehicle resetting to the nearest racetrack location.

        Args:
            test_board (numpy matrix): The racetrack represented within
                a numpy matrix.
            discount_rate (float): The discount rate between [0,1].
            epsilon (float): The epsilon-greedy probability between [0,1].
            learning_rate (float): The learning rate between [0,1].
            max_episodes (int): The number of maximum episodes.
            crash_type (integer): The crash type (1 or 2).
        """
        # Initialize model variables
        self.gamma = discount_rate
        self.epsilon = epsilon
        self.eta = learning_rate
        self.tau = max_episodes
        self.crash_type = crash_type
        self.crash_counter_train = 0
        self.crash_counter_test = 0
        self.stall_counter_train = 0
        self.stall_counter_test = 0

        self.all_states = test_board # Board variables
        board_dim = [dim for dim in test_board.shape]
        self.n_rows = board_dim[0]
        self.n_cols = board_dim[1]

        self.success_x = 0 # Coordinate variables
        self.success_y = 0
        self.fail_x = 0
        self.fail_y = 0

        self.curr_action = (0, 0) # State-action variables
        self.update_action = (0, 0)
        self.curr_state = (0, 0)
        self.update_state = (0, 0)
        self.reward = -1

        start_coords = np.where(test_board == 1) # For initial/reset zone
        self.start_coords = tuple(list(zip(start_coords[0], start_coords[1])))

        goal_coords = np.where(test_board == 2) # For terminate state
        self.goal_coords = tuple(list(zip(goal_coords[0], goal_coords[1])))

        possible_states = np.where(test_board != 0) # For possible states
        self.possible_states_list = tuple(list(zip(possible_states[0],
                                             possible_states[1])))
        possible_actions = [-1, 0 ,1] # For possible actions
        self.possible_actions_tuple = tuple([[accel_x, accel_y] for\
            accel_x in possible_actions for accel_y in possible_actions])
        self.actions_range = range(len(self.possible_actions_tuple))

        # For updating Q-values
        self.temp_Q = [0] * len(self.possible_actions_tuple)
        self.Q_indexer = tuple(list(range(len(self.temp_Q))))
        self.velocity_limit = (-5, 5)

        # Initialize Q-table n_col X n_row X (state-action-Q) (5)
        # (state-action-Q) = <velocity_x, velocity_y,
        # acceleration_x, acceleration_y, Q((x,y),a)>
        # (4-dim table for state spaces and possible moves)
        board_dim.append(5)
        self.table_Q = np.zeros(tuple(board_dim))

        # Set boundary positions for Q-table to NaN's
        boundary_states = np.where(test_board == 0)
        for coord in list(zip(boundary_states[0], boundary_states[1])):
            x_coord, y_coord = coord[0], coord[1]
            self.table_Q[x_coord][y_coord] = np.NaN

        # For boundary states
        boundary_states = np.asarray(boundary_states)
        boundary_states = boundary_states.T
        boundary_states = boundary_states.tolist()
        self.boundary_states = tuple(boundary_states)
        
    def octants3_4_7_8(self, x_init_local, y_init_local,
                       x_final_local, y_final_local):
        """A helper function to bresenham_line_alg().
        
        The octants3_4_7_8() function calculates the line
        path for the half of the octants where the slope
        is >1.

        Args:
            x_init_local (int): The x-coordinate of the starting pixel.
            y_init_local (int): The y-coordinate of the starting pixel.
            x_final_local (int): The x-coordinate of the ending pixel.
            y_final_local (int): The y-coordinate of the ending pixel. 
        Returns:
            line_path (list): A list of the path taken as a
                series of coordinates.
        """
        # Initialize variables
        delta_x = x_final_local - x_init_local
        delta_y = y_final_local - y_init_local
        y_indexer = 1; line_path = []
        y_temp = y_init_local

        if (delta_y < 0): # Check if slope positive or negative
            y_indexer = -1; delta_y = -delta_y

        coefficient_D = 2*delta_y - delta_x

        # Create forward or backward sequence of integers
        if ((x_init_local - x_final_local) < 0):
            x_range = list(range(x_init_local, x_final_local + 1))
        else:
            x_range = list(reversed(range(x_init_local, x_final_local + 1)))

        for x_idx in x_range: # Loop and create line path
            line_path.append([x_idx, y_temp])
            if (coefficient_D > 0):
                y_temp += y_indexer
                coefficient_D = coefficient_D - 2*delta_x

            coefficient_D = coefficient_D + 2*delta_y

        return line_path
            
    def octants1_2_5_6(self, x_init_local, y_init_local,
                       x_final_local, y_final_local):
        """A helper function to bresenham_line_alg().
        
        The octants1_2_5_6() function calculates the line
        path for the half of the octants where the slope
        is <=1.

        Args:
            x_init_local (int): The x-coordinate of the starting pixel.
            y_init_local (int): The y-coordinate of the starting pixel.
            x_final_local (int): The x-coordinate of the ending pixel.
            y_final_local (int): The y-coordinate of the ending pixel. 
        Returns:
            line_path (list): A list of the path taken as a
                series of coordinates.
        """
        # Initialize variables
        delta_x = x_final_local - x_init_local
        delta_y = y_final_local - y_init_local
        x_indexer = 1; line_path = []
        x_temp = x_init_local

        if (delta_x < 0): # Check if slope positive or negative
            x_indexer = -1; delta_x = -delta_x

        coefficient_D = 2*delta_x - delta_y

        # Create forward or backward sequence of integers
        if ((y_init_local - y_final_local) < 0):
            y_range = list(range(y_init_local, y_final_local + 1))
        else:
            y_range = list(reversed(range(y_init_local, y_final_local + 1)))

        for y_idx in y_range: # Loop and create line path
            line_path.append([x_temp, y_idx])
            if (coefficient_D > 0):
                x_temp += x_indexer
                coefficient_D = coefficient_D - 2*delta_y

            coefficient_D = coefficient_D + 2*delta_x

        return line_path
            
    def bresenham_line_alg(self, x_init, y_init, x_final, y_final):
        """Determine the path taken using the Bresenham
        line algorithm.
        
        The bresenham_line_alg() function determines the path
        taken between to coordinates by using the formula for
        the Bresenham line algorithm. The code is based off
        pseudocode from Wikipedia. Permission is given by the
        professor to do so.

        Args:
            x_init (int): The x-coordinate of the starting pixel.
            y_init (int): The y-coordinate of the starting pixel.
            x_final (int): The x-coordinate of the ending pixel.
            y_final (int): The y-coordinate of the ending pixel.
        Returns:
            line_path (list): A list of the path taken as a
                series of coordinates.
        
        Helper functions: octants1_2_5_6(), octants3_4_7_8()
        https://en.wikipedia.org/wiki/Bresenham's_line_algorithm
        """
        # When slope < 1
        if (abs(y_final - y_init) < abs(x_final - x_init)):
            if (x_init > x_final): # octant 4
                line_path = self.octants3_4_7_8(\
                    x_init_local=x_final, y_init_local=y_final,
                    x_final_local=x_init, y_final_local=y_init)
            else: # octants 3, 7, 8
                line_path = self.octants3_4_7_8(\
                    x_init_local=x_init, y_init_local=y_init,
                    x_final_local=x_final, y_final_local=y_final)
        else: # When slope >= 1
            if (y_init > y_final): # octant 6
                line_path = self.octants1_2_5_6(\
                    x_init_local=x_final, y_init_local=y_final,
                    x_final_local=x_init, y_final_local=y_init)
            else: # octants 2, 5, 7
                line_path = self.octants1_2_5_6(\
                    x_init_local=x_init, y_init_local=y_init,
                    x_final_local=x_final, y_final_local=y_final)

        # Check if line order has been reversed
        if (line_path[0] != [x_init, y_init]):
            line_path.reverse()

        return line_path

    def velocity_update(self, vel_local, accel_local):
        """A helper function to next_state().

        The velocity_update() function will perform the
        velocity update for next_state(). It checks first
        if the current velocity is already either the max
        or min and will update it accordingly.

        Args:
            vel_local (int): The current velocity.
            accel_local (int): The current acceleration.
        Returns:
            vel_local (int): If already max or min.
            vel_local + accel_local (int): The normal
                velocity update.
        """
        # Max velocity
        if ((vel_local == self.velocity_limit[1]) and (accel_local > 0)):
            return vel_local
        # Min velocity
        elif ((vel_local == self.velocity_limit[0]) and (accel_local < 0)):
            return vel_local
        else: # Normal velocity update
            return vel_local + accel_local
        
    def next_state(self, local_action, success=True,
        taking_action=False, testing_stage=False,
        curr_or_update_local='current'):
        """Determine the next state from a given state.
        
        The next_state() function will find the next state from
        a given state. The algorithm requires at various steps
        for a next state to be determined. This is used during
        both training and testing. There are also conditions
        such as whether or not acceleration has failed or not,
        if action is being taken, and if it applies to the
        current or update state.

        Args:
            local_action (int): The action to be taken at the
                current state.
            success (bool): Indicate if acceleration has worked.
            taking_action (bool): Indicate if action is being
                taken.
            testing_stage (bool): Indicate if the stage is
                testing or training.
            curr_or_update_local (string): Indicate whether the
                state is the current or update state.
        """
        # Initialize coords, acceleration and velocity variables
        if (testing_stage == False): # Training initialization
            if (curr_or_update_local == 'current'): # Current state coords
                state_x = self.curr_state[0]; state_y = self.curr_state[1]
            else: # Update state coords
                state_x = self.update_state[0]; state_y = self.update_state[1]
            vel_x = self.table_Q[state_x, state_y, 0]
            vel_y = self.table_Q[state_x, state_y, 1]
            accel_x = local_action[0]; accel_y = local_action[1]
        else: # Testing initialization
            state_x = self.curr_state[0]; state_y = self.curr_state[1]
            vel_x = self.pi_state[0]; vel_y = self.pi_state[1]
            accel_x = self.pi_state[2]; accel_y = self.pi_state[3]
        
        # Potential next state
        if success: # If acceleration works
            vel_x = self.velocity_update(vel_local=vel_x, accel_local=accel_x)
            vel_y = self.velocity_update(vel_local=vel_y, accel_local=accel_y)
            new_x_coord = int(state_x + vel_x)
            new_y_coord = int(state_y + vel_y)
        else: # If acceleration fails
            new_x_coord = int(state_x + vel_x)
            new_y_coord = int(state_y + vel_y)
        
        line_path = self.bresenham_line_alg(\
            x_init=state_x, y_init=state_y,
            x_final=new_x_coord, y_final=new_y_coord) # Line path

        # Check for either finishing or crashing
        for path_coord_idx in list(range(len(line_path))):
            # Crash case 1, reset to start zone
            if (line_path[path_coord_idx] in self.boundary_states):
                if testing_stage: # Update counter
                    self.crash_counter_test += 1
                else:
                    self.crash_counter_train += 1
                if (self.crash_type == 1): # Crash case 1
                    reset_coords = random.choice(self.start_coords)
                    new_x_coord = reset_coords[0]
                    new_y_coord = reset_coords[1]
                    vel_x, vel_y = 0, 0
                    break
                else: # Crash case 2, reset to nearest coords
                    reset_coords = line_path[path_coord_idx - 1]
                    new_x_coord = reset_coords[0]
                    new_y_coord = reset_coords[1]
                    vel_x, vel_y = 0, 0
                    break
            # Check if finish without crashing
            elif (tuple(line_path[path_coord_idx]) in self.goal_coords):
                new_x_coord = line_path[path_coord_idx][0]
                new_y_coord = line_path[path_coord_idx][1]
                # Update flag if taking action
                if (taking_action or testing_stage):
                    self.finished = True
                break

        if (taking_action == True): # Acceleration / velocity update
            # Update current acceleration
            self.table_Q[state_x, state_y, 2] = int(accel_x)
            self.table_Q[state_x, state_y, 3] = int(accel_y)

            # Update next state's velocity
            self.table_Q[new_x_coord, new_y_coord, 0] = int(vel_x)
            self.table_Q[new_x_coord, new_y_coord, 1] = int(vel_y)

        # Update next state coordinates
        if success: # If acceleration works
            self.success_x = new_x_coord
            self.success_y = new_y_coord
        else: # If acceleration fails
            self.fail_x = new_x_coord
            self.fail_y = new_y_coord

    def table_Q_update(self):
        """Update the Q-table for a given state along
        with updating the state.

        The table_Q_update() function will update the
        Q-table for a given state-action pair.
        """
        # Initialize variables
        curr_x = self.curr_state[0]
        curr_y = self.curr_state[1]
        update_x = self.update_state[0]
        update_y = self.update_state[1]
        curr_Q = copy.deepcopy(self.table_Q[curr_x, curr_y, 4])
        update_Q = copy.deepcopy(self.table_Q[update_x, update_y, 4])

        # Update the Q-table
        self.table_Q[curr_x, curr_y, 4] = curr_Q + self.eta *\
            (self.reward + self.gamma * update_Q - curr_Q)

    def argmax_Q(self, curr_or_update_argmax):
        """Find either the max Q(s,a) or the max action
        for a state (s).

        The argmax_Q() function is used to find the best
        possible action to take from a given state, where
        the resulting action would lead to a new state
        that has the highest Q-value. In case of ties, a
        random choice is made between the actions that
        lead to the maximum Q-value.

        Args:
            curr_or_update_argmax (string): Indicate if the
                next_step() function is to search based on
                the current or update state.
        Returns:
            argmax_action (int): The action to be
                taken.
        """
        # Loop through all actions, for all a in A
        for action_index in self.actions_range:
            # Current action
            action = self.possible_actions_tuple[action_index]
            self.next_state(local_action=action, success=True,
                            curr_or_update_local=curr_or_update_argmax)

            # Lookup Q_t(s,a)
            self.temp_Q[action_index] =\
                copy.deepcopy(self.table_Q[self.success_x, self.success_y, 4])

        max_exp_value = max(self.temp_Q) # Find the max Q-value
        max_indices = [max_idx for max_idx in self.Q_indexer if\
                       self.temp_Q[max_idx] == max_exp_value]
 
        # Randomly select one of the max Q_t(s,a) actions
        max_idx_choice = random.choice(max_indices)
        argmax_action = self.possible_actions_tuple[max_idx_choice]

        return argmax_action
        
    def epislon_greedy(self, curr_or_update):
        """Apply the epsilon-greedy method of exploration.

        The epislon_greedy() function is used to balance
        exploration vs. exploitation. With probability
        (1 - epsilon), the agent will exploit the space
        and with probability epsilon, it will explore the
        space.

        Args:
            curr_or_update (string): Indicate whether the
                state is the current or update state.
        """
        unif_num = random.uniform(0, 1) # Random number
        if (unif_num <= (1 - self.epsilon)): # Acceleration works
            greedy_action = self.argmax_Q(curr_or_update_argmax=curr_or_update)
        else: # Acceleration fails
            greedy_action = random.choice(self.possible_actions_tuple)

        # Update either the current or update action
        if (curr_or_update == 'current'): # (a)
            self.curr_action = greedy_action
        else: # (a')
            self.update_action = greedy_action

    def take_action(self):
        """Take action (a) while in state (s).

        The take_action() function will take the current
        action (a) while in the current state (s) and
        observe the reward (r) and the update state (s').
        """
        unif_num = random.uniform(0, 1) # Random number
        if (unif_num <= 0.8): # Action works
            self.next_state(local_action=self.curr_action,
                            success=True, taking_action=True)
            self.update_state = [self.success_x, self.success_y]
        else: # Action fails
            self.stall_counter_train += 1
            self.next_state(local_action=self.curr_action,
                            success=False, taking_action=True)
            self.update_state = [self.fail_x, self.fail_y]

        if self.finished == True: # Update (r)
            self.reward = 0
            
    def train(self):
        """Train the agent on the racetrack.

        The train() function will train the agent on the
        racetrack given the initialized hyperparameters.

        Returns:
            self.table_Q_df (pandas dataframe): The resulting
                state-action determined after training.
        """
        for _ in range(self.tau): # Loop through episodes
            # Re-initialize variables
            self.curr_state = random.choice(self.start_coords)
            self.finished = False; self.reward = -1

            # Choose (a) using policy derived from Q (epsilon-greedy)
            self.epislon_greedy(curr_or_update='current')
            
            while True: # Repeat until agent crosses finish
                # Take action (a) while in state (s), observe (r) and (s')
                # It will also update velocity_x,y (for s') and
                # acceleration_x,y (for s)
                self.take_action()

                # Choose (a') using policy derived from Q (epsilon-greedy)
                self.epislon_greedy(curr_or_update='update')

                # Update the Q-table
                self.table_Q_update()

                # Update state and actions (s) <- (s'); (a) <- (a')
                self.curr_state = self.update_state
                self.curr_action = self.update_action

                # Until (s) is terminal state
                if self.finished:
                    break

        # Represent the Q-table as a pandas dataframe
        self.table_Q_df = pd.DataFrame(np.zeros([self.n_rows, self.n_cols]))
        for possible_state in self.possible_states_list: # for all s in S
            s_row, s_col = possible_state[0], possible_state[1]
            self.table_Q_df.iloc[s_row, s_col] = ','.join([str(q_data) for\
                q_data in self.table_Q[s_row, s_col,:].tolist()])

        return self.table_Q_df

    def find_pi_state(self, xy_coords):
        """A helper function to test().

        The find_pi_state() function will determine the policy
        pi for a given state.

        Args:
            xy_coords (tuple): The x and y coordinates from
                a state.
        """
        # Find the policy to take at a given state
        pi_state = copy.deepcopy(self.pi_track[xy_coords[0],
                                               xy_coords[1]].split(','))
        pi_state = tuple([int(float(pi)) for pi in pi_state[:-1]])
        self.pi_state = pi_state
    
    def test(self, print_results=False):
        """Test the agent on the racetrack after training.

        The test() function will test the agent on the
        racetrack by initializing it at the start coordinates
        and having it follow the policy determined at each possible
        state. The same rules of acceleration failing also apply.

        Args:
            print_results (bool): Indicate whether results should
                be output.
        """
        # Intialize variables
        self.stall_counter_test = 0; self.crash_counter_test = 0
        self.pi_track = copy.deepcopy(self.table_Q_df.to_numpy())
        self.update_state = random.choice(self.start_coords)
        self.visited_coords = []
        self.visited_coords.append(self.update_state)
        self.finished = False; self.timer = 0

        while True: # Race the agent until it finishes
            self.timer += 1 # Update timer

            # Update policy for a new coordinate
            self.find_pi_state(xy_coords=self.update_state)
            action = tuple([self.pi_state[2], self.pi_state[3]])
            self.curr_state = self.update_state

            unif_num = random.uniform(0, 1) # Random number
            if (unif_num <= 0.8): # Successful accceleration
                self.next_state(local_action=action,
                                success=True, testing_stage=True)
                self.update_state = tuple([self.success_x, self.success_y])

            else: # Failed acceleration
                self.stall_counter_test += 1
                self.next_state(local_action=action,
                                success=False, testing_stage=True)
                self.update_state = tuple([self.fail_x, self.fail_y])

            # Save coords
            self.visited_coords.append(self.update_state)

            if (self.finished): # Check if agent crosses finish line
                break

# Plot racers
def plot_paths(model_list, plot_title, seed_list, mkr_size=20,
                size_divide=2, alpha_level=1,
                img_name='/static/images/new_plot.png'):
    """Plot the racetrack and resulting paths for the agent.

    The plot_paths() function will take in a trained
    model and test it over a range of random seeds. It
    will plot the racetrack and the resulting paths taken
    by the agent on each run through the racetrack. It
    will also print the following statistics: mean,
    median, max, and min for the number of steps taken
    for the agent to finish.

    Args:
        model (class object): A trained model.
        plot_title (string): The title for the plot.
        seeds (range(x)): A range of numbers for the
            random number generator.
        size_divide (number): The value to divide the
            plot dimensions by to reduce the size.
        img_name (string): Name of file to save.
        alpha_level (number): A number for the
            transparency of the lines in the plot.
    Returns:
        [total_races, time_stats, stall_stats,
        crash_stats] (list): A list of the
            statistics for the agent.
    """
    # Initialize variables
    size_y = model_list[0].n_rows / size_divide
    size_x = model_list[0].n_cols / size_divide
    fig, ax = plt.subplots(figsize=(size_x, size_y))
    fig.suptitle(plot_title, fontsize=20)
    ax.invert_yaxis()

    # Identify the boundary, start, and end
    # coordinates on the racetrack
    boundary_states = model_list[0].boundary_states
    start_coords = model_list[0].start_coords
    goal_coords = model_list[0].goal_coords

    for idx in range(len(boundary_states)): # Plot boundaries
        ax.plot(boundary_states[idx][1],
                boundary_states[idx][0], 's',
                color='brown', markersize=mkr_size)

    for idx in range(len(start_coords)): # Plot start zone
        ax.plot(start_coords[idx][1],
                start_coords[idx][0], 's', color='red')

    for idx in range(len(goal_coords)): # Plot finish zone
        ax.plot(goal_coords[idx][1],
                goal_coords[idx][0], 's', color='red')

    lines = []
    for model_idx in list(range(len(model_list))):
        seed = seed_list[model_idx]
        random.seed(seed)
        model = model_list[model_idx]
        model.test(print_results=False)
        test_path = model.visited_coords

        x_coords, y_coords = [], [] # Plot each path
        for coord in test_path:
            x_coords.append(coord[0])
            y_coords.append(coord[1])
        
        # Check if beta=1 or 2
        if (model.crash_type == 1):
            lines += ax.plot(y_coords, x_coords, 'o',
                    linestyle='dashed', alpha=alpha_level)
        else:
            lines += ax.plot(y_coords, x_coords, 'o',
                    linestyle='solid', alpha=alpha_level)
    plt.legend(lines, ('Your car',
        str('Agent ' + str(seed_list[1])),
        str('Agent ' + str(seed_list[2]))),
        bbox_to_anchor=(1, 0.7))

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(figfile.getvalue()).decode()

    return figdata_png
    # plt.savefig(img_name)

if __name__ == "__main__":
    # 3. L-track
    # 3.1 Value Iteration
    ## Crash type 1
    random.seed(1)
    v1_a = value_iteration(test_board=l_track_np, discount_rate=0.7,
                            bellman_error=0.001, crash_type=1)
    _ = v1_a.train()

    ## Crash type 2
    random.seed(1)
    v2_a = value_iteration(test_board=l_track_np, discount_rate=0.8,
                            bellman_error=0.01, crash_type=2)
    _ = v2_a.train()

    # 3.2 Q-learning
    ## Crash type 1
    random.seed(1)
    q1_e = q_learning(test_board=l_track_np, discount_rate=0.7, epsilon=0.1,
                    learning_rate=0.1, max_episodes=2000, crash_type=1)
    _ = q1_e.train()

    ## Crash type 2
    random.seed(1)
    q2_b = q_learning(test_board=l_track_np, discount_rate=0.7, epsilon=0.1,
                    learning_rate=0.1, max_episodes=1000, crash_type=2)
    _ = q2_b.train()

    # 3.3 SARSA
    ## Crash type 1
    random.seed(1)
    s1_b = sarsa(test_board=l_track_np, discount_rate=0.85, epsilon=0.1,
                learning_rate=0.1, max_episodes=1000, crash_type=1)
    _ = s1_b.train()

    ## Crash type 2
    random.seed(1)
    s2_f = sarsa(test_board=l_track_np, discount_rate=0.6, epsilon=0.1,
                learning_rate=0.1, max_episodes=2000, crash_type=2)
    _ = s2_f.train()

    # 4. O-track
    ### Crash type 1
    random.seed(1)
    v3_d = value_iteration(test_board=o_track_np, discount_rate=0.6,
                            bellman_error=0.01, crash_type=1)
    _ = v3_d.train()

    ### Crash type 2
    random.seed(1)
    v4_a = value_iteration(test_board=o_track_np, discount_rate=0.9,
                            bellman_error=0.1, crash_type=2)
    _ = v4_a.train()

    ## 4.2 Q-learning
    ### Crash type 1
    random.seed(1)
    q3_a = q_learning(test_board=o_track_np, discount_rate=0.9, epsilon=0.1,
                    learning_rate=0.1, max_episodes=1000, crash_type=1)
    _ = q3_a.train()

    ### Crash type 2
    random.seed(1)
    q4_b = q_learning(test_board=o_track_np, discount_rate=0.9, epsilon=0.1,
                    learning_rate=0.1, max_episodes=2000, crash_type=2)
    _ = q4_b.train()

    # 4.3 SARSA
    ## Crash type 1
    random.seed(1)
    s3 = sarsa(test_board=o_track_np, discount_rate=0.9, epsilon=0.1,
                learning_rate=0.1, max_episodes=1000, crash_type=1)
    _ = s3.train()

    ### Crash type 2
    random.seed(1)
    s4_a = sarsa(test_board=o_track_np, discount_rate=0.8, epsilon=0.1,
                learning_rate=0.1, max_episodes=1000, crash_type=2)
    _ = s4_a.train()

    # 5. R-track
    ## 5.1 Value Iteration
    ### Crash type 1
    random.seed(1)
    v5_a = value_iteration(test_board=r_track_np, discount_rate=0.9,
                            bellman_error=0.001, crash_type=1)
    _ = v5_a.train()

    ### Crash type 2
    random.seed(1)
    v6_a = value_iteration(test_board=r_track_np, discount_rate=0.9,
                            bellman_error=0.01, crash_type=2)
    _ = v6_a.train()

    # 5.2 Q-learning
    ### Crash type 1
    random.seed(1)
    q5_d = q_learning(test_board=r_track_np, discount_rate=0.7, epsilon=0.1,
                    learning_rate=0.1, max_episodes=1000, crash_type=1)
    _ = q5_d.train()

    ### Crash type 2
    random.seed(1)
    q6_a = q_learning(test_board=r_track_np, discount_rate=0.8, epsilon=0.1,
                    learning_rate=0.1, max_episodes=1000, crash_type=2)
    _ = q6_a.train()

    ## 5.3 SARSA
    ### Crash type 1
    random.seed(1)
    s5_b = sarsa(test_board=r_track_np, discount_rate=0.9, epsilon=0.1,
                learning_rate=0.1, max_episodes=2000, crash_type=1)
    _ = s5_b.train()

    ### Crash type 2
    random.seed(1)
    s6_a = sarsa(test_board=r_track_np, discount_rate=0.8, epsilon=0.1,
                learning_rate=0.1, max_episodes=1000, crash_type=2)
    _ = s6_a.train()

    ### Pickle the models
    model_data = [v2_a, v1_a, q2_b, q1_e, s2_f, s1_b, v4_a, v3_d, q4_b, q3_a,
        s4_a, s3, v6_a, v5_a, q6_a, q5_d, s6_a, s5_b]
    
    PIK = "pickle.dat"
    with open(PIK, 'wb') as f:
        pickle.dump(model_data, f)
    # # L-track
    # pickle.dump(v2_a, open('vi_l_1.pkl', 'wb'))
    # pickle.dump(v1_a, open('vi_l_2.pkl', 'wb'))

    # pickle.dump(q2_b, open('ql_l_1.pkl', 'wb'))
    # pickle.dump(q1_e, open('ql_l_2.pkl', 'wb'))

    # pickle.dump(s2_f, open('sa_l_1.pkl', 'wb'))
    # pickle.dump(s1_b, open('sa_l_2.pkl', 'wb'))

    # # O-track
    # pickle.dump(v4_a, open('vi_o_1.pkl', 'wb'))
    # pickle.dump(v3_d, open('vi_o_2.pkl', 'wb'))

    # pickle.dump(q4_b, open('ql_o_1.pkl', 'wb'))
    # pickle.dump(q3_a, open('ql_o_2.pkl', 'wb'))

    # pickle.dump(s4_a, open('sa_o_1.pkl', 'wb'))
    # pickle.dump(s3, open('sa_o_2.pkl', 'wb'))

    # # R-track
    # pickle.dump(v6_a, open('vi_r_1.pkl', 'wb'))
    # pickle.dump(v5_a, open('vi_r_2.pkl', 'wb'))

    # pickle.dump(q6_a, open('ql_r_1.pkl', 'wb'))
    # pickle.dump(q5_d, open('ql_r_2.pkl', 'wb'))

    # pickle.dump(s6_a, open('sa_r_1.pkl', 'wb'))
    # pickle.dump(s5_b, open('sa_r_2.pkl', 'wb'))

    # print("Finished training.")