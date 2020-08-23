import numpy as np
import pandas as pd
from flask import Flask, request, render_template, Markup
from flask_table import Table, Col
import pickle
import random
import matplotlib.pyplot as plt
from matplotlib import use
use('Agg')
from io import BytesIO
import base64
from model import value_iteration
from model import q_learning
from model import sarsa
from model import plot_paths

app = Flask(__name__, template_folder="templates")
if __name__ == "__main__":
    # L-track
    vi_l_1 = pickle.load(open('vi_l_1.pkl', 'rb'))
    vi_l_2 = pickle.load(open('vi_l_2.pkl', 'rb'))

    ql_l_1 = pickle.load(open('ql_l_1.pkl', 'rb'))
    ql_l_2 = pickle.load(open('ql_l_2.pkl', 'rb'))

    sa_l_1 = pickle.load(open('sa_l_1.pkl', 'rb'))
    sa_l_2 = pickle.load(open('sa_l_2.pkl', 'rb'))

    # O-track
    vi_o_1 = pickle.load(open('vi_o_1.pkl', 'rb'))
    vi_o_2 = pickle.load(open('vi_o_2.pkl', 'rb'))

    ql_o_1 = pickle.load(open('ql_o_1.pkl', 'rb'))
    ql_o_2 = pickle.load(open('ql_o_2.pkl', 'rb'))

    sa_o_1 = pickle.load(open('sa_o_1.pkl', 'rb'))
    sa_o_2 = pickle.load(open('sa_o_2.pkl', 'rb'))

    # R-track
    vi_r_1 = pickle.load(open('vi_r_1.pkl', 'rb'))
    vi_r_2 = pickle.load(open('vi_r_2.pkl', 'rb'))

    ql_r_1 = pickle.load(open('ql_r_1.pkl', 'rb'))
    ql_r_2 = pickle.load(open('ql_r_2.pkl', 'rb'))

    sa_r_1 = pickle.load(open('sa_r_1.pkl', 'rb'))
    sa_r_2 = pickle.load(open('sa_r_2.pkl', 'rb'))

# Categorize the agents into lists
# Crash type -> Track type -> Car type
# [crash1_list, crash2_list]
# crash1_list:[l_track_list, o_track_list, r_track_list]
agent_list = [[[vi_l_1, ql_l_1, sa_l_1],
                [vi_o_1, ql_o_1, sa_o_1],
                [vi_r_1, ql_r_1, sa_r_1]],
            [[vi_l_2, ql_l_2, sa_l_2],
                [vi_o_2, ql_o_2, sa_o_2],
                [vi_r_2, ql_r_2, sa_r_2]]]

# Declare your table
class ItemTable(Table):
    # place = Col('Place')
    name = Col('Drivers')
    time_taken = Col('Time')
    num_stalls = Col('Stalls')
    num_crashes = Col('Crashes')

# Get some objects
class Item(object):
    def __init__(self, name, time_taken, num_stalls, num_crashes):
        # self.place = place
        self.name = name
        self.time_taken = time_taken
        self.num_stalls = num_stalls
        self.num_crashes = num_crashes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/race', methods=['POST'])
def race():
    # Get user input
    user_inputs = [int(idx) for idx in request.form.values()]
    crash_type = user_inputs[0] # Determine crash type 1 or 2
    map_type = user_inputs[1] # Determine map type
    car_model = user_inputs[2] # Determine car model
    car_number = user_inputs[3] # Determine random seed

# Catch errors

    # Agent setup
    agent_setup = agent_list[crash_type][map_type]
    user_car = agent_setup[car_model]
    computer_cars = [rl_model for rl_model in agent_setup if\
        rl_model != user_car]
    car_list = [user_car]
    car_list.extend(computer_cars)
    agent_seeds = random.sample([seed_num for seed_num in range(100) if\
        seed_num != 3], 2)
    agent_seeds.insert(0, car_number)
    # Race the cars
    plot_url = plot_paths(model_list=car_list,
        plot_title="Race Results",
        seed_list=agent_seeds, mkr_size=20,
        size_divide=2, alpha_level=1)

    model_plot = Markup('<img src="data:image/png;base64,{}" width="70%" height="70%">'.format(plot_url))

    time_list = []; stall_crash_list = []
    for model_idx in range(3):
        seed = agent_seeds[model_idx]
        random.seed(seed)
        model = car_list[model_idx]
        model.test(print_results=False)
        test_path = model.visited_coords
        time_taken = len(test_path)
        time_list.append(time_taken)
        stall_counter = model.stall_counter_test
        crash_counter = model.crash_counter_test
        stall_crash_list.append([stall_counter, crash_counter])

    stall_crash_df = pd.DataFrame(stall_crash_list)
    stalls = stall_crash_df.iloc[:,0]
    crashes = stall_crash_df.iloc[:,1]

### Show race statistics per car
# Number of crashes, steps taken
    items = [Item('You',
                time_list[0], stalls[0], crashes[0]),
            Item(str('Agent ' + str(agent_seeds[1])),
                time_list[1], stalls[1], crashes[1]),
            Item(str('Agent ' + str(agent_seeds[2])),
                time_list[2], stalls[2], crashes[2])]

    # Populate the table
    table = ItemTable(items)

    return render_template('finished.html', model_plot=model_plot, table=table)


if __name__ == "__main__":
    app.run(debug=True)