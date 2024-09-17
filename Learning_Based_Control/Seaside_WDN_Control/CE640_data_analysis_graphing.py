import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
#%%
folder = r"E:\ce640_results"
rt = [100, 250, 500, 1000, 2500, 5000, 10000]

situations = ['no_evac','tsu_evac','bldg_evac']
sit_colors = ['green','blue','black']
metrics = ['demand_ratio','avg_pressure','leak_demand']
y_axis_label = {'demand_ratio':'Fraction of demand not met',
                'avg_pressure':'Average Pressure (PSI)',
                'leak_demand':'Loss due to leaks (GPM)'}
for metric in metrics:
    
    avg_score_list = []
    percentile_list = []
    for index, ret_prd in enumerate(rt):
        data_path = os.path.join(folder, f'{ret_prd}yr')
        with open(os.path.join(data_path, 'results.pkl'), 'rb') as file:
            loaded_data = pickle.load(file)
        avg_score_list.append([])
        percentile_list.append([])
        for situation in situations:
            if metric == 'demand_ratio':
                considered_list = [1-x for x in loaded_data[situation][metric] if not np.isnan(x)]
            else:
                considered_list = [x for x in loaded_data[situation][metric] if not np.isnan(x)]
            if metric == 'avg_pressure':
                considered_list = [max(x,0) for x in considered_list]
            avg_score = sum(considered_list)/len(considered_list)
            percentiles = list(np.percentile(considered_list, [25, 75]))
            # percentiles = np.percentile(loaded_data[situation][metric], [5, 95])
            avg_score_list[index].append(avg_score)
            percentile_list[index].append(percentiles)
    # Plot
    plt.figure()
    for index, situation in enumerate(situations):
        # scatter
        plt.scatter(rt, [x[index] for x in avg_score_list], label=situation, color=sit_colors[index])
        # connect lines
        plt.plot(rt, [x[index] for x in avg_score_list], linestyle='-', color=sit_colors[index])
        # Plot upper and lower bounds
        plt.fill_between(rt, [x[index][0] for x in percentile_list], [x[index][1] for x in percentile_list], color=sit_colors[index], alpha=0.3)

    
    # Set logarithmic x-axis scale
    plt.xscale('log')
    # Add labels, legend, and title
    plt.xlabel('Return Year')
    plt.ylabel(y_axis_label[metric])
    plt.legend()
    # Show plot
    plt.grid(True)
    plt.show()