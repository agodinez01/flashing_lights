import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats

myPath = r'C:/Users/angie/Box/Bike_lights_project/code/flashing_lights/data/'
figPath = r'C:/Users/angie/Box/Bike_lights_project/code/flashing_lights/figs/'

os.chdir(myPath)

data = pd.read_csv('subPSE_med.csv')
occlusion_data = pd.read_csv('dataForOcclusion.csv')

# Add meaningful labels to the data
data['session'] = data['session'].map({1:'No-flashing', 2:'6Hz-flashing'})
data['direction'] = data['direction'].map({1:'Lateral', 2:'In-depth'})
data['speed'] = data['speed'].map({1:'Slow', 2:'Medium', 3:'Fast'})
data['group'] = data['group'].map({0:'Low', 1:'Mid', 2:'High', 3:'None'})

drop_index = data[data['group'] == 'Mid'].index
data.drop(drop_index, inplace=True)

occlusion_data['direction'] = occlusion_data['direction'].map({1:'Lateral', 2:'Lateral', 3:'In-depth', 4:'In-depth', 5:'In-depth', 6:'In-depth'})
occlusion_data['speed'] = occlusion_data['speed'].map({3.75:'Slow', 7.5:'Medium', 15:'Fast'})
occlusion_data['session'] = occlusion_data['session'].map({2:'No-flashing', 3:'6Hz-flashing'})
occlusion_data['group'] = occlusion_data['group'].map({0:'Low', 1:'Mid', 2:'High', 3:'None'})

drop_index = occlusion_data[occlusion_data['group'] == 'Mid'].index
occlusion_data.drop(drop_index, inplace=True)

subjects = data['subject'].unique()
speeds = data['speed'].unique()
directions = data['direction'].unique()
sessions = data['session'].unique()
groups = data['group'].unique()

color_list_g = ['#b3b3b3','#8c8c8c', '#666666']
legend_g = ['#b3b3b3','#8c8c8c', '#666666']
color_blues = ['#99c2ff', '#66a3ff', '#3385ff']
colors = ['#99ccff', '#ffccff', '#ccffcc']
titles = ['Slow', 'Medium', 'Fast']
gap_size = 0.1
x_pos = [-0.25, 1, 2.25]
x_pos2 = [0, 1, 2]

tick_fontsize = 12
label_fontsize = 16

def makeFlatList(input_list):
    flatL = []

    for idx, itemL in enumerate(input_list):
        flat_list = [item for sublist in input_list[idx] for item in sublist]
        flatL.append(flat_list)

    return flatL

# Calculate 95% confidence intervals for each direction
def ci_95(series):
    # Sample mean
    mean = series.mean()
    # Standard error of the mean
    sem = stats.sem(series)
    # Degrees of freedom
    n = len(series) - 1
    # 95% confidence interval using t-distribution
    ci_range = sem * stats.t.ppf((1 + 0.95) / 2., n)
    return pd.Series([mean - ci_range, mean + ci_range], index=['lower', 'upper'])

# Plot the data with speed as columns and direction as rows. Each plot should contain PSE's for each session as a category and median split in different colors
fig, axes = plt.subplots(2,3, figsize=(16, 8), gridspec_kw={'wspace':0.3, 'hspace':0.1})
for ax, dir_ in enumerate(directions):
    for ax2, sp_ in enumerate(speeds):

        data_subset = data.loc[(data['speed'] == sp_) & (data['direction'] == dir_)]

        # Plot reference line at 1
        axes[ax, ax2].axhline(y=1.0, linestyle='--', color='#999999')

        # Overlay swarmplot, adjusting for split violins
        sns.swarmplot(
            ax=axes[ax, ax2],
            y='pse',
            x='group',
            hue='group',
            data=data_subset,
            palette=color_list_g,  # Match swarmplot color with direction color
            edgecolor=None,  # Set the edge color of the swarm points
            size=5,  # Adjust the size of the points if needed
            alpha=0.6,
            dodge=True,  # This will ensure the points are separated based on hue
            hue_order=groups  # Make sure the hue order matches
        )

        # Calculate the mean PSE for each combination of session and direction
        mean_pse = data_subset.groupby(['group'])['pse'].mean().reset_index()

        conf_intervals = data_subset.groupby(['group'])['pse'].apply(ci_95).reset_index()

        for idx, g in enumerate(groups):

            subset = mean_pse[mean_pse['group'] == g]
            subset_cis = conf_intervals[conf_intervals['group'] == g]

            if len(subset) == 0:
                continue
            else:
                axes[ax, ax2].plot(
                    x_pos[idx],
                    subset['pse'].iloc[0],
                    colors[idx],
                    marker='o',
                    markersize=8,
                    linewidth=2.5
                )

                # Add vertical lines for confidence intervals at each speed level
                axes[ax, ax2].plot(
                    [x_pos[idx], x_pos[idx]],  # X-position (fixed for each session)
                    [subset_cis.pse[(subset_cis['group'] == g) & (subset_cis['level_1'] == 'lower')],
                     subset_cis.pse[(subset_cis['group'] == g) & (subset_cis['level_1'] == 'upper')]],
                    # Y-values for CI
                    color=colors[idx],  # Match the color of the median line
                    linewidth=2.5  # Adjust line width for the confidence intervals
                )

        axes[ax, ax2].set_ylim(0, 2.5)

        axes[ax, ax2].get_legend().remove()
        sns.despine()

        plt.setp(axes[ax, ax2].get_yticklabels(), fontsize=tick_fontsize)
        plt.setp(axes[ax, ax2].get_xticklabels(), fontsize=tick_fontsize)

        if [ax, ax2] == [0, 0]:
            axes[ax, ax2].set_ylabel('Lateral PSE', fontsize=label_fontsize)
        elif [ax, ax2] == [1, 0]:
            axes[ax, ax2].set_ylabel('In-depth PSE', fontsize=label_fontsize)
        else:
            axes[ax, ax2].set_ylabel('')

        if ax2 == 1:
            axes[ax, ax2].set_xlabel('Session', fontsize=label_fontsize)
        else:
            axes[ax, ax2].set_xlabel('', fontsize=label_fontsize)

        # Set the title
        if ax == 0:
            axes[ax, ax2].set_title(titles[ax2], fontsize=label_fontsize)

fig_name = 'median_split_occlusion_swarm.svg'
plt.savefig(fname=figPath + fig_name, bbox_inches='tight', format='svg', dpi=300)

mean_pse = data.groupby(['group'])['pse'].mean().reset_index()
sd_pse = data.groupby(['group'])['pse'].std().reset_index()

def makeTable():
    sessVal = []
    dirVal = []
    speedVal = []
    low_mean = []
    high_mean = []
    low_sd = []
    high_sd = []

    for k, sess in enumerate(occlusion_data['session'].unique()):
        for i, dir_ in enumerate(occlusion_data['direction'].unique()):
            for j, sp_ in enumerate(occlusion_data['speed'].unique()):
                subset_data = occlusion_data.loc[(occlusion_data['direction'] == dir_) & (occlusion_data['speed'] == sp_) & (occlusion_data['session'] == sess)]

                low_mean_val = subset_data.totalOccluded[subset_data['group'] == 'Low'].mean()
                low_sd_val = subset_data.totalOccluded[subset_data['group'] == 'Low'].std()
                high_mean_val = subset_data.totalOccluded[subset_data['group'] == 'High'].mean()
                high_sd_val = subset_data.totalOccluded[subset_data['group'] == 'High'].std()

                sessVal.append([sess])
                dirVal.append([dir_])
                speedVal.append([sp_])
                low_mean.append([low_mean_val])
                high_mean.append([high_mean_val])
                low_sd.append([low_sd_val])
                high_sd.append([high_sd_val])

    return sessVal, dirVal, speedVal, low_mean, low_sd, high_mean, high_sd
sessList, dirList, speedList, lowMeanList, lowSdList, highMeanList, highSdList = makeTable()

listOfLists = [sessList, dirList, speedList,lowMeanList, lowSdList, highMeanList, highSdList]
flatL = makeFlatList(listOfLists)

frame = {'session':flatL[0], 'direction':flatL[1], 'speed':flatL[2], 'low_mean':flatL[3], 'low_sd':flatL[4], 'high_mean':flatL[5], 'high_sd':flatL[6]}
df = pd.DataFrame(frame)




# In the 3D/ Fast group, is there by chance a difference in actual trial speed between the 'Low' and 'High'
subData = speed_occlusion_data.loc[(speed_occlusion_data['session'] == 'Continuous') | (speed_occlusion_data['session'] == '6Hz')]
subData = subData.loc[(subData['speed'] == 'Fast') & (subData['direction'] == '3D')]
# Ensure the 'group' column has the correct categorical order
subData['group'] = pd.Categorical(subData['group'], categories=groups, ordered=True)

# Plot the trial speed as a function of # of additional frames occluded
fig, axes = plt.subplots(1,1, figsize=(8, 6), gridspec_kw={'wspace':0.3, 'hspace':0.1})
# Overlay swarmplot, adjusting for split violins
sns.swarmplot(
    ax=axes,
    y='trialSpeed',
    x='group',
    hue='group',
    data=subData,
    palette=color_list_g,  # Match swarmplot color with direction color
    edgecolor=None,  # Set the edge color of the swarm points
    size=5,  # Adjust the size of the points if needed
    alpha=0.6,
    dodge=True,  # This will ensure the points are separated based on hue
    hue_order=groups  # Make sure the hue order matches
)

# Calculate the mean PSE for each combination of session and direction
mean_speed = subData.groupby(['group'])['trialSpeed'].mean().reset_index()

conf_intervals = subData.groupby(['group'])['trialSpeed'].apply(ci_95).reset_index()

for idx, g in enumerate(groups):
    subset = mean_speed[mean_speed['group'] == g]
    subset_cis = conf_intervals[conf_intervals['group'] == g]

    axes.plot(
        x_pos[idx],
        subset['trialSpeed'],
        colors[idx],
        marker='o',
        markersize=8,
        linewidth=2.5
    )

    # Add vertical lines for confidence intervals at each speed level
    axes.plot(
        [x_pos[idx], x_pos[idx]],  # X-position (fixed for each session)
        [subset_cis.trialSpeed[(subset_cis['group'] == g) & (subset_cis['level_1'] == 'lower')],
            subset_cis.trialSpeed[(subset_cis['group'] == g) & (subset_cis['level_1'] == 'upper')]],
        # Y-values for CI
        color=colors[idx],  # Match the color of the median line
        linewidth=2.5  # Adjust line width for the confidence intervals
    )

axes.get_legend().remove()
sns.despine()
fig_name = 'speedTrial_Fast_3D_boxplot.svg'
plt.savefig(fname=figPath + fig_name, bbox_inches='tight', format='svg', dpi=300)


