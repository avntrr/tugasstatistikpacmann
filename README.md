# PREDIKSI SKUAD FPL DENGAN LINEAR PROGRAMMING
### 1. IMPORT DATASET
````python
import json
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import re
sns.set_style('whitegrid')
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 100)
# source - https://fantasy.premierleague.com/drf/bootstrap-static
with open('/Users/avntrr/Documents/Pacmann/FPL/data.json') as data_file:    
    data = json.load(data_file)
````

### 2. CEK KOLOM
````python
player_data_json = data['elements']
pdata = pd.DataFrame(player_data_json)
pdata.columns
````

### 3. DROP KOLOM YANG TIDAK RELEVAN
````python
to_drop = ['chance_of_playing_this_round','chance_of_playing_next_round','code','cost_change_event','cost_change_event_fall','cost_change_start','cost_change_start_fall','dreamteam_count','ep_this','event_points','form','ict_index','in_dreamteam','news','photo','special','squad_number','status','transfers_in','transfers_in_event','transfers_out','transfers_out_event','value_form','value_season']
pdata.drop(to_drop, axis=1, inplace = True)
pdata['full_name'] = pdata.first_name + " " + pdata.second_name
pdata['element_type_name'] = pdata.element_type.map({x['id']:x['singular_name_short'] for x in data['element_types']})
pdata = pdata.loc[:,['full_name','first_name','second_name', 'element_type','element_type_name','id','team', 'team_code', 'web_name',
                     'saves','penalties_saved','clean_sheets','goals_conceded',
                     'bonus', 'bps','creativity','ep_next','influence', 'threat',
                     'goals_scored','assists','minutes', 'own_goals',
                     'yellow_cards', 'red_cards','penalties_missed',
                     'selected_by_percent', 'now_cost','points_per_game','total_points']]
pdata['team'] = pdata.team.map({x['id']:x['name'] for x in data['teams']})
````

### 4. EDA
````python
#HEATMAP
plt.figure(figsize=(8, 6))
sns.heatmap(pdata.corr(), annot=True, cmap='RdYlBu', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
````
````python
#MEAN
pdata.pivot_table(index='element_type_name', values='total_points', aggfunc=np.mean)
````
````pyton
#BARCHART
f = plt.figure(figsize=(16,9))
ax1 = f.add_subplot(2,2,1)
ax2 = f.add_subplot(2,2,2,sharex=ax1, sharey=ax1)
ax3 = f.add_subplot(2,2,3,sharex=ax1, sharey=ax1)
ax4 = f.add_subplot(2,2,4,sharex=ax1, sharey=ax1)
ax1.set_title('FWD')
sns.distplot(pdata[pdata.element_type_name=='FWD'].total_points, label='FWD',ax=ax1)
ax1.axvline(np.mean(pdata[pdata.element_type_name=='FWD'].total_points),color='red', label='mean')
ax2.set_title('MID')
sns.distplot(pdata[pdata.element_type_name=='MID'].total_points, label='MID',ax=ax2)
ax2.axvline(np.mean(pdata[pdata.element_type_name=='MID'].total_points),color='red', label='mean')
ax3.set_title('DEF')
sns.distplot(pdata[pdata.element_type_name=='DEF'].total_points, label='DEF',ax=ax3)
ax3.axvline(np.mean(pdata[pdata.element_type_name=='DEF'].total_points),color='red', label='mean')
ax4.set_title('GKP')
sns.distplot(pdata[pdata.element_type_name=='GKP'].total_points, label='GKP',ax=ax4)
ax4.axvline(np.mean(pdata[pdata.element_type_name=='GKP'].total_points),color='red', label='mean')

# Add legends
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.show()
````

### 5. LINEAR MODELLING
#### a. Import Pulp
````python
from pulp import *
prob = LpProblem('FantasyTeam', LpMaximize)
````
#### b. Check total number of decision variables
````python
decision_variables = []
for rownum, row in pdata.iterrows():
    variable = str('x' + str(rownum))
    variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1, cat= 'Integer') #make variables binary
    decision_variables.append(variable)

print ("Total number of decision_variables: " + str(len(decision_variables)))
# Returns: Total number of decision_variables: 584
````
#### c. Optimazion function
````python
total_points = ""
for rownum, row in pdata.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            formula = row['total_points']*player
            total_points += formula

prob += total_points
print ("Optimization function: " + str(total_points))
````
#### d. Set avalilable cash
````python
avail_cash = 825
total_paid = ""
for rownum, row in pdata.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            formula = row['now_cost']*player
            total_paid += formula

prob += (total_paid <= avail_cash)
````
#### e. Define players based on position
````python
#GK
avail_gk = 1
total_gk = ""
for rownum, row in pdata.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['element_type_name'] == 'GKP':
                formula = 1*player
                total_gk += formula
prob += (total_gk == avail_gk)
print(total_gk)

#DEF
avail_def = 4
total_def = ""
for rownum, row in pdata.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['element_type_name'] == 'DEF':
                formula = 1*player
                total_def += formula
prob += (total_def == avail_def)
print((total_def))

#MID
avail_mid = 4
total_mid = ""
for rownum, row in pdata.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['element_type_name'] == 'MID':
                formula = 1*player
                total_mid += formula
prob += (total_mid == avail_mid)
print((total_mid))

#FWD
avail_fwd = 2
total_fwd = ""
for rownum, row in pdata.iterrows():
    for i, player in enumerate(decision_variables):
        if rownum == i:
            if row['element_type_name'] == 'FWD':
                formula = 1*player
                total_fwd += formula
prob += (total_fwd == avail_fwd)
print(total_fwd)
````
#### f. SET THE TEAM
````python
team_dict= {}
for team in set(pdata.team_code):
    team_dict[str(team)]=dict()
    team_dict[str(team)]['avail'] = 3
    team_dict[str(team)]['total'] = ""
    for rownum, row in pdata.iterrows():
        for i, player in enumerate(decision_variables):
            if rownum == i:
                if row['team_code'] == team:
                    formula = 1*player
                    team_dict[str(team)]['total'] += formula

    prob += (team_dict[str(team)]['total'] <= team_dict[str(team)]['avail'])
print(len(team_dict))
````
````python
prob.writeLP('/Users/avntrr/Documents/Pacmann/FPL/FantasyTeam.lp')
optimization_result = prob.solve()
assert optimization_result == LpStatusOptimal
print("Status:", LpStatus[prob.status])
print("Optimal Solution to the problem: ", value(prob.objective))
print ("Individual decision_variables: ")
for v in prob.variables():
	print(v.name, "=", v.varValue)
````
````python
variable_name = []
variable_value = []

for v in prob.variables():
    variable_name.append(v.name)
    variable_value.append(v.varValue)

df = pd.DataFrame({'variable': variable_name, 'value': variable_value})
for rownum, row in df.iterrows():
    value = re.findall(r'(\d+)', row['variable'])
    df.loc[rownum, 'variable'] = int(value[0])

df = df.sort_values('variable')

#append results
for rownum, row in pdata.iterrows():
    for results_rownum, results_row in df.iterrows():
        if rownum == results_row['variable']:
            pdata.loc[rownum, 'decision'] = results_row['value']

pdata[pdata.decision==1].now_cost.sum() # Returns 825
````
### 6. SHOW THE TEAM
````python
pdata[pdata.decision==1].sort_values('element_type').head(11)
````
<img width="883" alt="Screenshot 2023-07-15 at 07 45 41" src="https://github.com/avntrr/tugasstatistikpacmann/assets/54851225/a3fda116-c2bd-4800-9a77-6e0c5cfc9b05">




