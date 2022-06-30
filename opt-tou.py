# File to compute optimal TOU dispatch from load data and tariff rate pricing
# Kevin Moy, 5/29/2021
#%%
import cvxpy as cp
import time
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Plot formatting defaults
plt.rc('ytick', direction='out')
plt.rc('grid', color='w', linestyle='solid')
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams.update({'font.size': 18})
plt.rc('xtick', direction='out')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

# %%Set environment variables:
# LOAD_LEN = load.size  # length of optimization
BAT_KW = 5.0  # Rated power of battery, in kW, continuous power for the Powerwall
BAT_KWH = 14.0  # Rated energy of battery, in kWh.
# Note Tesla Powerwall rates their energy at 13.5kWh, but at 100% DoD,
# but I have also seen that it's actually 14kwh, 13.5kWh usable
BAT_KWH_MIN = 0.0 * BAT_KWH  # Minimum SOE of battery, 10% of rate
BAT_KWH_MAX = 1.0 * BAT_KWH  # Maximum SOE of battery, 90% of rated
BAT_KWH_INIT = 0.5 * BAT_KWH  # Starting SOE of battery, 50% of rated
HR_FRAC = (
    15 / 60
)  # Data at 15 minute intervals, which is 0.25 hours. Need for conversion between kW <-> kWh

# Import load and tariff rate data; convert to numpy array and get length
df = pd.read_csv("load_tariff.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# %% Function to compute optimal monthly dispatch given load, tariff, pv, and start and end dates
# Input: df with columns gridnopv, grid, solar, tariff, and index of datetime for timeseries
# Output: all optimal variables plus TOU costs (for both ESS and no ESS scenarios

def opt_tou(df, month_str):
    load = df.loc[month_str].gridnopv.to_numpy()
    grid_no_ess = df.loc[month_str].grid.to_numpy()
    pv = df.loc[month_str].solar.to_numpy()
    tariff = df.loc[month_str].tariff.to_numpy()
    times = df.loc[month_str].index

    load_opt = load
    tariff_opt = tariff
    pv_opt = np.maximum(pv, 0) ## force this to be positive
    times_opt = times

    # TODO: Force load, pv, tariff, times to be all the same length
    opt_len = np.shape(load)[0]

    # TOU + LMP Optimization configuration

    # Create a new model
    m = gp.Model('tou')
    m.Params.LogToConsole = 0  # suppress console output
    # Create variables:

    # each TOU power flw
    # format: to_from
    ess_load = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_load')
    grid_ess = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='grid_ess')
    grid_load = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='grid_load')
    pv_ess = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='pv_ess')
    pv_load = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='pv_load')
    pv_grid = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='pv_grid')
    grid = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='grid')
    load_curtail = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='load_curtail')

    # ESS Power dispatch to TOU (positive=discharge, negative=charge)
    ess_c = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_c')
    ess_d = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_d')

    #EXX binary variables for charge and discharge
    chg_bin = m.addMVar(opt_len, vtype=GRB.BINARY, name='chg_bin')
    dch_bin = m.addMVar(opt_len, vtype=GRB.BINARY, name='dch_bin')

    #Energy stored in ESS
    ess_E = m.addMVar(opt_len, vtype=GRB.CONTINUOUS, name='E')

    # Constrain initlal and final stored energy in battery
    # TODO: Modify this to account for MPC energy as an input
    m.addConstr(ess_E[0] == BAT_KWH_INIT)
    m.addConstr(ess_E[opt_len-1] == BAT_KWH_INIT)

    for t in range(opt_len):
        # ESS power constraints
        m.addConstr(ess_c[t] <= BAT_KW * chg_bin[t])
        m.addConstr(ess_d[t] <= BAT_KW * dch_bin[t])
        m.addConstr(ess_c[t] >= 0)
        m.addConstr(ess_d[t] >= 0)
        m.addConstr(pv_ess[t] >= 0)
        m.addConstr(pv_load[t] >= 0)
        m.addConstr(pv_grid[t] >= 0)

        # ESS energy constraints
        m.addConstr(ess_E[t] <= BAT_KWH_MAX)
        m.addConstr(ess_E[t] >= BAT_KWH_MIN) 

        # TOU power flow constraints
        m.addConstr(ess_c[t] == pv_ess[t])
        m.addConstr(grid[t] == grid_ess[t] + grid_load[t])
        m.addConstr(pv_opt[t] == pv_ess[t] + pv_grid[t] + pv_load[t])
        m.addConstr(ess_d[t] == ess_load[t])
        # TODO: Figure out how to remove and add this constraint as load_opt changes in each iteration
        m.addConstr(load_opt[t] == ess_load[t] + grid_load[t] + pv_load[t])

        # #Ensure non-simultaneous charge and discharge across all LMP and TOU
        m.addConstr(chg_bin[t] + dch_bin[t] <= 1)

    # Time evolution of stored energy
    for t in range(1,opt_len):
        m.addConstr(ess_E[t] == HR_FRAC*(ess_c[t-1]) + ess_E[t-1] - HR_FRAC*(ess_d[t-1]))

    # Prohibit power flow at the end of the horizon (otherwise energy balance is off)
    m.addConstr(ess_d[opt_len-1] == 0)
    m.addConstr(ess_c[opt_len-1] == 0)

    # Objective function
    m.setObjective(HR_FRAC*tariff_opt @ grid, GRB.MINIMIZE)

    # Solve the optimization
    # m.params.NonConvex = 2
    m.params.MIPGap = 2e-3
    t = time.time()
    m.optimize()
    elapsed = time.time() - t

    # print("Elapsed time for 1 month of optimization: {}".format(elapsed))
                    
    tou_run = HR_FRAC * grid.X * tariff_opt
    grid_run = HR_FRAC * load * tariff_opt

    print("\n")
    print("Cumulative savings:")
    print(np.sum(grid_run - tou_run))

    return tou_run, grid_run, grid.X, ess_d.X, ess_c.X, ess_E.X, pv_ess.X, pv_grid.X, pv_load.X



# %% 
MONTH_STRS = ["2015-01", "2015-02", "2015-03", "2015-04", "2015-05", "2015-06",
                "2015-07", "2015-08", "2015-09", "2015-10", "2015-11", "2015-12"]

# MONTH_STRS = ["2015-01", "2015-02", "2015-03"]


# Allocate arrays to store output data
tou_runs = []
grid_runs = []
grids = []
ess_ds = []
ess_cs = []
ess_Es = []
pv_esss = []
pv_grids = []
pv_loads = []

for month_str in MONTH_STRS:
    tou_run, grid_run, grid, ess_d, ess_c, ess_E, pv_ess, pv_grid, pv_load = opt_tou(df, month_str)
    tou_runs = np.hstack((tou_runs, tou_run))
    grid_runs = np.hstack((grid_runs, grid_run))
    grids = np.hstack((grids, grid))
    ess_ds = np.hstack((ess_ds, ess_d))
    ess_cs = np.hstack((ess_cs, ess_c))
    ess_Es = np.hstack((ess_Es, ess_E))
    pv_esss = np.hstack((pv_esss, pv_ess))
    pv_grids = np.hstack((pv_grids, pv_grid))
    pv_loads = np.hstack((pv_loads, pv_load))

# %% Stack output DF

df_output = pd.DataFrame()
df_output.index = df.index
df_output["pv"] = df.solar
df_output["load"] = df.gridnopv
df_output["tariff"] = df.tariff
df_output["tou_runs"] = tou_runs
df_output["grid_runs"] = grid_runs
df_output["grids"] = grids
df_output["ess_disp"] = ess_ds - ess_cs
df_output["ess_soe"] = ess_Es
df_output["pv_esss"] = pv_esss
df_output["pv_grids"] = pv_grids
df_output["pv_loads"] = pv_loads

df_output.to_csv("df_9836_output.csv")

# %% Net profit from ESS
# TODO: Fix these plots
times_plt = df.index
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Revenue, $")
ax1.set_title("ESS Net Profit")
p1 = ax1.plot(times_plt, df.gridnopv * df.tariff *HR_FRAC -tou_runs)
plt.grid()

# %% Cumulative profit from ESS
times_plt = df.index
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Savings [$]")
ax1.set_title("Cumulative ESS Savings")
p1 = ax1.plot(times_plt, np.cumsum(np.array(tou_runs)))
plt.grid()

# %% Test plots!
# Net dispatch of ESS
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("Power [kW]")
ax1.set_title("ESS Dispatch")
p1 = ax1.plot(times_plt, ess_ds - ess_cs)
plt.grid()

# %% Test plots!
# State of Energy (SOE) of ESS
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlabel("Date")
ax1.set_ylabel("SOE [-]")
ax1.set_title("ESS State of Energy")
p1 = ax1.plot(times_plt, ess_Es/BAT_KWH)
plt.grid()

# %% Load power flow disaggregation
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlim(times_plt[0], times_plt[0 + 4*24*7])
ax1.set_xlabel("Date")
ax1.set_ylabel("Power, kW")
ax1.set_title("System Power Flows")
p1 = ax1.plot(times_plt, df.gridnopv, linewidth=4, linestyle=":")
p2 = ax1.plot(times_plt, grids)
p3 = ax1.plot(times_plt, ess_ds - ess_cs)
p4 = ax1.plot(times_plt, pv_esss+pv_grids+pv_loads)
plt.legend(["Load Demand", "Grid Supply", "ESS Dispatch", "PV Generation"])
plt.grid()

# %%