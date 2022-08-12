# File to compute optimal TOU dispatch from load data and tariff rate pricing
# Kevin Moy, 5/29/2021
#%%
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
plt.rcParams.update({'font.size': 14})
plt.rc('xtick', direction='out')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)

# %% Function to compute optimal monthly dispatch given load, tariff, pv, and start and end dates
# Input: df with columns load, grid, solar, tariff, and index of datetime for timeseries
# Output: all optimal variables plus TOU costs (for both ESS and no ESS scenarios

def opt_tou(df, month_str, bat_kwh_init, bat_kwh_min, bat_kwh_max, bat_kw, hr_frac):

    # TODO: Force load, pv, tariff, times to be all the same length
    opt_len = len(df.loc[month_str])
    
    load = df.loc[month_str].load.to_numpy()
    tariff = df.loc[month_str].tariff.to_numpy()

    # Introduce pv as zeros if solar does not exist
    if not "solar" in df:
        pv = np.zeros((opt_len,))
    else:
        pv = df.loc[month_str].solar.to_numpy()

    # Introduce peakdemand as zero if peakdemand does not exist
    if not "peakdemand" in df:
        pd_opt = 0.0
    else:
        pd_opt = df.loc[month_str].peakdemand.to_numpy()[0]

    load_opt = load
    tariff_opt = tariff
    pv_opt = np.maximum(pv, 0) ## force this to be positive

    # TOU Optimization configuration
    # Create a new model
    m = gp.Model('tou')
    m.Params.LogToConsole = 0  # suppress console output
    
    # Create variables:
    # each TOU power flow
    # format: to_from
    ess_load = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_load')
    grid_ess = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='grid_ess')
    grid_load = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='grid_load')
    pv_ess = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='pv_ess')
    pv_load = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='pv_load')
    pv_grid = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='pv_grid')
    grid = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='grid')

    # ESS Power dispatch to TOU (positive=discharge, negative=charge)
    ess_c = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_c')
    ess_d = m.addMVar(opt_len, lb=0, vtype=GRB.CONTINUOUS, name='ess_d')

    # ESS binary variables for charge and discharge
    # chg_bin = m.addMVar(opt_len, vtype=GRB.BINARY, name='chg_bin')
    dch_bin = m.addMVar(opt_len, vtype=GRB.BINARY, name='dch_bin')

    # Energy stored in ESS
    ess_E = m.addMVar(opt_len, vtype=GRB.CONTINUOUS, name='E')

    # Peak demand from grid variable
    pk_demand_grid = m.addVar(vtype=GRB.CONTINUOUS, name='pk_demand_grid')

    # Constrain initlal and final stored energy in battery
    m.addConstr(ess_E[0] == bat_kwh_min)
    m.addConstr(ess_E[opt_len-1] == bat_kwh_min)

    for t in range(opt_len):
        # ESS power constraints
        # m.addConstr(ess_c[t] <= bat_kw * chg_bin[t])
        m.addConstr(ess_c[t] <= bat_kw * (1 - dch_bin[t]))
        m.addConstr(ess_d[t] <= bat_kw * dch_bin[t])
        m.addConstr(ess_c[t] >= 0)
        m.addConstr(ess_d[t] >= 0)
        m.addConstr(pv_ess[t] >= 0)
        m.addConstr(pv_load[t] >= 0)
        m.addConstr(pv_grid[t] >= 0)

        # ESS energy constraints
        m.addConstr(ess_E[t] <= bat_kwh_max)
        m.addConstr(ess_E[t] >= bat_kwh_min) 

        # TOU power flow constraints
        m.addConstr(ess_c[t] == pv_ess[t] + grid_ess[t])
        m.addConstr(grid[t] == grid_ess[t] + grid_load[t])
        m.addConstr(pv_opt[t] == pv_ess[t] + pv_grid[t] + pv_load[t])
        m.addConstr(ess_d[t] == ess_load[t])
        m.addConstr(load_opt[t] == ess_load[t] + grid_load[t] + pv_load[t])

        # # #Ensure non-simultaneous charge and discharge across all LMP and TOU
        # m.addConstr(chg_bin[t] + dch_bin[t] <= 1)
        # print(((t+1) * hr_frac) / 24)
        # Charge-sustaining for each day
        if ((t+1) * hr_frac) % 24 == 0:
            # print("one full day")
            m.addConstr(ess_E[t] == ess_E[t+1 - int(24/HR_FRAC)])
            # Prohibit power flow at the end of the horizon (otherwise energy balance is off)
            m.addConstr(ess_d[t] == 0)
            m.addConstr(ess_c[t] == 0)


    # Time evolution of stored energy
    for t in range(1,opt_len):
        m.addConstr(ess_E[t] == hr_frac*(ess_c[t-1]) + ess_E[t-1] - hr_frac*(ess_d[t-1]))

    # Prohibit power flow at the end of the horizon (otherwise energy balance is off)
    m.addConstr(ess_d[opt_len-1] == 0)
    m.addConstr(ess_c[opt_len-1] == 0)

    # Add in peak demand
    m.addConstr(pk_demand_grid == gp.max_(grid[t] for t in range(opt_len)))

    # Objective function
    m.setObjective(hr_frac*tariff_opt @ grid + pk_demand_grid*pd_opt, GRB.MINIMIZE)

    # Solve the optimization
    m.params.MIPGap = 2e-5
    m.optimize()
                    
    tou_run = hr_frac * grid.X * tariff_opt
    grid_run = hr_frac * load * tariff_opt

    print("\n")
    print("Cumulative savings:")
    print(np.sum(grid_run - tou_run))

    return tou_run, grid_run, grid.X, ess_d.X, ess_c.X, ess_E.X, pv_ess.X, pv_grid.X, pv_load.X



# %%Set optimization constants:
# ca_ids = [7062]

ca_ids = [3687, 6377, 7062, 8574, 9213, 203, 1450, 1524, 2606, 3864, 7114,
        1731, 4495, 8342, 3938, 5938, 8061, 9775, 4934, 8733, 9612,
        6547, 9836]

for dataid in ca_ids:
    # load_tariff_name = "9836"
    # load_tariff_name = "LBNL_bldg59"
    load_tariff_name = str(dataid)

    # Import load and tariff rate data; convert to numpy array and get length
    df = pd.read_csv("load_tariff_" + load_tariff_name + ".csv", index_col=0)
    df.index = pd.to_datetime(df.index)

    if load_tariff_name == "LBNL_bldg59":
        # C&I BATTERY (based off Tesla Powerpack, multiplied by 2 for 4 hour system)
        BAT_KW = 250.0  # Rated power of battery, in kW, continuous power for the Powerpack
        BAT_KWH = 475.0 * 2  # Rated energy of battery, in kWh.
    else:
        # RESIDENTIAL BATTERY
        BAT_KW = 5.0 # Rated power of battery, in kW, continuous power for the Powerwall
        BAT_KWH = 14.0  # Rated energy of battery, in kWh.
        # Note Tesla Powerwall rates their energy at 13.5kWh, but at 100% DoD,
        # but I have also seen that it's actually 14kwh, 13.5kWh usable

    BAT_KWH_MIN = 0.2 * BAT_KWH  # Minimum SOE of battery, 10% of rated
    BAT_KWH_MAX = 0.8 * BAT_KWH  # Maximum SOE of battery, 90% of rated
    BAT_KWH_INIT = 0.5 * BAT_KWH  # Starting SOE of battery, 50% of rated
    HR_FRAC = (
        15 / 60
    )  # Data at 15 minute intervals, which is 0.25 hours. Need for conversion between kW <-> kWh

    MONTH_STRS = df.index.strftime("%Y-%m").unique().tolist()
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

    t = time.time()

    for month_str in MONTH_STRS:
        tou_run, grid_run, grid, ess_d, ess_c, ess_E, pv_ess, pv_grid, pv_load = opt_tou(df, month_str, BAT_KWH_INIT, BAT_KWH_MIN, BAT_KWH_MAX, BAT_KW, HR_FRAC)
        tou_runs = np.hstack((tou_runs, tou_run))
        grid_runs = np.hstack((grid_runs, grid_run))
        grids = np.hstack((grids, grid))
        ess_ds = np.hstack((ess_ds, ess_d))
        ess_cs = np.hstack((ess_cs, ess_c))
        ess_Es = np.hstack((ess_Es, ess_E))
        pv_esss = np.hstack((pv_esss, pv_ess))
        pv_grids = np.hstack((pv_grids, pv_grid))
        pv_loads = np.hstack((pv_loads, pv_load))

    elapsed = time.time() - t

    print("Elapsed optimization time: {}".format(elapsed))

    # Stack output DF
    df_output = pd.DataFrame()
    df_output.index = df.index
    if not "solar" in df:
        df_output["pv"] = np.zeros((len(df),))
    else:
        df_output["pv"] = df.solar
    df_output["load"] = df.load
    df_output["tariff"] = df.tariff
    df_output["tou_runs"] = tou_runs
    df_output["grid_runs"] = grid_runs
    df_output["grids"] = grids
    df_output["ess_discharge"] = ess_ds
    df_output["ess_charge"] = ess_cs
    df_output["ess_soe"] = ess_Es
    df_output["pv_esss"] = pv_esss
    df_output["pv_grids"] = pv_grids
    df_output["pv_loads"] = pv_loads

    df_output.to_csv("opt_" + load_tariff_name + "_output_v2.csv")
    print("Saved opt_" + load_tariff_name + "_output_v2.csv")

    # df_output.to_csv("opt_LBNL_bldg59_output_v2.csv")
    # print("Saved opt_LBNL_bldg59_output_v2.csv")
    

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
p1 = ax1.plot(times_plt, df.load * df.tariff *HR_FRAC -tou_runs)
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

# Get random day in year
day = np.random.randint(0,365)
ndays = 2
st = day*4*24
end = day*4*24 + ndays*24*4
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
# ax1.set_xlim(times_plt[day*4*24], times_plt[day*4*24 + 4*24*4])
ax1.set_xlabel("Date")
ax1.set_ylabel("Power, kW")
ax1.set_title("System Power Flows")

p1 = ax1.plot(times_plt[st:end], df.load[st:end], linewidth=4, linestyle=":")
p2 = ax1.plot(times_plt[st:end], grids[st:end])
p3 = ax1.plot(times_plt[st:end], ess_ds[st:end] - ess_cs[st:end])
p4 = ax1.plot(times_plt[st:end], pv_esss[st:end]+pv_grids[st:end]+pv_loads[st:end])

# Shrink current axis by 20%
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax1.legend(["Load Demand", "Grid Supply", "ESS Dispatch", "PV Generation"],loc='center left', bbox_to_anchor=(1, 0.5))

# %% PV power flow disaggregation
# Get random day in year
day = np.random.randint(0,365)
ndays = 2
st = day*4*24
end = day*4*24 + ndays*24*4
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax2 = ax1.twinx()
ax1.set_xlabel("Date")
ax1.set_ylabel("Power, kW")
ax2.set_ylabel("Stored Energy, kWh")
ax2.set_ylim(0,BAT_KWH)
ax1.set_ylim(-BAT_KW,BAT_KW)
ax1.set_title("System Power Flows")

p1 = ax2.plot(times_plt[st:end], ess_Es[st:end], 'g-')
p2 = ax1.plot(times_plt[st:end], -ess_cs[st:end])
p3 = ax1.plot(times_plt[st:end], ess_ds[st:end])

# p4 = ax1.plot(times_plt[st:end], pv_esss[st:end]+pv_grids[st:end]+pv_loads[st:end])


# Shrink current axis by 20%
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
# ax1.legend(["Total PV", "PV to ESS", "PV to load", "PV to grid"],loc='center left', bbox_to_anchor=(1, 0.5))

# %% PV power flow disaggregation
# Get random day in year
# day = np.random.randint(0,365)
# ndays = 2
# st = day*4*24
# end = day*4*24 + ndays*24*4
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
fig.autofmt_xdate()
plt.gcf().autofmt_xdate()
xfmt = mdates.DateFormatter("%m-%d-%y %H:%M")
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_ylim(0, 2.5)
ax1.set_xlabel("Date")
ax1.set_ylabel("Power, kW")
ax1.set_title("System Power Flows")

p1 = ax1.plot(times_plt[st:end], pv_esss[st:end]+pv_grids[st:end]+pv_loads[st:end], linewidth=4, linestyle=":")
p2 = ax1.plot(times_plt[st:end], pv_esss[st:end])
p3 = ax1.plot(times_plt[st:end], pv_loads[st:end])
p4 = ax1.plot(times_plt[st:end], pv_grids[st:end])

# Shrink current axis by 20%
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
# ax1.legend(["Total PV", "PV to ESS", "PV to load", "PV to grid"],loc='center left', bbox_to_anchor=(1, 0.5))
# %%
