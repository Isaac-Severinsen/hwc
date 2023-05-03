from get_data import get_new_access_token, WITS_API_call
# import holidays
# NZ_holidays = holidays.country_holidays('NZ', subdiv='AUK')
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def get_prices():
    
    nodes_list = ['HEP0331']  #,'ROS1101']

    access_token = get_new_access_token()
    RTD = WITS_API_call(access_token, back = '48', nodes = nodes_list, schedules='RTD')      # 'RTD' - Real-time data - energy
    NRSS = WITS_API_call(access_token, forward = '48', nodes = nodes_list, schedules='NRSS')    # 'NRSS' - Non-responseive schedule short - energy
    NRSL = WITS_API_call(access_token, forward = '48', nodes = nodes_list, schedules='NRSL')    # 'NRSL' - Non-responseive schedule long - energy
    PRSS = WITS_API_call(access_token, forward = '48', nodes = nodes_list, schedules='PRSS')    # 'PRSS' - Price-responseive schedule short - energy
    PRSL = WITS_API_call(access_token, forward = '48', nodes = nodes_list, schedules='PRSL')    # 'PRSL' - Price-responseive schedule long - energy

    NRSS = NRSS.sort_values('tradingDateTime').reset_index()
    NRSL = NRSL.sort_values('tradingDateTime').reset_index()
    PRSS = PRSS.sort_values('tradingDateTime').reset_index()
    PRSL = PRSL.sort_values('tradingDateTime').reset_index()

    return RTD, NRSS, NRSL, PRSS, PRSL

def get_network_pricing(dt):
    # https://www.vector.co.nz/personal/electricity/about-our-network/pricing
    # https://blob-static.vector.co.nz/blob/vector/media/vector-2023/final-1-april-2023-pricing-schedule-and-policy_1.pdf

    is_weekday = dt.weekday() < 5  # Monday = 0, Sunday = 6

    if dt.month >= 10 or dt.month <= 3:
        is_summer = True
    else:
        is_summer = False
    
    # is_holiday = dt.date() in NZ_holidays.keys()
    is_morning = dt.time() >= pd.Timestamp('07:00:00').time() and dt.time() <= pd.Timestamp('11:00:00').time()
    is_evening = dt.time() >= pd.Timestamp('17:00:00').time() and dt.time() <= pd.Timestamp('21:00:00').time()

    if is_weekday and not is_summer and (is_morning or is_evening):
        price = price + 93.5
    else:
        pass

    return price

def find_presumed_energy(TP, TP_since_known, known_state, boost_arr, top_arr): # 2 1 unknown [False, False, False]
    # Inputs: 
    # - TP current TP (0-48)
    # - TP_since_known trading periods since the stored energy was known
    # - known_flag : ['hot','cold']

    # cold flag only happens if the power through the top element stops without bottom element activation
    # hot flag only occurs if the power through the bottom element stops when it is available.

    T_ref = 15 # [°C]
    Cp = 4.184 # [kJ/kgK]
    HWC_Volume = 250 # [kg]
    HWC_Top_Volume = 125 # [kg]
    Boost_SP = 80 # [°C]
    SP = 60 # [°C]
    db = 2 # [°C]

    hot_up = HWC_Volume * Cp * (Boost_SP + db -T_ref) / 3600 # kWh
    hot_down = HWC_Volume * Cp * (Boost_SP - db -T_ref) / 3600 # kWh

    cold_up = HWC_Top_Volume * Cp * (SP + db -T_ref) / 3600 # kWh
    cold_down = HWC_Top_Volume * Cp * (SP - db -T_ref) / 3600 # kWh

    dead_cold = HWC_Volume * Cp * (T_ref-T_ref) / 3600 # kWh

    TP_known = TP - TP_since_known # [what TP did we last know the actual energy stored]

    if known_state == 'hot': # whole cylinder is piping hot
        known_energy = hot_up
    elif known_state == 'cold': # top half is warm only
        known_energy = cold_up
    else: # dead cold
        known_energy = dead_cold

    # Loop through array of TP between known and now

    TP_arr = np.array(range(0, TP_since_known))
    TP_arr_plot = TP_arr.copy()
    TP_arr = TP_arr + TP_known
    TP_arr = TP_arr % 48

    known_energy_arr = []

    for i,TP_i in enumerate(TP_arr):
        known_energy -= 0.065 # losses kWh/TP

        morning_TP = random.randint(14, 18)
        evening_TP = random.randint(36, 40)
        morning_e = random.uniform(3, 5)
        evening_e = random.uniform(3, 5)

        if TP_i < 18 and TP_i > 14: # morning
            known_energy -= morning_e / (18-14) # kWh/TP
        elif TP_i < 40 and TP_i > 36: # evening
            known_energy -= evening_e / (40-36) # kWh/TP

        if boost_arr[i]: # boost heating
            if known_energy + 1.5 > hot_up: # It is unable to go above boost_SP (1)
                known_energy = hot_up
            else:
                known_energy += 1.5 # kWh/TP
        if top_arr[i]: # top heating
            if known_energy + 1.5 < cold_up: # It is unable to go above SP (0)
                known_energy += 1.5
            else:
                known_energy = cold_up

        if known_energy < dead_cold:
            known_energy = dead_cold
        
        known_energy_arr.append(known_energy)

    presumed_energy = known_energy # why does this not change properly?
    stored_energy = (known_energy - cold_up) / (hot_up-cold_up)

    return presumed_energy, stored_energy, known_state # kWh

def predict_future_energy(TP, stored_energy, verbose = False):
    T_ref = 15 # [°C]
    Cp = 4.184 # [kJ/kgK]
    HWC_Volume = 250 # [kg]
    HWC_Top_Volume = 125 # [kg]
    Boost_SP = 80 # [°C]
    SP = 60 # [°C]
    db = 2 # [°C]

    hot_up = HWC_Volume * Cp * (Boost_SP + db -T_ref) / 3600 # kWh
    hot_down = HWC_Volume * Cp * (Boost_SP - db -T_ref) / 3600 # kWh

    cold_up = HWC_Top_Volume * Cp * (SP + db -T_ref) / 3600 # kWh
    cold_down = HWC_Top_Volume * Cp * (SP - db -T_ref) / 3600 # kWh

    dead_cold = HWC_Volume * Cp * (T_ref-T_ref) / 3600 # kWh

    known_energy = stored_energy * (hot_up-cold_up) + cold_up

    future_energy_arr = []
    future_power_arr = []

    # Future
    TP_future_arr = np.array(range(0, 48))
    TP_future_arr = TP_future_arr + TP
    TP_future_arr = TP_future_arr % 48
    for i,TP_fi in enumerate(TP_future_arr):
        known_energy -= 0.065 # losses kWh/TP

        # # random

        # morning_TP = random.randint(14, 18)
        # evening_TP = random.randint(36, 40)
        # morning_e = random.uniform(3, 5)
        # evening_e = random.uniform(3, 5)

        # if TP_fi == morning_TP: # morning
        #     known_energy -= morning_e # kWh/TP
        # elif TP_fi == evening_TP: # evening
        #     known_energy -= evening_e # kWh/TP

        # linear

        morning_TP = random.randint(14, 18)
        evening_TP = random.randint(36, 40)
        morning_e = random.uniform(3, 5)
        evening_e = random.uniform(3, 5)

        if TP_fi < 18 and TP_fi > 14: # morning
            known_energy -= morning_e / (18-14) # kWh/TP
        elif TP_fi < 40 and TP_fi > 36: # evening
            known_energy -= evening_e / (40-36) # kWh/TP

        if known_energy < cold_down: # top heating
            if known_energy + 1.5 < cold_up: # It is unable to go above SP (0)
                future_power_arr.append(1.5)
                known_energy += 1.5
            else:
                future_power_arr.append(cold_up-known_energy)
                known_energy = cold_up
        else:
            future_power_arr.append(0)

        if known_energy < dead_cold:
            known_energy = dead_cold

        future_energy_arr.append(known_energy)

    future_energy_arr = np.array(future_energy_arr)
    future_energy_arr = (future_energy_arr - cold_up) / (hot_up-cold_up)

    return future_energy_arr, future_power_arr

def synth_cylinder(energy, TP, boost_allow):

    T_ref = 15 # [°C]
    Cp = 4.184 # [kJ/kgK]
    HWC_Volume = 250 # [kg]
    HWC_Top_Volume = 125 # [kg]
    Boost_SP = 80 # [°C]
    SP = 60 # [°C]
    db = 2 # [°C]

    top_power_rating = 1.5 #[kWh/TP]
    boost_power_rating = 1.5 #[kWh/TP]

    hot_up = HWC_Volume * Cp * (Boost_SP + db -T_ref) / 3600 # kWh
    hot_down = HWC_Volume * Cp * (Boost_SP - db -T_ref) / 3600 # kWh

    cold_up = HWC_Top_Volume * Cp * (SP + db -T_ref) / 3600 # kWh
    cold_down = HWC_Top_Volume * Cp * (SP - db -T_ref) / 3600 # kWh

    dead_cold = HWC_Volume * Cp * (T_ref-T_ref) / 3600 # kWh

    # Consumption

    energy -= 0.065 # losses kWh/TP
    
    morning_TP = random.randint(14, 18)
    evening_TP = random.randint(36, 40)
    morning_e = random.uniform(3, 5)
    evening_e = random.uniform(3, 5)

    # if TP == morning_TP: # morning
    #     energy -= morning_e # kWh/TP
    # elif TP == evening_TP: # evening
    #     energy -= evening_e # kWh/TP

    if TP < 18 and TP > 14: # morning
        energy -= morning_e / (18-14) # kWh/TP
    elif TP < 40 and TP > 36: # evening
        energy -= evening_e / (40-36) # kWh/TP

    # Generation

    if energy < cold_down: # top heating
        if energy + top_power_rating < cold_up: # It is unable to go above SP (60)
            top_power = top_power_rating
            energy += top_power_rating
        else:
            top_power = cold_up - energy
            energy = cold_up # Maximum possible 
    else:
        top_power = 0

    if boost_allow:
        if energy < hot_down:
            if energy + boost_power_rating < hot_up: # It is unable to go above SP (80)
                boost_power = boost_power_rating
                energy += boost_power_rating
            else:
                boost_power = hot_up - energy
                energy = hot_up # Maximum possible 
        else:
            boost_power = 0
    else:
        boost_power = 0

    if energy < dead_cold: # cannot go below dead_cold
        energy = dead_cold

    return energy, top_power, boost_power 

def analyse_state(boost_power, top_power, boost_allow, TP, TP_since_known, boost_arr,top_arr, known_state):

    # Two ways to known the state of the cylinder for certain.
    if boost_allow and not boost_power:
        state = 'hot'
        stored_energy = 1
    elif len(top_arr) < 2:
        state = 'unknown'
    elif top_arr[-2] and not top_power and not (boost_arr[-2] and boost_power):
        # top element was on previously, but is not now and boost was not on at any point in last 2
        state = 'cold'
        stored_energy = 0
    else:
        state = 'unknown'

    if not state == 'unknown':
        TP_since_known = 0
        boost_arr = [boost_power] # reset boost_arr
        top_arr = [top_power] # reset boost_arr
        known_state = state
        presumed_energy = np.nan
    else:
        TP_since_known += 1
        boost_arr.append(boost_power)
        top_arr.append(top_power)
        presumed_energy, stored_energy, known_state = find_presumed_energy(TP, TP_since_known, known_state, boost_arr, top_arr)

    return presumed_energy, stored_energy, TP_since_known, boost_arr,top_arr, known_state

def plott(TP_arr, stored_energy_arr, stored_energy_uncertainty):

    x = np.array(stored_energy_arr)
    x1 = np.array(stored_energy_uncertainty)
    plt.figure()
    plt.plot([0,TP_arr[-1]],[1,1],'r-', linewidth=0.8)
    plt.plot([0,TP_arr[-1]],[0,0],'g-', linewidth=0.8)
    plt.plot([0,TP_arr[-1]],[-0.53,-0.53],'k-', linewidth=0.8)
    plt.plot(stored_energy_arr, linewidth=3)
    plt.fill_between(range(len(x)), x-x1, x+x1 ,alpha=0.3)

    plt.legend(['80°C','65°C half','15°C', 'current energy'], loc='upper right')

    return

def read_old_data():
    d2020 = pd.read_csv('Old_data/2020_ROS1101.csv', names = ['node','date','TP','price','date2','data'])
    d2021 = pd.read_csv('Old_data/2021_ROS1101.csv', names = ['node','date','TP','price','date2','data'])
    d2022 = pd.read_csv('Old_data/2022_ROS1101.csv', names = ['node','date','TP','price','date2','data'])
    d2023 = pd.read_csv('Old_data/2023_ROS1101.csv', names = ['node','date','TP','price','date2','data'])
    df = pd.concat([d2020, d2021,d2022,d2023])

    td_arr = []
    for i in range(len(df)):
        td_arr.append(pd.Timedelta(df.iloc[i]['TP']*0.5-0.5,'hours'))

    df['dt'] = pd.to_datetime(df['date'], dayfirst=True) + pd.to_timedelta(td_arr)
    df = df.drop(['data','date2','date','node'],axis=1)

    df = df.reset_index()
    df = df.rename({'dt':'date'},axis=1)
    df = df.drop(['index'],axis=1)
    df.set_index('date', inplace=True)

    df_raw = df.copy()

    df.index = pd.to_datetime(df.index)
    mask1 = df.index.weekday > 4
    mask2 = (df.index.month >= 10) | (df.index.month <= 3)
    # mask2 = df.index.to_series().dt.date.astype('datetime64').isin(NZ_holidays.keys())
    mask3 = np.zeros(len(df.index), dtype=bool)
    mask3[df.index.indexer_between_time('07:00:00', '11:00:00')] = True
    mask3[df.index.indexer_between_time('17:00:00', '21:00:00')] = True
    mask = ~(mask1 | mask2) & (mask3)
    df.loc[mask,'price'] = df.loc[mask,'price'] + 93.5

    return df

def run_regress(regress_params, df):

    T_ref = 15 # [°C]
    Cp = 4.184 # [kJ/kgK]
    HWC_Volume = 250 # [kg]
    HWC_Top_Volume = 125 # [kg]
    Boost_SP = 80 # [°C]
    SP = 60 # [°C]
    db = 2 # [°C]

    top_power_rating = 1.5 #[kWh/TP]
    boost_power_rating = 1.5 #[kWh/TP]

    hot_up = HWC_Volume * Cp * (Boost_SP + db -T_ref) / 3600 # kWh
    hot_down = HWC_Volume * Cp * (Boost_SP - db -T_ref) / 3600 # kWh

    cold_up = HWC_Top_Volume * Cp * (SP + db -T_ref) / 3600 # kWh
    cold_down = HWC_Top_Volume * Cp * (SP - db -T_ref) / 3600 # kWh

    dead_cold = HWC_Volume * Cp * (T_ref-T_ref) / 3600 # kWh

    energy = hot_up # [kWh]

    save_price = []

    TP_since_known = 0
    boost_arr = []
    top_arr = []
    known_state = 'hot'
    boost_power = False
    boost_allow = False
    top_power = False

    # df = read_old_data()
    N = len(df) - 48
    # N = 10000
    df_data = df.iloc[:N]

    for i in range(N):

        # from hisotrical data
        TP = df.iloc[i]['TP']
        current_price = df.iloc[i]['price']
        RTD = df.iloc[i-48:i]['price'].to_numpy()

        PRSS = df.iloc[i:i+8]['price'].to_numpy()
        NRSS = df.iloc[i:i+8]['price'].to_numpy()
        PRSL = df.iloc[i:i+48]['price'].to_numpy()
        NRSL = df.iloc[i:i+48]['price'].to_numpy()

        # current data
        # RTD, NRSS, NRSL, PRSS, PRSL = get_prices()
        # current_price = RTD.iloc[-1]['price']
        # network_TOU = get_network_pricing(pd.Timestamp(RTD.iloc[-1].index))
        # current_price = current_price + network_TOU

        # RTD_TOU = np.array([get_network_pricing(pd.Timestamp(RTD.iloc[i].index)) for i in RTD.index])
        # long_TOU = np.array([get_network_pricing(pd.Timestamp(PRSL.iloc[i].index)) for i in PRSL.index])
        # short_TOU = np.array([get_network_pricing(pd.Timestamp(PRSS.iloc[i].index)) for i in PRSS.index])

        # PRSS = PRSS + short_TOU
        # PRSL = PRSL + long_TOU
        # NRSS = NRSS + short_TOU
        # NRSL = NRSL + long_TOU

        # (predicted_energy, boost_allow, 
        # TP_since_known, boost_arr,top_arr, 
        # known_state, threshold_price, flag) =        HWC_controller(TP, RTD, NRSS, NRSL, PRSS, PRSL,
        #                                                             current_price, boost_power, top_power, 
        #                                                             boost_allow, TP_since_known, boost_arr, 
        #                                                             top_arr, known_state,regress_params ,
        #                                                             verbose = False)

        (predicted_energy, boost_allow, 
        TP_since_known, boost_arr,top_arr, 
        known_state, threshold_price, flag) = simple_HWC_controller(TP, RTD, NRSS, NRSL, PRSS, PRSL,
                                                                    current_price, boost_power, top_power, 
                                                                    boost_allow, TP_since_known, boost_arr, 
                                                                    top_arr, known_state,regress_params ,
                                                                    verbose = False)

        energy, top_power, boost_power = synth_cylinder(energy, TP, boost_allow)

        boost_arr.append(boost_power)
        top_arr.append(top_power)
        
        if (boost_power or top_power):
            save_price.append(current_price)
        else:
            save_price.append(np.nan)

    return np.nanmean(save_price)

def run_plot(regress_params):

    energy = 15 # [kWh]

    save_predicted_energy = []
    save_energy = []
    save_boost_power = []
    save_top_power = []
    save_boost_cost = []
    save_top_cost = []
    save_threshold_price = []
    save_TP_since_known = []
    save_price = []
    save_flag = []

    TP_since_known = 0
    boost_arr = []
    top_arr = []
    known_state = 'hot'
    boost_power = False
    boost_allow = False
    top_power = False

    df = read_old_data()
    N = len(df) - 48
    # N = 10000
    df_data = df.iloc[:N]

    for i in range(N):

        # from hisotrical data
        TP = df.iloc[i]['TP']
        current_price = df.iloc[i]['price']
        RTD = df.iloc[i-48:i]['price'].to_numpy()

        PRSS = df.iloc[i:i+8]['price'].to_numpy()
        NRSS = df.iloc[i:i+8]['price'].to_numpy()
        PRSL = df.iloc[i:i+48]['price'].to_numpy()
        NRSL = df.iloc[i:i+48]['price'].to_numpy()

        # current data
        # RTD, NRSS, NRSL, PRSS, PRSL = get_prices()
        # current_price = RTD.iloc[-1]['price']
        # network_TOU = get_network_pricing(pd.Timestamp(RTD.iloc[-1].index))
        # current_price = current_price + network_TOU

        # RTD_TOU = np.array([get_network_pricing(pd.Timestamp(RTD.iloc[i].index)) for i in RTD.index])
        # long_TOU = np.array([get_network_pricing(pd.Timestamp(PRSL.iloc[i].index)) for i in PRSL.index])
        # short_TOU = np.array([get_network_pricing(pd.Timestamp(PRSS.iloc[i].index)) for i in PRSS.index])

        # PRSS = PRSS + short_TOU
        # PRSL = PRSL + long_TOU
        # NRSS = NRSS + short_TOU
        # NRSL = NRSL + long_TOU

        # predicted_energy, boost_allow, TP_since_known, boost_arr,top_arr, known_state, threshold_price, flag = HWC_controller(TP, RTD, NRSS, NRSL, PRSS, PRSL, current_price, boost_power, top_power, boost_allow, TP_since_known, boost_arr, top_arr, known_state ,verbose = False)

        # (predicted_energy, boost_allow, 
        # TP_since_known, boost_arr,top_arr, 
        # known_state, threshold_price, flag) =        HWC_controller(TP, RTD, NRSS, NRSL, PRSS, PRSL,
        #                                                             current_price, boost_power, top_power, 
        #                                                             boost_allow, TP_since_known, boost_arr, 
        #                                                             top_arr, known_state,regress_params ,
        #                                                             verbose = False)

        (predicted_energy, boost_allow, 
        TP_since_known, boost_arr,top_arr, 
        known_state, threshold_price, flag) = simple_HWC_controller(TP, RTD, NRSS, NRSL, PRSS, PRSL,
                                                                    current_price, boost_power, top_power, 
                                                                    boost_allow, TP_since_known, boost_arr, 
                                                                    top_arr, known_state,regress_params ,
                                                                    verbose = False)

        energy, top_power, boost_power = synth_cylinder(energy, TP, boost_allow)

        boost_arr.append(boost_power)
        top_arr.append(top_power)
        
        save_predicted_energy.append(predicted_energy)
        save_energy.append(energy)
        save_boost_power.append(boost_power)
        save_top_power.append(top_power)
        if (boost_power or top_power):
            save_price.append(current_price)
        else:
            save_price.append(np.nan)
        save_boost_cost.append(boost_power * current_price/1000)
        save_top_cost.append(top_power * current_price/1000)
        save_threshold_price.append(threshold_price)
        save_TP_since_known.append(TP_since_known)
        save_flag.append(flag)

    df_data['predicted_energy'] = save_predicted_energy
    df_data['stored_energy'] = save_energy
    df_data['boost_power'] = save_boost_power
    df_data['top_power'] = save_top_power
    df_data['boost_cost'] = save_boost_cost
    df_data['top_cost'] = save_top_cost
    df_data['threshold_price'] = save_threshold_price
    df_data['TP_since_known'] = save_TP_since_known
    df_data['price_paid'] = save_price
    df_data['flag'] = save_flag

    return df, df_data

def HWC_controller(TP, RTD, NRSS, NRSL, PRSS, PRSL, current_price, boost_power, top_power, boost_allow, TP_since_known, boost_arr, top_arr, known_state ,regress_params ,verbose = False):

    HWC_boost_power = 3 #[kW]
    HWC_boost_power_per_TP = HWC_boost_power*0.5 #[kW]

    presumed_energy, stored_energy, TP_since_known, boost_arr,top_arr, known_state = analyse_state(boost_power, top_power, boost_allow, TP, TP_since_known, boost_arr, top_arr, known_state)
    # stored_energy_uncertainty = TP_since_known/200
    
    future_energy_arr, future_power_arr = predict_future_energy(TP, stored_energy, True)

    index_future = int(np.argmax((np.array(future_power_arr) != 0).astype(bool)))

    try:
        prices_till_hot = NRSL[:index_future]
        Nth_lowest = np.sort(prices_till_hot)[8]
    except:
        Nth_lowest = current_price-10

    # # find the first required energy and look up till that point only.
    # predicted_energy_24hrs = sum(future_power_arr)
    # total_boost_periods_required = predicted_energy_24hrs/HWC_boost_power_per_TP # kWh/TP
    # N_periods = int(np.ceil(total_boost_periods_required))

    # if verbose:
    #     print('predicted_energy_24hrs', round(predicted_energy_24hrs,3), 'kWh')
    #     print('total_boost_periods_required',round(total_boost_periods_required,3))

    # N_periods = np.max([1,N_periods])
    # long_price = NRSL[:48] # check if this is actually 48 long

    # # controlled
    # lowest_TPs = np.argpartition(long_price, N_periods)[:N_periods]
    # lowest_TPs_price_sorted = np.sort(long_price[lowest_TPs])
    # max_price_controlled = lowest_TPs_price_sorted[-1]

    uncertainty_mult = np.max([1,TP_since_known / 500])

    price_threshold = (Nth_lowest*1.01 + 5) * uncertainty_mult
    # price_threshold = (max_price_controlled*1.01 + 5) * uncertainty_mult
    flag = 3
    
    if current_price < price_threshold:
        boost_allow = True
    else:
        boost_allow = False

    # if current_price < 30:
    #     boost_allow = True
    # else:
    #     pass

    return presumed_energy, boost_allow, TP_since_known, boost_arr,top_arr, known_state, price_threshold, flag

def simple_HWC_controller(TP, RTD, NRSS, NRSL, PRSS, PRSL, current_price, boost_power, top_power, boost_allow, TP_since_known, boost_arr, top_arr, known_state,regress_params ,verbose = False):

    # A = int(regress_params['A'].value)
    # B = int(regress_params['B'].value)
    # # C = int(regress_params['C'].value)
    # # D = int(regress_params['D'].value)
    # # E = int(regress_params['E'].value)
    # # F = int(regress_params['F'].value)
    # # G = int(regress_params['G'].value)

    try:
        price_threshold_fwd = np.sort(np.unique(NRSL[:40]))[3]
        # price_threshold_back = np.sort(np.unique(RTD[-C:]))[D] 
        # price_threshold = (E*price_threshold_back + (1-E)*price_threshold_fwd) + 5
        price_threshold = (price_threshold_fwd) + 5

    except:
        price_threshold = current_price * 1.1
        boost_allow = False      

    flag = 3
    presumed_energy = 15
    
    if current_price < price_threshold:
        boost_allow = True
    else:
        boost_allow = False

    if current_price < 30:
        boost_allow = True
    else:
        pass

    return presumed_energy, boost_allow, TP_since_known, boost_arr,top_arr, known_state, price_threshold, flag


