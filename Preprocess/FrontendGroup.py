import pandas as pd
import time

def group_data(file_number):    
    for v in file_number:
        file_name = f".\File\FrontendFile\storm-frontend-202003{v}.txt"
#         file_name = f".\File\FrontendFileMask\storm-frontend-202003{v}-mask.txt"

#         frontend = pd.read_csv(file_name, nrows = 1e4)
        frontend = pd.read_csv(file_name)
        print("reading ", file_name)
        
        frontend = frontend.sort_values(by = ['id'])
        frontend = frontend.set_index('id')

        new_frontend = pd.DataFrame(columns = ['PID', 'Level', 'date_time_start', 'date_time_end', 'msg1'], index = set(frontend.index))

        for i in set(frontend.index):
            PID = frontend['PID'][i]
            msg = list(frontend.loc[i]['message'])


            if type(frontend.loc[i]) == pd.core.series.Series:
                new_frontend.loc[i]['PID'] = PID
                new_frontend.loc[i]['msg1'] = msg

            else:
                frontend.loc[i] = frontend.loc[i].sort_values(by = ['time'])

                if len(set(PID)) == 1:
                    new_frontend.loc[i]['PID'] = PID[0]
                else:
                    print(f"Different threads associated to the same ID {i}")
                    break    

                while new_frontend.shape[1] - 4 < len(msg):
                    new_frontend[f'msg{new_frontend.shape[1] - 3}'] = None
                    print(i, new_frontend.columns)

                for l in range(len(msg)):
                    new_frontend.loc[i][f'msg{l+1}'] = msg[l]

            Level = set(frontend['Level'][i])
            new_frontend.loc[i]['Level'] = str()
            for l in Level:
                new_frontend.loc[i]['Level'] += l

            date_time = list(frontend.loc[i]['date'] + ' ' + frontend.loc[i]['time'])
            new_frontend.loc[i]['date_time_start'] = date_time[0]
            new_frontend.loc[i]['date_time_end'] = date_time[-1]
                
        print(f"saving storm-frontend-202003{v}-group.txt")
        new_frontend.to_csv(f"./File/FrontendFileGroup/storm-frontend-202003{v}-group.txt")

print("Starting grouping logs")

t0 = time.time()

group_data(["07"])
# group_data(["08","09","10","11","12", "13"])

print(f"done in {int((time.time()-t0)/60)} minutes and {((time.time()-t0)%60)} seconds")