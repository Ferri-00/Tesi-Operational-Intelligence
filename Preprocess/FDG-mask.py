import pandas as pd
import time

def group_data(file_number):    
    for v in file_number:
        file_name = f".\File\FrontendFile\storm-frontend-202003{v}-mask.txt"
#         file_name = f".\File\FrontendFile\storm-frontend-202003{v}-mask.txt"

#         frontend = pd.read_csv(file_name, nrows = 1e4)
        frontend = pd.read_csv(file_name)
        print("reading ", file_name)
        
        frontend = frontend.sort_values(by = ['id'])
        frontend = frontend.set_index('id')

        new_frontend = pd.DataFrame(columns = ['PID', 'Level', 'date_time_start', 'date_time_end', 'message'], index = set(frontend.index))

        for i in set(frontend.index):
            PID = frontend['PID'][i]

            if type(frontend.loc[i]) == pd.core.series.Series:
                new_frontend.loc[i]['PID'] = PID
                new_frontend.loc[i]['message'] = frontend.loc[i]['message']
                new_frontend.loc[i]['Level'] = frontend['Level'][i]
                date_time = frontend.loc[i]['date'] + ' ' + frontend.loc[i]['time']
                new_frontend.loc[i]['date_time_start'] = date_time
                new_frontend.loc[i]['date_time_end'] = date_time

            else:
                frontend.loc[i] = frontend.loc[i].sort_values(by = ['time'])

                if len(set(PID)) == 1:
                    new_frontend.loc[i]['PID'] = PID[0]
                else:
                    print(f"Different threads associated to the same ID {i}")
                    break    

                message = list(frontend.loc[i]['message'])
                new_frontend.loc[i]['message'] = message[0]
                for msg in message[1:]:
                    new_frontend.loc[i]['message'] += ' ' + msg
                    
                Level = set(frontend['Level'][i])
                new_frontend.loc[i]['Level'] = str()
                for l in Level:
                    new_frontend.loc[i]['Level'] += l

                date_time = list(frontend.loc[i]['date'] + ' ' + frontend.loc[i]['time'])
                new_frontend.loc[i]['date_time_start'] = date_time[0]
                new_frontend.loc[i]['date_time_end'] = date_time[-1]
                
        print(f"saving storm-frontend-202003{v}-mask-group.txt")
        new_frontend.to_csv(f"./File/FrontendFileGroup/storm-frontend-202003{v}-mask-group.txt")

print("Starting grouping logs")

t0 = time.time()

# group_data(["07","08"])
group_data(["09","10","11","12","13"])

print(f"done in {int((time.time()-t0)/60)} minutes and {((time.time()-t0)%60)} seconds")