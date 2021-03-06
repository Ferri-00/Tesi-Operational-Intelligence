import pandas as pd
import time
import os

def frontend_data(file_number):
    temporary_file = './temporaryFile.csv'
    
    for v in file_number:
        file_name = f".\logs_7-13march\storm-atlas-2\storm-frontend-server.log-202003{v}.gz"

        print("reading ", file_name)
        frontend = pd.read_table(file_name , sep = " -  " , header= None , engine = 'python')

        for w, i in zip(frontend[1], frontend.index):
            if w == None:
                frontend = frontend.drop(labels=i, axis=0)
                print(f"Line {i} of file \'{file_name}\' has been droped")

        frontend[0].to_csv(temporary_file , header=False , index=False)

        temporary = pd.read_table(temporary_file , sep=None , header=None , engine='python' , dtype = str )

        temporary[2] = temporary[2] + ' ' + temporary[3]

        frontend[1].to_csv(temporary_file , header=False , index=False)

        frontend = temporary.drop(3, axis=1)
        frontend.columns = ["date", "time", "Thread"]

        temporary = pd.read_table(temporary_file , sep= "]: " , header=None , engine='python' , dtype = str )

        frontend = pd.concat([frontend, temporary[0] + "]", temporary[1]], axis=1)
        frontend = frontend.rename(columns={0:"INFO", 1:"message"})

        print(f"saving storm-frontend-202003{v}.txt")
        frontend.to_csv(f".\FrontendFile2\storm-frontend-202003{v}.txt" , index=False)

    if os.path.exists(temporary_file):
        os.remove(temporary_file)
    else:
        print("The file does not exist")


print("Starting division of files")
t0= time.time()
frontend_data(["07","08","09","10","11","12", "13"])
print(f"done in {(time.time()-t0)}")