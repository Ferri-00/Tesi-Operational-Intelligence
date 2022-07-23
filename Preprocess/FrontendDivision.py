import pandas as pd

from time import time
import sys
import os

def frontend_data(file_number):
    for v in file_number:
        file_name = f"/home/ATLAS-T3/eferri/File/PreFile/storm-frontend-server.log-202003{v}"
        temporary_file = f"./temporaryFile{v}.csv"

        print("reading ", file_name)
        frontend = pd.read_table(file_name , sep = " -  " , header= None , engine = 'python')

        for w, i in zip(frontend[1], frontend.index):
            if w == None:
                frontend = frontend.drop(labels=i, axis=0)
                print(f"Line {i} of file \'{file_name}\' has been droped")

        frontend[0].to_csv(temporary_file , header=False , index=False)

        temporary = pd.read_csv(temporary_file , sep=None , header=None , engine='python' , dtype = str )

        temporary[2] = temporary[2] + ' ' + temporary[3]

        frontend[1].to_csv(temporary_file , header=False , index=False)

        frontend = temporary.drop(3, axis=1)
        frontend.columns = ["date", "time", "PID"]

        temporary = pd.read_table(temporary_file , sep= "]: " , header=None , engine='python' )

        frontend = pd.concat([frontend, temporary[0], temporary[1]], axis=1)
        frontend = frontend.rename(columns={0:"Level", 1:"message"})
        
        frontend.Level.to_csv(temporary_file , header=False , index=False)
        temporary = pd.read_csv(temporary_file , sep= "[" , header=None , engine='python', dtype = str )
        frontend.Level = temporary[0]
        frontend.insert(4, "id", temporary[1])

        print(f"saving storm-frontend-202003{v}.csv")
        frontend.to_csv(f"/home/ATLAS-T3/eferri/File/FrontendFile/storm-frontend-202003{v}.csv" , index=False)

    if os.path.exists(temporary_file):
        os.remove(temporary_file)
    else:
        print("The file does not exist")


if __name__ == "__main__":
    t0 = time()

    file_number = list(sys.argv[1:])
    print('File number:', file_number)
    frontend_data(file_number)

    print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")
