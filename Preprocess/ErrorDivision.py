import pandas as pd
import numpy as np
import time
from time import time

import sys

import re

def division_errors(file_number):
    for v in file_number:
        file_name = f"/home/ATLAS-T3/eferri/File/FrontendFileGroup/storm-frontend-202003{v}-mask-group.csv"
        
        print('reading', file_name)
        logs = pd.read_csv(file_name, index_col=0)
        errors = pd.DataFrame(columns=logs.columns)
        errors.to_csv(f"/home/ATLAS-T3/eferri/File/FrontendFileErr/storm-frontend-202003{v}-err.csv")
        
        for idx, msg in zip(logs.index, logs.message):
            resultE = re.search('error', msg.lower())
            resultF = re.search('failure', msg.lower())
            # resultP = re.search('problem', msg.lower())
            if resultE!=None or resultF!=None:
                errors.loc[idx] = logs.loc[idx]
                logs = logs.drop(idx, axis=0)
                errors.to_csv(f"/home/ATLAS-T3/eferri/File/FrontendFileErr/storm-frontend-202003{v}-err.csv", sep=',',
                              mode='a', header=False)
                errors.drop(idx, axis=0, inplace=True)

        logs.to_csv(f"/home/ATLAS-T3/eferri/File/FrontendFileErr/storm-frontend-202003{v}-msg.csv")
        
print("Starting division of files")

if __name__ == "__main__":
    t0 = time()

    file_number = list(sys.argv[1:])
    print('File number:', file_number)
    division_errors(file_number)

    print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")
