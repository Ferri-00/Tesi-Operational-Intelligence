import pandas as pd
import re

import time
from time import time

import sys

def masking_frontend_data(file_number):
    #IP masking
    mask=[]
    mask.append(r'([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,2}|')
    mask.append(r'([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|')
    mask.append(r'([0-9a-fA-F]{1,4}:){1,7}:|')
    mask.append(r'([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|')
    mask.append(r'([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|')
    mask.append(r'([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|')
    mask.append(r'([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|')
    mask.append(r'([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}')
    mask.append(r'[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})')
    mask.append(r':((:[0-9a-fA-F]{1,4}){1,7}|:)|')
    mask.append(r'fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|')
    mask.append(r'::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|')
    mask.append(r'([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|')
    mask.append(r'((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])')
    mask.append(r'[fF][eE]80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}')

    ipmask="".join(mask)

    ipv6_address=re.compile(ipmask)     
    ipv4_address = re.compile(r'((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])')


    specific_substitute = '<IP>'

    for v in file_number:
        file_name = f"/home/ATLAS-T3/eferri/File/FrontendFileGroup/storm-frontend-202003{v}-group.csv"
        print("reading ", file_name)
    
        data = pd.read_csv(file_name)
        f = data.message
        
        for i in f.index:
            f[i] = re.sub(ipv6_address, specific_substitute, f[i])
            f[i] = re.sub(ipv4_address, specific_substitute, f[i])
            f[i] = re.sub('(\\w{8}-\\w{4}-\\w{4}-\\w{4}-\\w{12})', '<TOKEN>', f[i])
            f[i] = re.sub('srm:\/\/storm-fe.cr.cnaf.infn.it\/[a-zA-Z0-9_/.-]+', '<URL>', f[i])

        print(f"saving storm-frontend-202003{v}-mask-group.csv")
        data.to_csv(f"/home/ATLAS-T3/eferri/File/FrontendFileGroup/storm-frontend-202003{v}-mask-group.csv", index=False)
        

if __name__ == "__main__":
    t0 = time()

    file_number = list(sys.argv[1:])
    print('File number:', file_number)
    masking_frontend_data(file_number)

    print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")
