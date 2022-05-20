import pandas as pd
import re

import time

def masking_frontend_data(file_number):
    ipv4_address = re.compile(r'((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])')
    ipv6_address = re.compile(r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}: | ([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|[fF][eE]80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::([fF]{4}(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))')
    
    for v in file_number:
        file_name = f"/home/ATLAS-T3/eferri/File/FrontendFile/storm-frontend-202003{v}.csv"
        print("reading ", file_name)
    
        data = pd.read_csv(file_name)
        f = data.message

        for i in f.index:
            f[i] = re.sub(ipv6_address, '<IP>', f[i])
            f[i] = re.sub(ipv4_address, '<IP>', f[i])
            f[i] = re.sub('(\\w{8}-\\w{4}-\\w{4}-\\w{4}-\\w{12})', '<TOKEN>', f[i])    
        
        print(f"saving storm-frontend-202003{v}-mask.txt")
        data.to_csv(f"/home/ATLAS-T3/eferri/File/FrontendFile/storm-frontend-202003{v}-mask.csv", index=False)
#         data.to_csv(f"./Tesi-Operational-Intelligence/storm-frontend-202003{v}-mask.txt", index=False)
        
t0 = time.time()

masking_frontend_data(["07"])
masking_frontend_data(["08","09","10","11"])
masking_frontend_data(["12", "13"])

print(f"done in {int((time.time()-t0)/60)} minutes and {((time.time()-t0)%60)} seconds")