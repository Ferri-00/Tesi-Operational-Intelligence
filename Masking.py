import pandas as pd
import re

import time

def masking_frontend_data(file_number):
    ipv4_address = re.compile(r'((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])')
    ipv6_address = re.compile(r'(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}: | ([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|[fF][eE]80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::([fF]{4}(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))')
    
    for v in file_number:
        file_name = f"./File/FrontendFile/storm-frontend-202003{v}.txt"
        print("reading ", file_name)
    
        data = pd.read_csv(file_name, nrows=5e4)
        f = data.message

        specific_substitute = 'IP'

        for i in f.index:
            f[i] = re.sub(ipv6_address, specific_substitute, f[i])
            f[i] = re.sub(ipv4_address, specific_substitute, f[i])

        data.message = f
        
        print(f"saving storm-frontend-202003{v}-mask.txt")
#         data.to_csv(f"./File/FrontendFileMask/storm-frontend-202003{v}-mask.txt", index=False)
        data.to_csv(f"./Tesi-Operational-Intelligence/storm-frontend-202003{v}-mask.txt", index=False)
        
t0= time.time()

masking_frontend_data(["07"])
# masking_frontend_data(["08","09","10","11","12", "13"])

print(f"done in {(time.time()-t0)}")