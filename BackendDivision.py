import pandas as pd

def backend_data(file_number):
    for v in file_number:
#         file_name = f"./logs_7-13march/storm-atlas/storm-backend-2020-03-{v}.log"
        file_name = f"./logs_7-13march/storm-atlas-1/storm-backend-2020-03-{v}.log"

        backend = pd.DataFrame(open(file_name).readlines())
        backend = backend[0].str.split(' - ', expand=True)
        backend = backend.rename(columns={0:"time", 1:"INFO", 2: "message", 3:"message", 4:"message"})
        
        print(f"saving storm-backend-2020-03-{v}.txt")
        backend.to_csv(f"./File/BackendFile/storm-backend-2020-03-{v}.txt", index=False)

print("Starting division of files")
# backend_data(["07","08","09","10","11","12", "13"])
backend_data(["07"])