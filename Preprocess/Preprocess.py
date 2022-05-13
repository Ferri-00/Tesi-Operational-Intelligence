def preprocess_data(file_number):
    for v in file_number:
#         file_name = f"./logs_7-13march/storm-atlas/storm-frontend-server.log-202003{v}.gz"
        file_name = f"./logs_7-13march/storm-atlas-1/storm-frontend-server.log-202003{v}"
        
        print("reading ", file_name)
        
        with open(f'./File/PreFile/storm-frontend-server.log-202003{v}', 'w') as fwrite:
            with open(file_name, 'r') as fread:
                for row in fread:
                    fwrite.write(row.replace(',', ' '))
                    
preprocess_data(["07"])
preprocess_data(["08","09","10","11","12", "13"])