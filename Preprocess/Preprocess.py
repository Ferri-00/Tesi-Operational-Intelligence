from time import time
import sys

def preprocess_data(file_number):
    for v in file_number:
        file_name = f"/home/ATLAS-T3/eferri/logs_7-13march/storm-atlas/storm-frontend-server.log-202003{v}"
        
        print("reading ", file_name)
        
        with open(f'/home/ATLAS-T3/eferri/File/PreFile/storm-frontend-server.log-202003{v}', 'w') as fwrite:
            with open(file_name, 'r') as fread:
                for row in fread:
                    fwrite.write(row.replace(',', ' '))

if __name__ == "__main__":
    t0 = time()

    file_number = list(sys.argv[1:])
    print('File number:', file_number)
    preprocess_data(file_number)

    print(f"done in {int((time()-t0)/60)} minutes and {((time()-t0)%60)} seconds")
