import os

def check_condor_errors(directory="/home/rayllon/INCEPTION/orchestator_cpu/condor_files/condor_output"):
    """
        This function checks the condor output files for errors, and print the file in case of finding an error.
    """
    for file in os.listdir(directory):
        if ".err" in file:
            with open(directory + "/" + file, "r") as f:
                error_found = False
                content = ""
                for line in f:
                    content += line
                    if "error" in line.lower():
                        error_found = True
                if error_found:
                    print("\n \n")
                    print("--- ERROR found in file: " + file + " ---\n")
                    print("************************************************************************************\n")
                    print(content)
                    print("************************************************************************************\n")
                    print("---")
                f.close()