import os


def check_condor_errors(
    directory=None, father_dir="/home/rayllon/INCEPTION/orchestator/", cpu_or_gpu="cpu"
):
    """
    This function checks the condor output files for errors, and print the file in case of finding an error.
    """
    if directory is None:
        if cpu_or_gpu == "cpu":
            directory = os.path.join(father_dir, "condor_files_cpu/condor_output")
        elif cpu_or_gpu == "gpu":
            directory = os.path.join(father_dir, "condor_files_gpu/condor_output")
    if "condor_output" not in directory:
        print("ERROR: The directory must be a condor_output directory.")
        return
    files = os.listdir(directory)
    files.sort()
    for file in files:
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
                    print(
                        "************************************************************************************\n"
                    )
                    print(content)
                    print(
                        "************************************************************************************\n"
                    )
                    print("---")
                f.close()


def check_condor_dones(
    directory=None,
):
    """
    This function checks the condor output files for errors, and print the file in case of finding an error.
    """
    if "condor_output" not in directory:
        print("ERROR: The directory must be a condor_output directory.")
        return
    dones_found = 0
    total = 0
    files = os.listdir(directory)
    files.sort()
    for file in files:
        if ".err" in file:
            total += 1
            with open(directory + "/" + file, "r") as f:
                done_found = False
                content = ""
                for line in f:
                    content += line
                    if ("done" in line.lower()) and (not "error" in line.lower()):
                        done_found = True
                if done_found:
                    dones_found += 1
                    print("--- DONE: " + file + " ---\n")
                f.close()
    print("\n\n")
    print(f"Done Processes --> {dones_found}/{total}")


def check_condor_undones(
    directory=None,
):
    """
    This function checks the condor output files for errors, and print the file in case of finding an error.
    """
    if "condor_output" not in directory:
        print("ERROR: The directory must be a condor_output directory.")
        return
    undones_found = 0
    total = 0
    files = os.listdir(directory)
    files.sort()
    for file in files:
        if ".err" in file:
            total += 1
            with open(directory + "/" + file, "r") as f:
                undone_found = True
                content = ""
                for line in f:
                    content += line
                    if ("done" in line.lower()) or ("error" in line.lower()):
                        undone_found = False
                if undone_found:
                    undones_found += 1
                    print("--- UNDONE: " + file + " ---\n")
                f.close()
    print("\n\n")
    print(f"Undone Processes --> {undones_found}/{total}")
