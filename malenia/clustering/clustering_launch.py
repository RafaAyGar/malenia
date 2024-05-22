import os
import sys

import malenia
from joblib import dump


class ClusteringLauncher:
    def __init__(
        self,
        methods,
        datasets,
        results_path,
        condor_files_path="condor_files",
        submission_params=None,
        do_train_test_split=False,
    ):
        self.methods = methods
        self.datasets = datasets
        self.results_path = results_path

        self.submission_params = submission_params
        self.do_train_test_split = do_train_test_split

        self.condor_files_path = condor_files_path
        self.condor_tmp_path = os.path.join(self.condor_files_path, "tmp")

        # remove all files and folders from self.condor_files_path if they exist
        if os.path.exists(self.condor_files_path):
            for root, dirs, files in os.walk(self.condor_files_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

    def _dump_in_condor_tmp_path(self, file_name, content):
        dest_path = os.path.join(self.condor_tmp_path, file_name)
        if not os.path.exists(os.path.dirname(dest_path)):
            os.makedirs(os.path.dirname(dest_path))
        with open(dest_path, "wb") as f:
            dump(content, f)
        return dest_path

    def _extract_global_and_specific_method_name(self, method_name):
        method_name_global = method_name.split("_")[0]
        method_full_name = method_name.split("_seed")[0]

        if (
            method_name_global == method_full_name
        ):  # if the specific method name is not specified
            method_name_specific = "Default"
            seed = method_name.split("_seed")[1]
        else:
            method_name_specif_with_seed = ""
            for part in method_name.split("_")[1:]:
                method_name_specif_with_seed += part + "_"
            method_name_specific = method_name_specif_with_seed.split("_seed")[0]
            seed = method_name_specif_with_seed.split("_seed")[1][
                :-1
            ]  # [:-1] to remove last "_"

        return method_name_global, method_name_specific, seed

    def launch(
        self,
        overwrite_predictions=False,
        save_final_cluster_distances=False,
    ):
        params = ""
        for dataset in self.datasets:
            dataset_path = self._dump_in_condor_tmp_path(dataset.name, dataset)
            for method_name, method in self.methods.items():
                (
                    method_name_global,
                    method_name_specif,
                    seed,
                ) = self._extract_global_and_specific_method_name(method_name)

                method_path = self._dump_in_condor_tmp_path(method_name, method)
                results_filename = "seed_" + seed
                results_path = os.path.join(
                    self.results_path,
                    method_name_global,
                    method_name_specif,
                    dataset.name,
                    results_filename,
                )
                if (
                    os.path.exists(results_path + "_test.csv")
                    and (os.path.exists(results_path + "_train.csv"))
                    and overwrite_predictions == False
                ):
                    print(f"SKIPPING - {results_path} already exists")
                    continue

                params += (
                    dataset_path
                    + ","
                    + method_path
                    + ","
                    + seed
                    + ","
                    + str(overwrite_predictions)
                    + ","
                    + str(save_final_cluster_distances)
                    + ","
                    + (
                        str(method_name_global)
                        + "___"
                        + str(method_name_specif)
                        + "___"
                        + str(dataset.name)
                        + "___"
                        + seed
                    )
                    + ","
                    + self.results_path
                    + ","
                    + dataset.name
                    + "\n"
                )

        with open(self.condor_tmp_path + "/task_params.txt", "w") as f:
            f.write(params)
            f.close()

        self._write_condor_task_sub(self.submission_params)
        os.system("condor_submit " + self.condor_tmp_path + "/task.sub")

    def _write_condor_task_sub(self, condor_params):
        if not os.path.exists(self.condor_files_path):
            os.system("mkdir " + self.condor_files_path)
        output_path = os.path.join(self.condor_files_path, "condor_output")
        if not os.path.exists(output_path):
            os.system("mkdir " + output_path)

        for file in os.listdir(output_path):
            os.remove(output_path + file)

        # get python environment path
        python_path = sys.executable

        if self.do_train_test_split:
            thread_path = os.path.join(
                malenia.__path__[0], "clustering/clustering_thread_train_split.py"
            )
        else:
            thread_path = os.path.join(malenia.__path__[0], "clustering/clustering_thread.py")

        if not os.path.exists(thread_path):
            raise Exception("thread.py not found in", thread_path)

        with open(self.condor_tmp_path + "/task.sub", "w") as f:
            file_str = f"""
                    batch_name \t = {condor_params.batch_name}
                    executable \t  = {python_path}
                    arguments \t  = {thread_path} $(dataset_path) $(method_path) $(seed) $(overwrite_preds) $(save_final_cluster_distances) $(job_output_path) $(results_path) $(dataset_name)
                    getenv \t  = {str(condor_params.getenv)}
                    output \t  =   {output_path}/$(job_output_path)_out.out
                    error \t  =   {output_path}/$(job_output_path)_error.err
                    log \t  =   {output_path}/log.log
                    should_transfer_files \t  =   {condor_params.should_transfer_files}
                    request_CPUs \t  =   {str(condor_params.request_CPUs)}
                    request_GPUs \t  =   {str(condor_params.request_GPUs)}
                    request_memory \t  =   {condor_params.request_memory}
                    requirements \t  =   {condor_params.requirements}
                    queue dataset_path, method_path, seed, overwrite_preds, save_final_cluster_distances, job_output_path, results_path, dataset_name, from {self.condor_tmp_path}/task_params.txt
                """
            f.write(file_str)
            f.close()
