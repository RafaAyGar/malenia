import os
import sys
import malenia
from joblib import dump


class Launcher:
    def __init__(
        self,
        methods,
        datasets,
        cv,
        results_path,
        seeds = 30,
        submission_params=None,
    ):
        self.methods = methods
        self.datasets = datasets
        self.cv = cv
        self.results_path = results_path
        self.seeds = seeds
        self.submission_params = submission_params

        self.condor_files_path = "condor_files/"
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
        with open(dest_path, 'wb') as f:
            dump(content, f)
        return dest_path
    
    def _extract_global_and_specific_method_name(self, method_name):
        method_name_global = method_name.split("_")[0]
        method_name_specif = ""
        for part in method_name.split("_")[1:]:
            method_name_specif += part + "_"
        method_name_specif = method_name_specif[:-1] # remove last "_"

        return method_name_global, method_name_specif

    def launch(
        self,
        overwrite_predictions = False,
        predict_on_train = False,
        save_fitted_methods = False,
        overwrite_fitted_methods = False
    ):
        params = ""
        for dataset in self.datasets:
            dataset_path = self._dump_in_condor_tmp_path(dataset.name, dataset)
            for method_name, method in self.methods.items():
                method_name_global, method_name_specif = self._extract_global_and_specific_method_name(method_name)
                for seed in range(self.seeds):
                    if hasattr(method, "random_state"):
                        method.random_state = seed
                    cv_path = self._dump_in_condor_tmp_path("cv", self.cv)
                    method_path = self._dump_in_condor_tmp_path(method_name, method)
                    results_filename = "seed_" + str(seed)
                    results_path = os.path.join(self.results_path, method_name_global, method_name_specif, dataset.name, results_filename)
                    if (
                        os.path.exists(results_path + "_test.csv")
                        and (os.path.exists(results_path + "_train.csv") or not predict_on_train)
                        and overwrite_predictions == False
                        and overwrite_fitted_methods == False
                    ):
                        print(f"SKIPPING - {results_path} already exists")
                        continue

                    params += (
                        dataset_path + "," +
                        method_path  + "," +
                        cv_path      + "," +
                        str(seed) + "," +
                        str(overwrite_fitted_methods) + "," +
                        str(overwrite_predictions) + "," +
                        str(predict_on_train) + "," +
                        str(save_fitted_methods) + "," +
                        str(method_name_global) + "__" + str(method_name_specif) + "__" + str(dataset.name) + "__" + str(seed) + "," +
                        self.results_path + "," +
                        dataset.name + "\n"
                    )

        with open(self.condor_tmp_path + "/task_params.txt", 'w') as f:
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
        thread_path = os.path.join(malenia.__path__[0], "thread.py")
        if not os.path.exists(thread_path):
            raise Exception("thread.py not found in", thread_path)

        with open(self.condor_tmp_path + '/task.sub', 'w') as f:
                file_str = ( f"""
                    batch_name \t = {condor_params.batch_name}
                    executable \t  = {python_path}
                    arguments \t  = {thread_path} $(dataset_path) $(method_path) $(cv_path) $(seed) $(overwrite_methods) $(overwrite_preds) $(predict_on_train) $(save_fitted_methods) $(job_output_path) $(results_path) $(dataset_name)
                    getenv \t  = {str(condor_params.getenv)}
                    output \t  =   {output_path}/$(job_output_path)_out.out
                    error \t  =   {output_path}/$(job_output_path)_error.err
                    log \t  =   {output_path}/log.log
                    should_transfer_files \t  =   {condor_params.should_transfer_files}
                    request_CPUs \t  =   {str(condor_params.request_CPUs)}
                    request_GPUs \t  =   {str(condor_params.request_GPUs)}
                    request_memory \t  =   {condor_params.request_memory}
                    requirements \t  =   {condor_params.requirements}
                    queue dataset_path, method_path, cv_path, seed, overwrite_methods, overwrite_preds, predict_on_train, save_fitted_methods, job_output_path, results_path, dataset_name from {self.condor_tmp_path}/task_params.txt
                """
                )
                f.write(file_str)
                f.close()



        