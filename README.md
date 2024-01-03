# malenia
Enhance your experiment launch and results extraction processes with the advanced machine learning automation package *malenia*. Streamline the entire workflow, making it more efficient and user-friendly for seamless experimentation and effortless result retrieval. Currently the malenia experiments launching module is specialized in condor administrated systems, but we are working to extend its functionalities to any type of launching environnement.

## Functionalities

#### Check condor errors:
Check if there is any error in your condor_output .err files.
- In an integrated terminal, with the malenia package installed in your python environnement, run:

```bash
malenia_check_condor_errors
```

#### Testing with pytest:
Check that the malenia package functionalities work properly (currently only support results extraction functionality tests, i.e. checks that results tables are being contructed without errors).
- In an integrated terminal, with the malenia package installed in your python environnement, run:

```bash
malenia_test
```
