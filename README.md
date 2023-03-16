# malenia
#### Machine learning automatization package to ease experiments launching and results extraction

## Functionalities

#### Check condor errors:
Check if there is any error in your condor_output .err files.
- In an integrated terminal, with the malenia package installed in your python environnement, run:

```bash
python
import malenia
malenia.check_condor_errors("path_to_condor_output")
```