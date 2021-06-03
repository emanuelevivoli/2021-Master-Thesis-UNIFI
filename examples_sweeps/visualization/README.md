# Sweep folder
---
Each file is a sweep configuration varing by:
- pre dimentionality reduction algorithm
- clustering algorithm
- post dimentionality reduction algorithm

In each file has been wrote the number of total combination that will be 
investivate by the `grid search`. In this direction, to know the total run
you just need to either run the `bash` file (`count_tot.sh`) that counts all `tot: x` for each yaml
file in this folder, or run the command:
```bash
cat sweep*.yaml | grep tot: |  sed 's/[^0-9]*//g' | awk '{s+=$1} END {print s}'
```
from inside this folder.