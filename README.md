# Double-Mix Pseudo-Label Framework

Here is the code for our work Double-Mix Pseudo-Label Framework.

## Data Preprocess
To preprocess the dataset, you can follow the work of [DHC](https://github.com/xmed-lab/DHC).

You can also use the ``` lzhang/dmp/code/data/preprocess_amos.py ``` to preprocess the dataset.

The splits are available at ``` {dataset}_data/split ```.
## Model Training
Run 
```
cd dmp
bash train3times_seeds_20p.sh -c 0 -t cdifw_dmp_ours -m  -e '' -l 3e-2 -w 0.1
```
Have fun.
