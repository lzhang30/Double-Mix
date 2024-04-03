# Double-Mix Pseudo-Label Framework

Here is the code for our proposed Double-Mix Pseudo-Label Framework.

## Data Preprocess
To prepare the dataset, you can follow the work of [DHC](https://github.com/xmed-lab/DHC).

You can also use the ``` dmp/code/data/preprocess_amos.py ``` to prepare the dataset.

The splits are available at ``` {dataset}_data/split ```.
## Model Training
Run 
```
cd dmp
bash train3times_seeds_20p.sh -c 0 -t synapse -m cdifw_dmp_ours -e 'test' -l 3e-2 -w 0.1
```
Have fun.
