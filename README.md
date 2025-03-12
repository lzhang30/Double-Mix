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

### Training Data Percentage:

The notation 20p represents training with 20% labeled data. You can modify this value to `train3times_seeds_40p`, `train3times_seeds_5p`, etc., to indicate training with 40%, 5%, and so on.

### Command-line Parameters:

`-c`: Specifies which GPU to use for training.

`-t`: Defines the task, which can be either synapse or amos.

`-m`: Specifies the training method. The available methods include:

1 `cdifw_dmp_ours` (our proposed method)

2 `cdifw` (ablation studies)

`-e`: Defines the name of the current experiment. default: `'test'`

`-l`: Sets the learning rate. In this experiment, it was set to `0.1`

`-w`: Specifies the weight of the unsupervised loss.


Have fun.

## Cite

If this code is helpful for your study, welcome to cite our paper
```
@article{zhang2025double,
  title={Double-mix pseudo-label framework: enhancing semi-supervised segmentation on category-imbalanced CT volumes},
  author={Zhang, Luyang and Hayashi, Yuichiro and Oda, Masahiro and Mori, Kensaku},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--12},
  year={2025},
  publisher={Springer}
}
```
