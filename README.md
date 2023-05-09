# private-quantiles

Code for [Learning-augmented private algorithms for multiple quantile release](https://arxiv.org/abs/2210.11222), to appear in ICML 2023. 
The scripts `static.py`, `pubpri.py`, and `online.py` are for the experiments in Sections 3.3, 5.1, and 5.2, respectively.
Note some scripts may download potentially large datasets, and the file `citibike/worldnews/comments.pkl.zip` needs to be unzipped before running `online.py`.
The `Dockerfile` describes the Python environment used.

```
@inproceedings{khodak2023learning,
  author={Mikhail Khodak and Kareem Amin and Travis Dick and Sergei Vassilvitskii},
  title={Learning-Augmented Private Algorithms for Multiple Quantile Release},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  year={2023}
}
```
