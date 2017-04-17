# jupyter notebook
## Installation
```bash
sudo pip install --upgrade pip
sudo pip install ipython
sudo pip install jupyter
```
## Run
```bash
jupyter notebook
```

## Troubleshoot
* Trouble with TensorFlow in Jupyter Notebook
http://stackoverflow.com/a/39355763

* Version
> 어제 안되었던 부분은 numpy가 아마 버전이 낮아서 그랬던것 같음
(내거에 설치된건 1.6.x)
numpy 버전을 upgrade(1.11.x)하고  jupyter  지우고 다시 설치하니 되네..
numpy때문에 코드중에서 아래 부분 에러 남
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

