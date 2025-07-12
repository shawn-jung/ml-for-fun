## League Of Legends - Winning Prediction from the First 10-minutes Game Data

This repo includes an exploration Notebook file as I build the League of Legends game outcome prediction with Naive Bayesian method.
For the detailed explanation, please refer to [this blog post](https://shawnjung.blog/2020/07/03/league-of-legend-winning-prediction/)
The deployed prediction app can be accessed [from this link](http://shawnjung.pythonanywhere.com/)

### Artifacts

- [Notebook on the Approach and Data Exploration](league_of_legend_win_prediction_with_bayesian.ipynb)
- [Dash App implementation (app.py)](https://github.com/shawn-jung/ml-for-fun/blob/master/lol_predict/app.py)
- model.joblib - Exported NB training result from the Notebook

### For readers who want to run the app locally

While the original blog post was written in 2020 and I used Conda back then,
I advise you to use UV for your sanity. [Install UV](https://docs.astral.sh/uv/) on your machine and synchronize your local repo.

```bash
uv sync
```

And run the app

```
uv run app.py
```

You can access http://localhost:8051/ from your local browser

Docker is also a good option to taste the local app.

```bash
docker build -t lol-app .
docker run -d -p 8051:8051 lol-app
```
