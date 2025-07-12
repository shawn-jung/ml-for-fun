# Dash App for League of Legends Gameplay Outcome Prediction
# Authored by Shawn Jung
# Last Update: 7/12/2025

import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import joblib
import numpy as np


# Load NB classifier
nb_clf = joblib.load("serving/model.joblib")

# Function to create new test data from inputs


def binning(feature, bins):
    if feature is not None:
        digit_bin = [-float("inf")] + bins + [float("inf")]
        feature = np.digitize(feature, digit_bin, right=False) - 1
        return feature
    else:
        return None


def new_row_helper(
    blueWardsPlaced=None,
    blueWardsDestroyed=None,
    blueFirstBlood=None,
    blueKills=None,
    blueDeaths=None,
    blueAssists=None,
    blueDragons=None,
    blueHeralds=None,
    blueTowersDestroyed=None,
    blueTotalGold=None,
    blueTotalExperience=None,
    blueTotalMinionsKilled=None,
    blueTotalJungleMinionsKilled=None,
    redWardsPlaced=None,
    redWardsDestroyed=None,
    redAssists=None,
    redDragons=None,
    redHeralds=None,
    redTowersDestroyed=None,
    redTotalGold=None,
    redTotalExperience=None,
    redTotalMinionsKilled=None,
    redTotalJungleMinionsKilled=None,
):
    """This fuctions gets feature values and create them into a single row dataframe for prediction."""

    # binning for four features

    blueWardsPlaced = binning(blueWardsPlaced, [14, 16, 20])
    redWardsPlaced = binning(redWardsPlaced, [14, 16, 20])
    blueWardsDestroyed = binning(blueWardsDestroyed, [1, 3, 4])
    redWardsDestroyed = binning(redWardsDestroyed, [1, 3, 4])

    test_row = np.array(
        [
            blueWardsPlaced,
            blueWardsDestroyed,
            blueFirstBlood,
            blueKills,
            blueDeaths,
            blueAssists,
            blueDragons,
            blueHeralds,
            blueTowersDestroyed,
            blueTotalGold,
            blueTotalExperience,
            blueTotalMinionsKilled,
            blueTotalJungleMinionsKilled,
            redWardsPlaced,
            redWardsDestroyed,
            redAssists,
            redDragons,
            redHeralds,
            redTowersDestroyed,
            redTotalGold,
            redTotalExperience,
            redTotalMinionsKilled,
            redTotalJungleMinionsKilled,
        ]
    )

    return test_row


# reshape the fitted NB classifier information
reshaped_summary = {}

for target in range(nb_clf.theta_.shape[0]):
    lst = []
    for feature in range(nb_clf.theta_.shape[1]):
        lst.append(
            (
                nb_clf.theta_[target][feature],
                nb_clf.var_[target][feature],
                nb_clf.class_count_[target],
            )
        )

    reshaped_summary[target] = lst


def custom_predict_proba(summaries, test_row):
    """This function gets gaussian summaries of each feature, a test row and return joint probabilities.
    I refer to the Scikit Learn's naive-bayess.py code for the calculation of joint log likelihood and
    LogSumExp trick application https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/naive_bayes.py"""

    total_rows = sum([summaries[label][0][2] for label in summaries])
    joint_log_likelihood = []

    for class_value, class_summaries in summaries.items():
        jointi = np.log(summaries[class_value][0][2] / float(total_rows))

        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            if test_row[i] is not None:
                n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * stdev))
                n_ij -= 0.5 * np.sum(((test_row[i] - mean) ** 2) / stdev)

                jointi += n_ij

        joint_log_likelihood.append(jointi)

    log_sum_exp = np.log(np.sum(np.exp(joint_log_likelihood)))
    proba = np.exp(joint_log_likelihood - log_sum_exp)

    blue_win_odds = round(proba[1] / proba[0], 4)
    red_win_odds = round(proba[0] / proba[1], 4)
    blue_win_prob = round(proba[1], 4)
    red_win_prob = round(proba[0], 4)

    if blue_win_prob > red_win_prob:
        blue_win = 1
    else:
        blue_win = 0

    return blue_win, blue_win_odds, red_win_odds, blue_win_prob, red_win_prob


# Initialize odds and probabilities for the plot
blue_win_odds, red_win_odds = 1.0, 1.0
win_probs = [50.0, 50.0]
teams = ["Blue Team", "Red Team"]
colors = ["skyblue", "crimson"]
odds_text = "Odds 1 to 1"

fig = go.Figure(
    [
        go.Bar(
            x=win_probs,
            y=teams,
            text=win_probs,
            textposition="auto",
            marker_color=colors,
            orientation="h",
            width=[0.6, 0.6],
        )
    ]
)
fig.update_layout(height=200, margin=dict(l=50, r=50, b=30, t=40))
fig.update_xaxes(
    title_text="Team Winning Probability (%)",
    title_font=dict(color="black", size=15),
    range=[0, 100],
)
fig.update_yaxes(title_font=dict(size=15), range=[-0.5, 2])
fig.add_annotation(
    x=90,
    y=1.7,
    text=odds_text,
    showarrow=False,
    align="left",
    bordercolor="#c7c7c7",
    borderwidth=2,
    borderpad=4,
    bgcolor="pink",
    font=dict(size=10, color="navy"),
)

# loading Dash
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "LoL Wins Prediction"

app.layout = html.Div(
    [
        html.Div(
            [
                html.H4("League of Legends - Winning Team Prediction"),
                html.P("This app predicts the winning team from the first 10 minutes of game data. "),
                html.A(
                    "The related blog post",
                    href="https://shawnjung.blog/2020/06/29/league-of-legend-winning-prediction/",
                    target="_blank",
                ),
                html.Br(),
                dcc.Graph(
                    id="win-plot",
                    figure=fig,
                    style={
                        "width": "80%",
                        "align-items": "center",
                        "display": "inline-block",
                        "justify-content": "center",
                    },
                ),
            ],
            id="my-div",
            style={"textAlign": "center"},
        ),
        html.Div(
            [
                html.H6("Wards & Towers"),
                html.Label("Wards Placed by Blue"),
                dcc.Input(id="blueWardsPlaced", type="number", placeholder="(0~300)"),
                html.Label("Wards Destroyed by Blue"),
                dcc.Input(
                    id="blueWardsDestroyed",
                    type="number",
                    placeholder="(0~300)",
                    min=0,
                    max=300,
                ),
                html.Label("Wards Placed by Red"),
                dcc.Input(
                    id="redWardsPlaced",
                    type="number",
                    placeholder="(0~300)",
                    min=0,
                    max=300,
                ),
                html.Label("Wards Destroyed by Red"),
                dcc.Input(
                    id="redWardsDestroyed",
                    type="number",
                    placeholder="(0~300)",
                    min=0,
                    max=300,
                ),
                html.Label("Towers Destroyed by Blue"),
                dcc.Dropdown(
                    id="blueTowersDestroyed",
                    options=[
                        {"label": "0", "value": 0},
                        {"label": "1", "value": 1},
                        {"label": "2", "value": 2},
                    ],
                    multi=False,
                ),
                html.Label("Towers Destroyed by Red"),
                dcc.Dropdown(
                    id="redTowersDestroyed",
                    options=[
                        {"label": "0", "value": 0},
                        {"label": "1", "value": 1},
                        {"label": "2", "value": 2},
                    ],
                    multi=False,
                ),
            ],
            id="warding_totems",
            className="pretty_container two columns",
        ),
        html.Div(
            [
                html.H6("Kills"),
                html.Label("First Kill of The Game"),
                dcc.RadioItems(
                    id="blueFirstBlood",
                    options=[
                        {"label": "Blue", "value": 1},
                        {"label": "Red", "value": 0},
                    ],
                    labelStyle={"display": "inline-block"},
                ),
                html.Label("Kills by Blue(Red Deaths)"),
                dcc.Input(
                    id="blueKills",
                    type="number",
                    placeholder="Input data",
                    min=0,
                    max=40,
                ),
                html.Label("Kills by Red(Blue Deaths)"),
                dcc.Input(
                    id="blueDeaths",
                    type="number",
                    placeholder="Input data",
                    min=0,
                    max=40,
                ),
                html.Label("Kill Assists by Blue"),
                dcc.Input(
                    id="blueAssists",
                    type="number",
                    placeholder="Input data",
                    min=0,
                    max=40,
                ),
                html.Label("Kill Assists by Red"),
                dcc.Input(
                    id="redAssists",
                    type="number",
                    placeholder="Input data",
                    min=0,
                    max=40,
                ),
            ],
            id="kills",
            className="pretty_container two columns",
        ),
        html.Div(
            [
                html.H6("Monsters - Blue"),
                html.Label("Dragons Kill by Blue"),
                dcc.Dropdown(
                    id="blueDragons",
                    options=[
                        {"label": "0", "value": 0},
                        {"label": "1", "value": 1},
                        {"label": "2", "value": 2},
                        {"label": "3", "value": 3},
                        {"label": "4+", "value": 4},
                    ],
                    multi=False,
                ),
                html.Label("Heralds Kill by Blue"),
                dcc.Dropdown(
                    id="blueHeralds",
                    options=[
                        {"label": "0", "value": 0},
                        {"label": "1", "value": 1},
                        {"label": "2", "value": 2},
                        {"label": "3", "value": 3},
                        {"label": "4+", "value": 4},
                    ],
                    multi=False,
                ),
                html.Label("Minions Kill by Blue"),
                dcc.Input(
                    id="blueTotalMinionsKilled",
                    type="number",
                    placeholder="(0~300)",
                    min=0,
                    max=300,
                ),
                html.P("Jungle Minions Kill by Blue"),
                dcc.Input(
                    id="blueTotalJungleMinionsKilled",
                    type="number",
                    placeholder="(0~100)",
                    min=0,
                    max=100,
                ),
            ],
            id="monsters-blue",
            className="pretty_container two columns",
        ),
        html.Div(
            [
                html.H6("Monsters - Red"),
                html.Label("Dragons Kill by Red"),
                dcc.Dropdown(
                    id="redDragons",
                    options=[
                        {"label": "0", "value": 0},
                        {"label": "1", "value": 1},
                        {"label": "2", "value": 2},
                        {"label": "3", "value": 3},
                        {"label": "4+", "value": 4},
                    ],
                    multi=False,
                ),
                html.Label("Heralds Kill by Red"),
                dcc.Dropdown(
                    id="redHeralds",
                    options=[
                        {"label": "0", "value": 0},
                        {"label": "1", "value": 1},
                        {"label": "2", "value": 2},
                        {"label": "3", "value": 3},
                        {"label": "4+", "value": 4},
                    ],
                    multi=False,
                ),
                html.Label("Total Minions Kill by Red"),
                dcc.Input(
                    id="redTotalMinionsKilled",
                    type="number",
                    placeholder="(0~300)",
                    min=0,
                    max=300,
                ),
                html.Label("Total Jungle Minions Kill by Red"),
                dcc.Input(
                    id="redTotalJungleMinionsKilled",
                    type="number",
                    placeholder="(0~100)",
                    min=0,
                    max=100,
                ),
            ],
            id="monsters-red",
            className="pretty_container two columns",
        ),
        html.Div(
            [
                html.H6("Gold & Exp"),
                html.Label("Total Gold by Blue"),
                dcc.Input(
                    id="blueTotalGold",
                    min=0,
                    max=30000,
                    placeholder="(0~30000)",
                    type="number",
                ),
                html.Label("Total Gold by Red"),
                dcc.Input(
                    id="redTotalGold",
                    min=0,
                    max=30000,
                    placeholder="(0~30000)",
                    type="number",
                ),
                html.Label("Total Exp by Blue"),
                dcc.Input(
                    id="blueTotalExperience",
                    min=0,
                    max=30000,
                    placeholder="(0~30000)",
                    type="number",
                ),
                html.Label("Total Exp by Red"),
                dcc.Input(
                    id="redTotalExperience",
                    min=0,
                    max=30000,
                    placeholder="(0~30000)",
                    type="number",
                ),
            ],
            id="gold-exp",
            className="pretty_container two columns",
        ),
    ]
)


@app.callback(
    Output(component_id="win-plot", component_property="figure"),
    [
        Input(component_id="blueWardsPlaced", component_property="value"),
        Input(component_id="blueWardsDestroyed", component_property="value"),
        Input(component_id="blueFirstBlood", component_property="value"),
        Input(component_id="blueKills", component_property="value"),
        Input(component_id="blueDeaths", component_property="value"),
        Input(component_id="blueAssists", component_property="value"),
        Input(component_id="blueDragons", component_property="value"),
        Input(component_id="blueHeralds", component_property="value"),
        Input(component_id="blueTowersDestroyed", component_property="value"),
        Input(component_id="blueTotalGold", component_property="value"),
        Input(component_id="blueTotalExperience", component_property="value"),
        Input(component_id="blueTotalMinionsKilled", component_property="value"),
        Input(component_id="blueTotalJungleMinionsKilled", component_property="value"),
        Input(component_id="redWardsPlaced", component_property="value"),
        Input(component_id="redWardsDestroyed", component_property="value"),
        Input(component_id="redAssists", component_property="value"),
        Input(component_id="redDragons", component_property="value"),
        Input(component_id="redHeralds", component_property="value"),
        Input(component_id="redTowersDestroyed", component_property="value"),
        Input(component_id="redTotalGold", component_property="value"),
        Input(component_id="redTotalExperience", component_property="value"),
        Input(component_id="redTotalMinionsKilled", component_property="value"),
        Input(component_id="redTotalJungleMinionsKilled", component_property="value"),
    ],
)
def update_figure(
    blueWardsPlaced,
    blueWardsDestroyed,
    blueFirstBlood,
    blueKills,
    blueDeaths,
    blueAssists,
    blueDragons,
    blueHeralds,
    blueTowersDestroyed,
    blueTotalGold,
    blueTotalExperience,
    blueTotalMinionsKilled,
    blueTotalJungleMinionsKilled,
    redWardsPlaced,
    redWardsDestroyed,
    redAssists,
    redDragons,
    redHeralds,
    redTowersDestroyed,
    redTotalGold,
    redTotalExperience,
    redTotalMinionsKilled,
    redTotalJungleMinionsKilled,
):
    test_row = new_row_helper(
        blueWardsPlaced,
        blueWardsDestroyed,
        blueFirstBlood,
        blueKills,
        blueDeaths,
        blueAssists,
        blueDragons,
        blueHeralds,
        blueTowersDestroyed,
        blueTotalGold,
        blueTotalExperience,
        blueTotalMinionsKilled,
        blueTotalJungleMinionsKilled,
        redWardsPlaced,
        redWardsDestroyed,
        redAssists,
        redDragons,
        redHeralds,
        redTowersDestroyed,
        redTotalGold,
        redTotalExperience,
        redTotalMinionsKilled,
        redTotalJungleMinionsKilled,
    )
    _, blue_win_odds, red_win_odds, blue_win_prob, red_win_prob = custom_predict_proba(reshaped_summary, test_row)
    win_probs = [round(100 * blue_win_prob, 2), round(100 * red_win_prob, 2)]

    if blue_win_odds >= red_win_odds:
        odds_text = "Odds " + str(round(blue_win_odds, 2)) + " to 1"
    else:
        odds_text = "Odds " + str(round(red_win_odds, 2)) + " to 1"

    fig = go.Figure(
        [
            go.Bar(
                x=win_probs,
                y=teams,
                text=win_probs,
                textposition="auto",
                marker_color=colors,
                orientation="h",
                width=[0.6, 0.6],
            )
        ]
    )
    fig.update_layout(height=200, margin=dict(l=50, r=50, b=30, t=40))
    fig.update_xaxes(
        title_text="Team Winning Probability (%)",
        title_font=dict(color="black", size=15),
        range=[0, 100],
    )
    fig.update_yaxes(title_font=dict(size=15), range=[-0.5, 2])
    fig.add_annotation(
        x=90,
        y=1.7,
        text=odds_text,
        showarrow=False,
        align="left",
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="pink",
        font=dict(size=10, color="navy"),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
