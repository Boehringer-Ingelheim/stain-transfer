import base64
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from dash import dcc, html, Dash, dash_table
from dash.dependencies import Input, Output, State
from dash_extensions import BeforeAfter
from plotly.subplots import make_subplots

from src import ASSETS_DIR
from src.data_processing.metrics import get_hist
from src.utils.utils import csv2multicolumn, mt_rename, get_triplets

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from dash_extensions.enrich import DashProxy, MultiplexerTransform

# Folder names
ASSETS = Path("assets")
HISTS = 'hists'
# Component IDs
I = "img"
M = "Metrics"
ST = "Stats"
BA = "BeforeAfter"
D = "Div"
S = 'source'
F = 'fakes'
T = 'target'


def assets_symlink(file: Path):
    """
    Makes an assets symbolic link to file.
    :param file: File to link to.
    """

    try:
        (ASSETS_DIR / file.name).symlink_to(file.resolve())
    except FileExistsError:
        pass


def read_image(img_file):
    """
    Reads an image file.

    :param img_file: Image file to be read.
    """

    return base64.b64encode(open(img_file, 'rb').read()).decode()


def df2table(df: pd.DataFrame, row: str, column_substring: str):
    """
    Styles a DataFrame for the dash table.

    :param df: DataFrame.
    :param row: Row indexer.
    :param column_substring: Substring contained in desired columns.
    """

    cols = [x for x in df.columns if column_substring in x]
    metrics = pd.DataFrame(df.loc[row, cols]).round(5).astype(str)
    metrics[row] = metrics[row].apply(lambda x: ": " + x)
    metrics.reset_index(inplace=True)
    metrics['index'] = metrics['index'].apply(
        lambda x: x.split(column_substring)[-1])
    metrics[M] = metrics['index'] + metrics[row]

    return metrics.to_dict('records')


def save_hist(hist_path: Path, img_file: Path, suffix: str = ""):
    """
    Save a histogram for an image if it doesn't exist.

    :param hist_path: path where histograms are stored.
    :param img_file: input image file.
    :param suffix: suffix to add to image name. Only for images generated without
     model name.
    :return: saved histogram file name.
    """

    suffixed = hist_path / f"{img_file.stem}{suffix}{img_file.suffix}"
    if not suffixed.exists():
        img_array = np.array(Image.open(img_file).convert('RGB'))
        h, w, _ = img_array.shape
        hist = pd.DataFrame(get_hist(img_array)).T
        fig = px.bar(hist, width=w, height=h,
                     color_discrete_sequence=["red", "green", "blue"])
        fig.update_layout(yaxis={'visible': False}, xaxis={'visible': False},
                          showlegend=False,
                          margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
        fig.write_image(suffixed)


def compute_hists(triplets: dict, hist_path: Path):
    """
    Compute all necessary histograms.

    :param triplets: triplet dictionary of source, fakes and target images.
    :param hist_path: path where histograms are stored.
    """

    hist_path.mkdir(exist_ok=True, parents=True)
    for k, v in triplets.items():
        if 'fake' in k:
            for m, m_values in v.items():
                for stem, file in m_values.items():
                    suffix = ""
                    if not Path(file).stem.endswith(m):
                        # Images generated without model suffix.
                        suffix = f"_{m}"
                    save_hist(hist_path, Path(file), suffix)
        else:
            for stem, file in v.items():
                save_hist(hist_path, Path(file))


def get_file_title_img(triplets: dict, image_type: str, image_file: str,
                       model: str = None):
    """
    Gets the file, title and image from a triplet dictionary.

    :param triplets: triplet dictionary of source, fakes and target images.
    :param image_type: one of ['source','fakes','target']
    :param image_file: image file common stem.
    :param model: model name, only valid for fake images.
    :return:
    """

    if 'fake' in image_type and model is not None:
        file = Path(triplets[image_type][model][image_file])
        title = model
    else:
        file = Path(triplets[image_type][image_file])
        title = image_type.title()
    img = f"data:image/{file.suffix.replace('.', '')};base64,{read_image(file)}"

    return file, title, img


def get_df(path: str, stats: list):
    """
    Gets a multi column csv file with metrics results and removes unnecessary columns.

    :param path: path to csv file.
    :param stats: List of statistics to keep.
    :return: Filtered Dataframe and stats.
    """

    df = csv2multicolumn(path)
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    stats = [x.lower() for x in stats]
    df = df[[x for x in df.columns if x.split(' ')[-1].lower() in stats]]
    available_stats = list(set([x.split(' ')[-1].lower() for x in df.columns]))

    return df, available_stats


def get_children(models: list):
    """
    Dynamically create children for the app.

    :param models: list of available models.
    :return: list of children.
    """

    children = []
    style = {"margin-left": "25px", 'display': 'inline-block'}
    for i in range(len(models)):
        children.append(html.Center([
            html.Div(BeforeAfter(id=f'{BA}_{i}'),
                     style={'display': 'inline-block'}),
            html.Div([dcc.Markdown(id=f'{S}_{i}'), html.Img(id=f'{S}_{I}_{i}')],
                     style=style),
            html.Div([dcc.Markdown(id=f'{F}_{i}'), html.Img(id=f'{F}_{I}_{i}')],
                     style=style),
            html.Div([dcc.Markdown(id=f'{T}_{i}'), html.Img(id=f'{T}_{I}_{i}')],
                     style=style),
            html.Div([dash_table.DataTable(
                id=f'{ST}_{i}',
                style_cell={'textAlign': 'center'},
                columns=[{"name": M, "id": M}])],
                style={'display': 'none'}, id=f'{D}_{i}'),
        ]))

    return children


def get_click_imgs_dependencies(models: list, img_types: list):
    """
    Get Output, Input and State dependencies for clickable images.

    :param models: list of available models.
    :param img_types: list of image types.
    """

    click_imgs_inputs, click_imgs_outputs, click_imgs_states = [], [], []
    for i in range(len(models)):
        for t in img_types:
            click_imgs_outputs += [Output(f'{t}_{I}_{i}', 'src')]
            click_imgs_inputs += [Input(f'{t}_{I}_{i}', 'n_clicks')]
            click_imgs_states += [State(f'{t}_{I}_{i}', 'id'),
                                  State(f'{t}_{I}_{i}', 'title'),
                                  State(f'{t}_{I}_{i}', 'n_clicks'),
                                  State(f'{t}_{I}_{i}', 'alt')]

    return click_imgs_outputs + click_imgs_inputs + click_imgs_states


def get_display_click_data_outputs(models: list, img_types: list):
    """
    Dynamically get outputs.

    :param models: list of available models.
    :param img_types: list of image types.
    :return: list of outputs
    """

    outputs = []
    for i in range(len(models)):
        for t in img_types:
            outputs += [Output(f'{t}_{i}', 'children'),
                        Output(f'{t}_{I}_{i}', 'title'),
                        Output(f'{t}_{I}_{i}', 'src'),
                        Output(f'{t}_{I}_{i}', 'n_clicks'),
                        Output(f'{t}_{I}_{i}', 'alt')]
        outputs += [Output(f'{BA}_{i}', 'before'), Output(f'{BA}_{i}', 'after'),
                    Output(f'{BA}_{i}', 'width'), Output(f'{BA}_{i}', 'height'),
                    Output(f'{ST}_{i}', 'data'), Output(f'{D}_{i}', 'style')]

    return outputs


def get_app(img_files: dict, models: list, img_types: list, df: pd.DataFrame,
            hist_path: Path):
    """
    Instantiates a Dash app and defines the callbacks for clicking.

    :param img_files: triplet dictionary of source, fake and target images.
    :param models: list of available models.
    :param img_types: list of image types.
    :param df: statistics dataframe.
    :param hist_path: path where histograms are stored.
    :return: Dash app
    """

    app = DashProxy(__name__, prevent_initial_callbacks=True,
                    transforms=[MultiplexerTransform()])

    @app.callback(get_click_imgs_dependencies(models, img_types))
    def display_image_hist(*n_clicks):
        data = n_clicks[len(models) * len(img_types):]
        display = []
        for id, title, clicks, model in zip(*[data[i::4] for i in range(4)]):
            f = Path(title)
            if clicks is None:
                display.append(
                    f"data:image/{f.suffix.replace('.', '')};base64,{read_image(f)}")
            else:
                if clicks % 2 == 0:
                    display.append(
                        f"data:image/{f.suffix.replace('.', '')};base64,{read_image(f)}")
                else:
                    if F in id and not f.stem.endswith(model):
                        # Keep in mind fake images generated without suffix.
                        f = Path(f"{f.stem}_{model}{f.suffix}")
                    display.append(
                        f"data:image/{f.suffix.replace('.', '')};base64,{read_image(hist_path / f.name)}")

        return display

    @app.callback(get_display_click_data_outputs(models, img_types),
                  Input('METRICS', 'clickData'))
    def display_click_data(clickData):
        outputs = []
        style = {"margin-left": "30px", 'display': 'inline-block',
                 'overflowY': 'auto'}
        try:
            model = clickData['points'][0]['x'].split(' ')[0]
            img_stem = mt_rename(Path(clickData['points'][0]['hovertext']).stem)
            other_models = sorted(models)
            other_models.remove(model)
            for i, m in enumerate([model] + other_models):
                for t in img_types:
                    if t == F:
                        file, title, img = get_file_title_img(img_files, t,
                                                              img_stem, m)
                        f_file = file
                        assets_symlink(file)
                        img = Image.open(file)
                        width = img.width
                        height = img.height
                        style['height'] = height
                    else:
                        file, title, img = get_file_title_img(img_files, t,
                                                              img_stem)
                        if t == S:
                            assets_symlink(file)
                            s_file = file
                    outputs += [str(title), str(file), img, None, m]
                metrics = df2table(df, s_file.name, m)
                outputs += [str(ASSETS / s_file.name),
                            str(ASSETS / f_file.name),
                            width, height, metrics, style]
        except TypeError:
            outputs = []
            for m in models:
                for t in img_types:
                    outputs += [None, None, None, None]
                outputs += [None, None, None, None, None, {'display': 'none'}]

        return outputs

    return app


def render(application: Dash, df: pd.DataFrame, stats: list, models: list):
    """
    Adds figure to the app.

    :param application: Dash app
    :param df: DataFrame with data.
    :param stats: Statistics to add to the app.
    :param models: list of available models.
    """

    fig = make_subplots(rows=1, cols=len(stats))
    for x in df.columns:
        col = stats.index(x.split(' ')[-1].lower()) + 1
        fig.add_trace((go.Violin(name=x, y=df[x], hovertext=df.index,
                                 points='all', pointpos=0)), row=1, col=col)

    fig.update_layout(title=M, yaxis_title="Score", clickmode='event+select',
                      height=800, autosize=True,
                      font=dict(family="Courier New, monospace", size=18,
                                color="RebeccaPurple"))
    fig.update_traces(marker_size=5)

    application.layout = html.Div([
        dcc.Graph(id='METRICS', figure=fig),
        html.Div(children=get_children(models))])


def inspect(source: str, fake: str, stats_file: str, stats: list,
            target: str = None):
    """
    Starts the inspection server.

    :param source: source images path.
    :param fake: fake generated images path.
    :param stats_file: statistic file generated by the generate_classic_metrics()
     function from metrics.py.
    :param stats: list of statistics to choose from available ones in the stats_file.
    :param target: target/expected images path.
    """

    image_types = [S, F]
    if target is not None: image_types.append(T)
    triplets = get_triplets(source, fake, target)
    print("Checking histograms ...")
    compute_hists(triplets, Path(fake) / HISTS)
    stats_df, stats = get_df(stats_file, stats)
    models = list(set([x.split(' ')[0] for x in stats_df.columns]))
    app = get_app(triplets, models, image_types, stats_df, Path(fake) / HISTS)
    render(app, stats_df, stats, models)
    app.run_server(debug=False)
