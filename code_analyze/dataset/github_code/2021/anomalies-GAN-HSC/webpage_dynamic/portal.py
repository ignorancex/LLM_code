import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, gridplot, row
import socket
import numpy as np
import os
from bokeh.colors import RGB
from bokeh.models.widgets import Select, Button, Dropdown
from bokeh.palettes import Viridis256, Plasma256, Inferno256, Magma256, all_palettes
from bokeh.models import LinearColorMapper, ColorBar, ColumnDataSource, Range1d, CustomJS, Div, \
    CDSView, \
    IndexFilter, BooleanFilter, Span, Label, BoxZoomTool, TapTool
from bokeh.transform import transform
from bokeh.transform import linear_cmap
from bokeh.models.widgets import TextInput, RadioButtonGroup, DataTable, TableColumn, AutocompleteInput, NumberFormatter
from bokeh.layouts import column
import re
from bokeh.models.annotations import Title
from bokeh.events import DoubleTap, PinchEnd, PanEnd, Reset
import sys
from pylru import lrudecorator
from bokeh.server.server import Server
import logging
import json
import h5py
from astropy.visualization import make_lupton_rgb

# The logger causes the __init__ function of astro_web to be called twice.
# Not using it now, so comment out
#logger = logging.getLogger(__name__)


path_dict = dict(
    #galaxies='data/images_h5/'
    galaxies='results/'
)

tag = 'gri_3signorm'
#tag = 'gri_cosmos'
data_type = 'galaxies'
data_path = path_dict[data_type]
results_path = 'results/'
umaps_path = 'results/embeddings'
info_fn = 'data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'

UPDATE_CIRCLE_SIZE = False


def sdss_link(SpecObjID):
    return 'http://skyserver.sdss.org/dr14/en/tools/explore/summary.aspx?sid={}&apid='.format(int(SpecObjID))


def get_umaps(self, path, embedding=None):
    #print(f"Loading embeddings {os.listdir(path)}")
    # Embeddings as made by make_umap_rgb.py
    umap_dict = {}
    for p in np.sort(os.listdir(path)):
        # Load all embeddings with the overall tag
        if tag in p:
            fn = os.path.join(path, p)
            e1, e2, colorby, idxs = np.load(fn, allow_pickle=True)
            # Should check that the idxs are the same; for now, assume they are (I checked on these!)
            emb_name = get_embedding_name(p)
            umap_dict[emb_name] = np.array([e1, e2]).T 
    return umap_dict  


def filename_params(filename, params='', extension='.npy'):
    return '{}{}{}'.format(filename, params, extension)


def load_emission_absorption_lines():
    with open('galaxies/extra/spectrum_lines.json', 'r') as f:
        lines = json.load(f)

    emission_lines = lines['emission']
    absorption_lines = lines['absorption']

    lines = []
    for l in emission_lines:
        lines.append(
            Span(location=l['wavelength'], dimension='height', line_color='teal', line_width=3, line_alpha=0.3))
        lines.append(
            Label(x=l['wavelength'], y=l['y'], y_units='screen', text=l['name'], text_color='teal'))

    for l in absorption_lines:
        lines.append(Span(location=l['wavelength'], dimension='height', line_color='orangered', line_width=3, line_alpha=0.5))
        lines.append(Label(x=l['wavelength'], y=l['y'], y_units='screen', text=l['name'], text_color='orangered'))
    return lines


@lrudecorator(5)
def load_data(data_path):
    #print("Loading data")
    #logger.info('Started Loading')
    data_fn = os.path.join(data_path, f'results_{tag}.h5')
    res = h5py.File(data_fn, 'r')
    #logger.info('Ended Loading')

    data = res["reals"]
    #idxs = [int(idx) for idx in res['idxs'][:]]
    idxs = np.array(res['idxs'])
    object_ids = np.array(res['object_ids'])
    #print("Getting object ids")
    # read object ids from file; might want other data from here eventually
    # TODO: reading this takes a good bit - slowest part of init! should save to res file
    #info_df = pd.read_csv(info_fn)
    #print("Read in info file {} with {} rows".format(info_fn, len(info_df)))
    #info_df = info_df.set_index('Unnamed: 0')
    #object_ids = [str(info_df['object_id'].loc[idx]) for idx in idxs]    
    return data, idxs, object_ids


# makes a dictionary of info_ids to indices
def reverse_galaxy_links(galaxy_links):
    gl = galaxy_links
    return {str(int(gl[v])): str(v) for v in range(len(gl))}


def load_score_data(idxs_data):
    #print("Score data")
    results_dir = os.path.join(results_path, f'results_{tag}.h5')
    res = h5py.File(results_dir, 'r')

    score_dict = {'Anomaly Score': res['anomaly_scores_norm'][:],
                  'Generator Score': res['gen_scores_norm'][:],
                  'Discriminator Score': res['disc_scores_norm'][:]}
    recon = res['reconstructed'][:]

    #print("Done with score data")
    return score_dict, recon


@lrudecorator(100)
def get_score_data(field, idxs_data):
    #print("Score data")
    results_dir = os.path.join(results_path, f'results_{tag}.h5')
    res = h5py.File(results_dir, 'r')
    N = len(res['idxs'])

    idx2resloc = {}
    for i in range(N):
        idx2resloc[res['idxs'][i]] = i

    a_score = np.empty(N)
    g_score = np.empty(N)
    d_score = np.empty(N)
    for i in range(len(idxs_data)):
        idx = idxs_data[i]
        resloc = idx2resloc[idx]
        a_score[i] = res['anomaly_scores'][resloc]
        g_score[i] = res['gen_scores'][resloc]
        d_score[i] = res['disc_scores'][resloc]

    score_dict = {'Anomaly Score': a_score,
                  'Generator Score': g_score,
                  'Discriminator Score': d_score}
    return score_dict[field]


def get_anomalies_dict(idxs_data):

    anomalies_dict = {'GAN': range(len(idxs_data))}
    return anomalies_dict


def get_numbers(u):
    return [int(n) for n in np.concatenate([s.split('-') for s in u.split('_')]) if n.isdigit()]


def get_embedding_name(u):

    if 'umap' in u:
        if 'auto' in u:
            if 'real' in u:
                name = 'UMAP on Autoencoded Real Images'
            else:
                name = 'UMAP on Autoencoded Residuals'
        elif 'reals' in u:
            name = 'UMAP on Real Image Pixels'
        elif 'residuals' in u:
            name = 'UMAP on Residual Image Pixels'
        #name = 'UMAP {}-{} A'.format(get_numbers(u)[-2], get_numbers(u)[-1])
        #name = u[:-4]#.replace('_', ' ', 1)
        #name = '_'.join((u.split('_')[:2]))
    elif 'blendextend' in u:
        name = 'Blendedness vs. log(Extendedness)'
    else:
        name = u
    return name


def get_rank(X_values):
    """
    Replace feature values with rank values
    For each feature, rank the objects according to their feature value.
    These rank values are the new features.
    :param X_values: data matrix
    :return: rank value matrix
    """
    nof_objects = X_values.shape[0]

    X_loc = np.arange(nof_objects)
    X_rank = np.zeros(nof_objects, dtype=int)

    X_argsort = X_values.argsort()
    X_rank[X_argsort] = X_loc

    return X_rank

if False:
    def get_order_from_umap(datasource, selected_objects):

        nof_objects = len(selected_objects)
        xs = np.array(datasource['xs'])[selected_objects]
        ys = np.array(datasource['ys'])[selected_objects]
        X = np.vstack([xs, ys]).T
        umap_fitter = UMAP(n_components=1, n_neighbors=min(30,nof_objects-1), min_dist=0.5)
        order = umap_fitter.fit_transform(X)
        order = order[:,0]
        order = get_rank(order)
        return order

def get_region_points(x_min, x_max, y_min, y_max, datasource):
    IGNORE_TH = -9999
    xs = np.array(datasource['xs'])
    ys = np.array(datasource['ys'])
    cd = datasource['color_data']
    nof_objects = len(cd)
    if True:
        is_in_box = np.logical_and.reduce([xs >= x_min, xs <= x_max, ys >= y_min, ys <= y_max, ys > IGNORE_TH, xs > IGNORE_TH])
    else:
        is_in_box = np.logical_and.reduce([xs >= x_min, xs <= x_max, ys >= y_min, ys <= y_max])
    return np.where(is_in_box)[0]


def get_relevant_objects_coords(datasource):
    IGNORE_TH = -999
    xs = np.array(datasource['xs'])
    ys = np.array(datasource['ys'])
    relevant_objects = np.logical_and.reduce([ys > IGNORE_TH, xs > IGNORE_TH])
    return xs[relevant_objects], ys[relevant_objects], relevant_objects


def get_decimated_region_points(x_min, x_max, y_min, y_max, datasource, DECIMATE_NUMBER):
    is_in_box_inds = get_region_points(x_min, x_max, y_min, y_max, datasource)
    print('total points before decimation', len(is_in_box_inds))
    if len(is_in_box_inds) < DECIMATE_NUMBER:
        return is_in_box_inds
    random_objects_ = np.random.choice(is_in_box_inds, DECIMATE_NUMBER, replace=False)
    random_objects = [datasource['names'][r] for r in random_objects_]
    return random_objects


def reorder_spectra_mat(spectra_matrix, indecies_to_plot, order_by):
    nof_objects = len(indecies_to_plot)

    if len(order_by) > nof_objects:
        order_by = order_by[indecies_to_plot]

    order = np.argsort(order_by)
    spectra_matrix = spectra_matrix[indecies_to_plot]

    plot_matrix = spectra_matrix[order]
    return plot_matrix


def ladder_plot_smooth(spectra_matrix, indecies_to_plot, order_by, nof_spectra, delta=1):
    plot_matrix = reorder_spectra_mat(spectra_matrix.copy(), indecies_to_plot, order_by)

    n_groups = nof_spectra
    l = int(len(indecies_to_plot) / n_groups) * n_groups
    groups = np.split(plot_matrix[:l], n_groups)
    stacks = []
    for g_idx, g in enumerate(groups):
        d = delta * g_idx
        stack = np.nanmedian(g, axis=0)
        stacks += [stack]
    return stacks


class astro_web(object):
    def __init__(self, data_type, data_path):

        self.ims_gal, self.galaxy_links, self.object_ids = load_data(data_path)
        self.sd_dict, self.recons_gal = load_score_data(self.galaxy_links)
        self.N = len(self.ims_gal)
        self.imsize = [self.ims_gal[0].shape[0], self.ims_gal[0].shape[1]]
        self.reverse_galaxy_links = reverse_galaxy_links(self.galaxy_links)
        self.umap_data = get_umaps(self, umaps_path, embedding='umap_auto')
        Plasma256.reverse()
        self.color_mapper = LinearColorMapper(palette=Plasma256, low=0, high=1, nan_color=RGB(220, 220, 220, a = 0.1))
        self.high_colormap_factor = 0.1
        self.R_DOT = 6#10
        self.DECIMATE_NUMBER = 5000
        self.UMAP_XYLIM_DELTA = 0.5
        self.umap_on_load = 1 #index of initial umap to load
        self.nof_stacks = 1
        self.n_anomalies = 51
        self.stack_by = 'x'
        self.stacks_colors = ['#fde725', '#90d743', '#35b779', '#21908d', '#31688e', '#443983', '#440154'][::-1]
        self.nrow, self.ncol = 2, 5

        self.selected_objects = ColumnDataSource(data=dict(index=[], score=[], order=[], info_id=[], object_id=[]))

        self.generate_buttons()
        self.generate_sources()
        self.generate_figures()
        self.generate_plots()

        self.register_callbacks()

    def __call__(self, doc):
        doc.add_root(
        column(
        row( self.main_title_div ),
        #row( self.info_div ),
        #row(  self.show_anomalies, self.select_anomalies_from,),
        row(
        column( self.info_div ),
        column(self.umap_figure, self.stacks_figure, row(self.prev_button, self.next_button)),
                         column(#Div(text=""" """),
                                self.select_umap, self.select_score, self.select_colormap,  #self.save_selected, #self.get_order,
                                self.search_galaxy, #self.select_galaxy, 
                                self.data_figure, self.selected_galaxies_table#, self.title_div  
                                ))))
        doc.title = 'HSC Galaxies'



    def use_log(self):
        return self.select_spectrum_plot_type.value == 'log'

    def galaxy_link(self, idx):
        return sdss_link(self.galaxy_links[idx])

    def generate_buttons(self):


        select_score_menu = [('Anomaly Score', 'Anomaly Score'),
                              ('Generator Score', 'Generator Score'),
                              ('Discriminator Score', 'Discriminator Score')]

        self.select_score = Dropdown(label="Color by:", button_type="danger", menu=select_score_menu, value='Anomaly Score')

        self.select_score_table = Select(title="Inactive", value="",
                                         options=[])

        self.update_table = Select(title="Inactive", value="",
                                         options=[])

        self.internal_reset = Select(title="Inactive", value="",
                                         options=[])

        self.select_spectrum_plot_type = Dropdown(label='Spectrum scale', button_type="default",
                                                   menu = [('log','log'),('linear','linear')], value='log')
        self.select_nof_stacks = Dropdown(label='Number of stacks', button_type="default",
                                                    menu = [(str(i), str(i)) for i in np.arange(1,11)], value='7')

        self.select_stack_by = Dropdown(label='Stack by', button_type="default",
                                                    menu = [('x', 'x'), ('y', 'y'), ('auto', 'auto')], value='x')
        self.cmap_menu = [(p,p) for p in list(all_palettes.keys())]
        #[('Viridis','Viridis256'),('Plasma','Plasma256'),('Inferno','Inferno256'), ('Magma','Magma256')]
        self.select_colormap =  Dropdown(label='Colormap', button_type="default",
                                                   menu = self.cmap_menu , value='plasma_r')

        maps = list(self.umap_data.keys())
        menu_all = [(u,u) for u in maps]

        self.select_umap = Dropdown(label='Embedding:', button_type='primary',menu=menu_all, value=maps[self.umap_on_load])


        self.anomaly_detection_algorithms = get_anomalies_dict(self.galaxy_links)
        self.select_anomalies_from = RadioButtonGroup(labels=list(self.anomaly_detection_algorithms.keys()), active=0 )
        self.show_anomalies = Button(label="Show anomalies (detected by ... )", button_type="warning")
        self.first_im_index = 8175
        self.select_galaxy = TextInput(title='Select Galaxy Index:', value=str(self.first_im_index))

        self.next_button = Button(label="Next", button_type="default")
        self.prev_button = Button(label="Previous", button_type="default")

        self.main_title_div = Div(text='<center>Anomalous Galaxy Visualization Tool</center>', style={'font-size': '300%', 'color': 'black'}, sizing_mode="stretch_width")
        info_text = """
        <p><b>What is this?</b></p>
        <p>This is an interactive tool for visualizing the results of an anomaly detection approach on a set of galaxies. Each dot is a galaxy image from the Hyper Suprime-Cam survey. The distribution is a UMAP, a 2D representation of the images. The colors show how anomalous our generative adversarial network (GAN) found each galaxy. Choose the UMAP embedding and the colormapping on the right; the default is UMAP on autencoded residuals, colored by total anomaly score.</p>
        
        <p><b>Cool! How do I use it?</b></p>
        <ul>
            <li>Use the lasso to select a region of galaxies; these will appear in the bottom squares. Scroll through them with the 'Previous' and 'Next' buttons.
            <li>The selected galaxies will also appear in the table on the right; sort them by clicking the column names. Click a row to see the image in the large viewbox.
            <li>Use the zoom tool to see more galaxies in an area; only a subset are shown (and will be selected) on the main plot.
            <li>Type a galaxy ID in the text box and hit enter to jump to that image on the plot and in the viewbox.
            <li>Double click anywhere on the plot to reset it.
        </ul>
        <p>Find the code for this project on <a href="https://github.com/kstoreyf/anomalies-GAN-HSC">github</a>. Happy weird-galaxy-finding!</p>
        <p><b>Author:</b> Kate Storey-Fisher</p>
        <p><i>Adapted from the <a href="https://toast-docs.readthedocs.io/en/latest/">SDSS galaxy portal</a> by Itamar Reis</i></p>
        """
        self.info_div = Div(text=info_text, style={'font-size': '120%', 'color': 'black'})#, sizing_mode="stretch_width")
        self.title_div = Div(text='<center>Author: Kate Storey-Fisher \n Adapted from: Itamar Reis</center>', style={'font-size': '200%', 'color': 'teal'})
        self.link_div = Div(text='<center>View galaxy in <a href="{}" target="_blank">{}</a></center>'.format(
            self.galaxy_link(int(self.select_galaxy.value)), 'SDSS object explorer'),
            style={'font-size': '120%', 'color': 'teal'})

        self.selected_galaxies_columns = [
            TableColumn(field="index", title="Index"),
            TableColumn(field="info_id", title="Info ID"),
            TableColumn(field="object_id", title="Object ID"),
            TableColumn(field="score", title="Score", formatter=NumberFormatter(format = '0.0000')),
        ]

        self.search_galaxy = TextInput(title='Select Galaxy Info ID:')


    def umap_figure_axes(self):

        embedding_name = self.select_umap.value
        if self.select_score.value == 'No color':
            self.umap_figure.title.text  = '{}'.format(embedding_name)
        else:
            self.umap_figure.title.text  = '{} - Colored by {}'.format(embedding_name , self.select_score.value)
        self.umap_figure.title.text_font_size = '17pt'

        if 'UMAP' in embedding_name:
            self.umap_figure.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
            self.umap_figure.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

            self.umap_figure.yaxis.minor_tick_line_color = None  # turn off y-axis major ticks
            self.umap_figure.yaxis.major_tick_line_color = None  # turn off y-axis minor ticks

            self.umap_figure.xaxis.major_label_text_font_size = "0pt"
            self.umap_figure.yaxis.major_label_text_font_size = "0pt"

            self.umap_figure.xaxis.axis_label_text_font_size = "0pt"
            self.umap_figure.yaxis.axis_label_text_font_size = "0pt"
        else:
            self.umap_figure.xaxis.major_tick_line_color = 'black'  # turn off x-axis major ticks
            self.umap_figure.xaxis.minor_tick_line_color = 'black'  # turn off x-axis minor ticks

            self.umap_figure.yaxis.minor_tick_line_color = 'black'  # turn off y-axis major ticks
            self.umap_figure.yaxis.major_tick_line_color = 'black'  # turn off y-axis minor ticks

            self.umap_figure.xaxis.major_label_text_font_size = "15pt"
            self.umap_figure.yaxis.major_label_text_font_size = "15pt"
            self.umap_figure.xaxis.axis_label_text_font_size = "15pt"
            self.umap_figure.yaxis.axis_label_text_font_size = "15pt"

            # self.umap_figure.yaxis.axis_label = 'Log(OIII / Hb)'
            # if 'NII' in embedding_name:
            #     self.umap_figure.xaxis.axis_label = 'Log(NII / Ha)'
            # elif 'OI' in embedding_name:
            #     self.umap_figure.xaxis.axis_label = 'Log(OI / Ha)'
            # elif 'SII' in embedding_name:
            #     self.umap_figure.xaxis.axis_label = 'Log(SII / Ha)'

        return


    def generate_figures(self):

        umap_plot_width = 800
        column_width = 350
        #taptool = TapTool(callback=self.select_galaxy_callback)
        self.umap_figure = figure(tools='lasso_select,tap,box_zoom,save,reset',
                                  plot_width=umap_plot_width,
                                  plot_height=600,
                                  #title='UMAP',
                                  toolbar_location="above", output_backend='webgl', )  # x_range=(-10, 10),
        #self.umap_figure.add_tools(taptool)

        # y_range=(-10, 10))
        #self.umap_figure.toolbar.active_scroll = 'auto'
        self.data_figure = figure(tools="box_zoom,save,reset", plot_width=column_width,
                                      plot_height=column_width,
                                      toolbar_location="above", output_backend='webgl',
                                      x_range=(0,96), y_range=(0,96))

        title_height = 20
        buffer = 10*self.ncol
        collage_im_width = int((umap_plot_width-buffer)/self.ncol)
        self.stacks_figures = []
        for _ in range(self.nrow*self.ncol):
            sfig = figure(tools="box_zoom,save,reset", plot_width=collage_im_width, plot_height=collage_im_width+title_height, 
            toolbar_location="above", output_backend='webgl', x_range=(0,96), y_range=(0,96))
            self.stacks_figures.append(sfig)

        self.stacks_figure = gridplot(self.stacks_figures, ncols=self.ncol)


        self.umap_colorbar = ColorBar(color_mapper=self.color_mapper, location=(0, 0), major_label_text_font_size='15pt', label_standoff=13)
        self.umap_figure.add_layout(self.umap_colorbar, 'right')

        self.umap_figure_axes()

        # TODO: make this the index
        t = Title()
        #info_id = self.select_galaxy.value
        ind = self.select_galaxy.value
        info_id = int(self.galaxy_links[int(ind)])
        t.text = '{}'.format(info_id)
        self.data_figure.title = t

        self.remove_ticks_and_labels(self.data_figure)

        for i in range(len(self.stacks_figures)):
            self.remove_ticks_and_labels(self.stacks_figures[i])
            t = Title()
            t.text = ' '
            self.stacks_figures[i].title = t

        self.selected_galaxies_table = DataTable(source=self.selected_galaxies_source,
                                                 columns=self.selected_galaxies_columns,
                                                 width=column_width,
                                                 height=200,
                                                 scroll_to_selection=False)


    def remove_ticks_and_labels(self, figure):
        figure.xaxis.major_label_text_font_size = "0pt"
        figure.xaxis.axis_label_text_font_size = "0pt"
        figure.yaxis.major_label_text_font_size = "0pt"

        figure.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
        figure.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

        figure.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        figure.yaxis.minor_tick_line_color = None  # turn off y-axis minor tick


    def generate_sources(self):

        sd = self.sd_dict[self.select_score.value]
        self.set_colormap(sd)
        print('generate sources')
        self.umap_source = ColumnDataSource(
            data=dict(xs=self.umap_data[self.select_umap.value][:, 0],
                      ys=self.umap_data[self.select_umap.value][:, 1],
                      color_data=sd[:],
                      names=list(np.arange(len(sd))),
                      radius=[self.R_DOT] * len(sd)),
                      )

        self.xlim = (np.min(self.umap_source.data['xs']) - self.UMAP_XYLIM_DELTA, np.max(self.umap_source.data['xs']) + self.UMAP_XYLIM_DELTA)
        self.ylim = (np.min(self.umap_source.data['ys']) - self.UMAP_XYLIM_DELTA, np.max(self.umap_source.data['ys']) + self.UMAP_XYLIM_DELTA)

        self.xlim_all = {}
        self.ylim_all = {}

        for umap in list(self.umap_data.keys()):
            temp_umap_source = ColumnDataSource(
                data=dict(xs=self.umap_data[umap][:, 0],
                          ys=self.umap_data[umap][:, 1],
                          color_data=sd[:],
                          names=list(np.arange(len(sd))),
                          radius=[self.R_DOT] * len(sd)),
                          )
            rxs, rys, _ = get_relevant_objects_coords(temp_umap_source.data)
            temp_xlim = (np.min(rxs) - self.UMAP_XYLIM_DELTA, np.max(rxs) + self.UMAP_XYLIM_DELTA)
            temp_ylim = (np.min(rys) - self.UMAP_XYLIM_DELTA, np.max(rys) + self.UMAP_XYLIM_DELTA)
            self.xlim_all[umap] = temp_xlim
            self.ylim_all[umap] = temp_ylim

        points = get_decimated_region_points(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1],
                                             self.umap_source.data, self.DECIMATE_NUMBER)

        self.umap_source_view = ColumnDataSource(
            data=dict(xs=self.umap_data[self.select_umap.value][points, 0],
                      ys=self.umap_data[self.select_umap.value][points, 1],
                      color_data=sd[points],
                      names=list(points),
                      radius=[self.R_DOT] * len(points)),
                    )
        self.points = np.array(points)
        self.umap_view = CDSView(source=self.umap_source_view)

        im = process_image(self.ims_gal[self.first_im_index])
        xsize, ysize = self.imsize
        self.data_source = ColumnDataSource(
            data = {'image':[im], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
        )

        xsize, ysize = self.imsize
        self.stacks_sources = []
        count = 0
        ncollage = self.nrow*self.ncol
        im_empty = self.get_im_empty()
        while count < ncollage:
            source = ColumnDataSource(
                data = {'image':[im_empty], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
            )
            self.stacks_sources.append(source)
            count += 1

        self.stacks_source = ColumnDataSource(
            data = {'image':[im], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
        )

        self.selected_galaxies_source = ColumnDataSource(dict(
            index=[],
            score=[],
            order=[],
            info_id=[],
            object_id=[]
        ))

        self.search_galaxy_source = ColumnDataSource(dict(
            xs=[self.umap_data[self.select_umap.value][0, 0]],
            ys=[self.umap_data[self.select_umap.value][0, 1]],
        ))


    def get_im_empty(self):
        return np.zeros((self.imsize[0], self.imsize[1], 4)).astype(np.uint8)


    def generate_plots(self):
        print("gen plots")
        self.umap_scatter = self.umap_figure.scatter('xs', 'ys', source=self.umap_source_view,
                                                     color=transform('color_data', self.color_mapper),
                                                     nonselection_fill_color = 'moccasin',
                                                     nonselection_line_color = 'moccasin',
                                                     nonselection_alpha = 1,
                                                     nonselection_line_alpha = 0,
                                                     alpha=0.5,
                                                     line_color=None,
                                                     size='radius',
                                                     view=self.umap_view)
        self.data_image = self.data_figure.image_rgba('image', 'x', 'y', 'dw', 'dh', source=self.data_source)

        self.spectrum_stacks = []
        for i in range(self.nrow*self.ncol):
            spec_stack = self.stacks_figures[i].image_rgba('image', 'x', 'y', 'dw', 'dh', source=self.stacks_sources[i])
            self.spectrum_stacks.append(spec_stack)

        self.stacks_figure = gridplot(self.stacks_figures, ncols=self.ncol)

        self.umap_search_galaxy = self.umap_figure.circle('xs', 'ys', source=self.search_galaxy_source, alpha=0.5,
                                                          color='tomato', size=self.R_DOT*4, line_color="black", line_width=2)

        LINE_ARGS = dict(color="#3A5785", line_color=None)


    def update_umap_figure(self):
        print("update umap figure")
        def callback(attrname, old, new):

            sd = self.sd_dict[self.select_score.value]
            self.set_colormap(sd)

            self.umap_source = ColumnDataSource(
                data=dict(xs=self.umap_data[self.select_umap.value][:, 0],
                          ys=self.umap_data[self.select_umap.value][:, 1],
                          color_data=sd,
                          radius=[self.R_DOT] * len(sd),
                          names=list(np.arange(len(sd))),
                          ))

            self.umap_figure_axes()
            if True:

                self.xlim = self.xlim_all[self.select_umap.value]
                self.ylim = self.ylim_all[self.select_umap.value]

                self.umap_figure.x_range.start = self.xlim[0]
                self.umap_figure.x_range.end = self.xlim[1]
                self.umap_figure.y_range.start = self.ylim[0]
                self.umap_figure.y_range.end = self.ylim[1]

            background_objects = get_decimated_region_points(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], self.umap_source.data, self.DECIMATE_NUMBER)

            selected_objects = self.selected_objects.data['index']
            self.get_new_view_keep_selected(background_objects, selected_objects)

            self.select_score_table.value = self.select_score.value

        return callback

    def set_colormap(self, sd):
        mx = np.nanmax(sd)
        mn = np.nanmin(sd)
        if mn == mx:
            high = mx + 1
            low = mn - 1
        else:
            high = mx + (mx - mn)*self.high_colormap_factor
            low = mn
            # set max of colormap to Nth largets val, to deal with outliers
            nth = 100
            if len(sd)>nth:
                nmx = np.sort(sd)[-nth]
                if nmx*1.2 < mx:
                    high = nmx

        self.color_mapper.high = high
        self.color_mapper.low = low

        return


    def select_stack_by_callback(self):
        def callback(attrname, old, new):
            self.stack_by = self.select_stack_by.value
        return callback


    def select_nof_stacks_callback(self):
        def callback(attrname, old, new):
            self.nof_stacks = int(self.select_nof_stacks.value)
        return callback


    def update_color(self):
        def callback(attrname, old, new):

            self.umap_figure_axes()
            #sd = get_score_data(self.select_score.value, self.galaxy_links)
            sd = self.sd_dict[self.select_score.value]

            self.set_colormap(sd)

            self.umap_source = ColumnDataSource(
                data=dict(xs=self.umap_data[self.select_umap.value][:, 0],
                          ys=self.umap_data[self.select_umap.value][:, 1],
                          color_data=sd,
                          radius=[self.R_DOT] * len(sd),
                          names=list(np.arange(len(sd))),
                        ))

            selected_objects = self.selected_objects.data['index']
            background_objects = self.umap_source_view.data['names']

            if (self.select_score.value == 'Order') and len(selected_objects) > 0:
                #custom_sd = sd.copy()
                custom_sd = np.ones(sd.shape)*np.nan
                selected_inds = np.array([int(s) for s in selected_objects])
                order = np.array([float(o) for o in self.selected_objects.data['order']])
                custom_sd[selected_inds] = order
                #print(np.max(custom_sd))
                self.set_colormap(custom_sd)
                self.get_new_view_keep_selected(background_objects, selected_objects, custom_sd = custom_sd)
            else:
                self.get_new_view_keep_selected(background_objects, selected_objects)
                self.select_score_table.value = self.select_score.value

        return callback


    def update_spectrum(self):
        print("update spectrum")
        #TODO: BE CAREFUL W INDEX VS ID
        index = self.select_galaxy.value
        specobjid = int(self.galaxy_links[int(index)])

        im = process_image(self.ims_gal[int(index)])
        xsize, ysize = self.imsize
        self.data_source = ColumnDataSource(
            data = {'image':[im], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}  
        )
        
        self.data_image.data_source.data = self.data_source.data

        self.data_figure.title.text = '{}'.format(int(specobjid))
        self.link_div.text='<center>View galaxy in <a href="{}" target="_blank">{}</a></center>'.format(
            self.galaxy_link(int(self.select_galaxy.value)), 'SDSS object explorer')

        return


    def register_reset_on_double_tap_event(self, obj):
        obj.js_on_event(DoubleTap, CustomJS(args=dict(p=obj), code="""
            p.reset.emit()
            console.debug(p.x_range.start)
            """))


    def reset_stack_index(self):
        self.stack_index = 0


    def next_stack_index(self):
        selected_objects = self.selected_objects.data['index']
        nof_selected = len(selected_objects)
        new_index = self.stack_index + self.ncol*self.nrow
        if new_index < nof_selected:
            self.stack_index = new_index
            self.stacks_callback()


    def prev_stack_index(self):
        if self.stack_index != 0:
            self.stack_index -= self.ncol*self.nrow
            self.stacks_callback()


    # TODO: can this be simplified?
    def select_stacks_callback(self):
        def callback(event):
            self.stacks_callback()
        return callback


    def stacks_callback(self):
        
        print('select_stacks_callback')
        selected_objects = self.selected_objects.data['index']
        selected_inds = np.array([int(s) for s in selected_objects])
        nof_selected = selected_inds.size
        inds_visible = selected_inds[self.stack_index:self.stack_index+self.nrow*self.ncol]

        xsize, ysize = self.imsize
        self.stacks_sources = []
        count = 0
        im_empty = self.get_im_empty()
        while count < self.nrow*self.ncol:
            if count < len(inds_visible):
                ind = inds_visible[count]
                im = process_image(self.ims_gal[ind])
                info_id = self.galaxy_links[ind]
                new_title = '{}'.format(int(info_id))
                #im = self.ims_gal[ind]
            else:
                im = im_empty
                new_title = ' '

            source = ColumnDataSource(
                data = {'image':[im], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
            )
            self.stacks_sources.append(source)
            self.stacks_figures[count].title.text = new_title
            self.spectrum_stacks[count].data_source.data = self.stacks_sources[count].data

            count += 1


    def get_stacks_callback(self):

        print('get_stacks_callback')
        selected_objects = self.selected_objects.data['index']
        selected_inds = np.array([int(s) for s in selected_objects])

        nof_selected = selected_inds.size

        print("Update stacks")
        xsize, ysize = self.imsize
        self.stacks_sources = []
        nrow, ncol = 2, 3
        count = 0
        while count < nrow*ncol:
            if count >= nof_selected:
                self.stacks_sources.append(None)
                count += 1
                continue 
            ind = selected_inds[count]
            im = process_image(self.ims_gal[ind])
            source = ColumnDataSource(
                data = {'image':[im], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
            )
            self.stacks_sources.append(source)
            count += 1

        xsize, ysize = self.imsize
        im = process_image(self.ims_gal[0])
        self.stacks_source = ColumnDataSource(
            data = {'image':[im], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
        )

        for i in range(nof_selected):
            self.spectrum_stacks[i].data_source.data = self.stacks_sources[i].data

        return


    def show_anomalies_callback(self):

        print('show_anomalies_callback')
        selected_anomaly_method = list(self.anomaly_detection_algorithms.keys())[self.select_anomalies_from.active]
        selected_objects = self.anomaly_detection_algorithms[selected_anomaly_method]
        selected_objects = [int(s) for s in selected_objects]

        backgroud_objects = self.umap_source_view.data['names']
        self.get_new_view_keep_selected(backgroud_objects, selected_objects)

        return


    def select_galaxy_callback(self):
        print('select gal')
        def callback(attr, old, new):
            index = self.select_galaxy.value
            index_str = str(index)
            if ',' in index_str:
                print('list input')
                selected_objects = index_str.replace(' ','').split(',')
                selected_objects = [int(s) for s in selected_objects]

                backgroud_objects = self.umap_source_view.data['names']
                self.get_new_view_keep_selected(backgroud_objects, selected_objects)

                return
            print('galaxy callback')
            specobjid = str(self.search_galaxy.value)
            new_specobjid = str(int(self.galaxy_links[int(index)]))
            #new_specobjid = str(
            #logger.debug(type(specobjid), specobjid, type(new_specobjid), new_specobjid)
            #print(specobjid, new_specobjid)
            if specobjid != new_specobjid:
                print('not equal')
                self.search_galaxy.value = new_specobjid
            else:
                print('Update spec from select')
                self.update_spectrum()

        return callback


    def search_galaxy_callback(self):
        print('search gal')
        def callback(attr, old, new):
            #logger.debug(self.search_galaxy.value)
            specobjid_str = str(self.search_galaxy.value)
            if ',' in specobjid_str:
                print('list input')
                selected_objects_ids = specobjid_str.replace(' ','').split(',')
                index_str = str(self.reverse_galaxy_links[selected_objects_ids[0]])
                for idx, specobjid in enumerate(selected_objects_ids[1:]):
                    index_str = '{}, {}'.format(index_str, str(self.reverse_galaxy_links[specobjid]))
                self.select_galaxy.value = index_str
                return

            if specobjid_str in self.reverse_galaxy_links:
                print('search galaxy')
                index = str(self.select_galaxy.value)
                new_index = str(self.reverse_galaxy_links[specobjid_str])
                self.update_search_circle(new_index)
                print('search galaxy - updated circle')
                #logger.debug(type(index), index, type(new_index), new_index)
                if index != new_index:
                    self.select_galaxy.value = new_index
                else:
                    print('Update spec from search')
                    self.update_spectrum()

        return callback


    def update_search_circle(self, index):
        print("update search circle")
        self.search_galaxy_source.data = dict(
            xs=[self.umap_source.data['xs'][int(index)]],
            ys=[self.umap_source.data['ys'][int(index)]],
        )
        return


    def get_new_view_keep_selected(self, background_objects, selected_objects_, custom_sd = None):

        print('get_new_view_keep_selected')
        _, _, is_relevant = get_relevant_objects_coords(self.umap_source.data)
        selected_objects = [s for s in selected_objects_ if is_relevant[int(s)]]
        selected_objects = np.array(selected_objects)
        background_objects = np.array(background_objects)

        nof_selected_objects = selected_objects.size

        max_nof_selected_objects = int(self.DECIMATE_NUMBER)/2
        if nof_selected_objects > max_nof_selected_objects:
            nof_selected_objects = max_nof_selected_objects
            new_objects = selected_objects[:nof_selected_objects]
        else:
            new_objects = np.concatenate([selected_objects, background_objects])
            new_objects, order = np.unique(new_objects, return_index=True)
            new_objects = new_objects[np.argsort(order)]
            new_objects = new_objects[:self.DECIMATE_NUMBER]

        new_objects = new_objects.astype(int)
        if custom_sd is None:
            sd = self.sd_dict[self.select_score.value]
        else:
            sd = custom_sd

        self.umap_source_view = ColumnDataSource(
                data=dict(xs=self.umap_data[self.select_umap.value][new_objects, 0],
                          ys=self.umap_data[self.select_umap.value][new_objects, 1],
                          color_data=sd[new_objects],
                          names=list(new_objects),
                          radius=[self.R_DOT] * len(new_objects),
                        ))
        self.points = np.array(new_objects)
        self.umap_scatter.data_source.data = self.umap_source_view.data

        if nof_selected_objects > 0:
            order = np.array([float(o) for o in self.selected_objects.data['order']])
            self.selected_objects.data = dict(
                index=list(selected_objects), 
                score=[-999999 if np.isnan(sd[s]) else sd[s] for s in selected_objects],
                order=list(order), 
                info_id=[self.galaxy_links[s] for s in selected_objects],
                object_id=[self.object_ids[s] for s in selected_objects]
            )
            self.update_table.value = str(np.random.rand())
        elif len(selected_objects_) > 0:
            self.selected_objects = ColumnDataSource(data=dict(index=[], score=[], order=[], info_id=[], object_id=[]))
            self.update_table.value = str(np.random.rand())
            self.internal_reset.value = str(np.random.rand())
        else:
            self.update_table.value = str(np.random.rand())

        # Update circle
        index = self.select_galaxy.value

        if (index in set(background_objects)) :
            pass
        else:
            if len(selected_objects) > 0:
                if (index in set(selected_objects)):
                    pass
                else:
                    index = str(selected_objects[0])
                    self.select_galaxy.value = index
            else:
                index = str(background_objects[0])
                self.select_galaxy.value = index

        self.update_search_circle(index)

        return


    def update_umap_filter_event(self):
        print("update umap")
        def callback(event):
            print('update_umap_filter_event')

            ux = self.umap_source_view.data['xs']
            uy = self.umap_source_view.data['ys']

            px_start = self.umap_figure.x_range.start
            px_end = self.umap_figure.x_range.end
            py_start = self.umap_figure.y_range.start
            py_end = self.umap_figure.y_range.end

            if ( (px_start > np.min(ux) ) or
                 (px_end   < np.max(ux) ) or
                 (py_start > np.min(uy) ) or
                 (py_end   < np.max(uy) )   ):

                background_objects = get_decimated_region_points(
                    self.umap_figure.x_range.start,
                    self.umap_figure.x_range.end,
                    self.umap_figure.y_range.start,
                    self.umap_figure.y_range.end,
                    self.umap_source.data,
                    self.DECIMATE_NUMBER)

                print("umap selected:")
                selected_objects = self.selected_objects.data['index']
                self.get_new_view_keep_selected(background_objects, selected_objects)
            else:
                print('Pan event did not require doing anything')
                pass

        return callback

    def update_umap_filter_reset(self):
        def callback(event):
            print('reset double tap')

            background_objects = get_decimated_region_points(
                self.xlim[0],
                self.xlim[1],
                self.ylim[0],
                self.ylim[1],
                self.umap_source.data,
                self.DECIMATE_NUMBER)

            selected_objects = self.selected_objects.data['index']
            self.get_new_view_keep_selected(background_objects, selected_objects)

            self.umap_figure.x_range.start = self.xlim[0]

            self.umap_figure.x_range.end = self.xlim[1]

            self.umap_figure.y_range.start = self.ylim[0]

            self.umap_figure.y_range.end = self.ylim[1]

            xsize, ysize = self.imsize
            self.stacks_sources = []
            count = 0
            ncollage = self.nrow*self.ncol
            im_empty = self.get_im_empty()
            while count < ncollage:
                source = ColumnDataSource(
                    data = {'image':[im_empty], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
                )
                self.stacks_sources.append(source)
                self.stacks_figures[count].title.text = ' '
                self.spectrum_stacks[count].data_source.data = self.stacks_sources[count].data

                count += 1

        return callback


    def select_colormap_callback(self):
        def callback(attr, old, new):
            print("colormap callback")
            if 'Plasma' in self.select_colormap.value:
                self.color_mapper.palette = Plasma256
                self.umap_scatter.nonselection_glyph.fill_color = 'lightgray'
                #self.umap_scatter.nonselection_glyph.fill_alpha = 0.2
            elif 'Inferno' in self.select_colormap.value:
                self.color_mapper.palette = Inferno256
                self.umap_scatter.nonselection_glyph.fill_color = 'lightgray'
                #self.umap_scatter.nonselection_glyph.fill_alpha = 0.2
            elif 'Viridis' in self.select_colormap.value:
                self.color_mapper.palette = Viridis256
                self.umap_scatter.nonselection_glyph.fill_color = 'moccasin'
                #self.umap_scatter.nonselection_glyph.fill_alpha = 0.2
            elif 'Magma' in self.select_colormap.value:
                self.color_mapper.palette = Magma256
                self.umap_scatter.nonselection_glyph.fill_color = 'lightgray'
                #self.umap_scatter.nonselection_glyph.fill_alpha = 0.2
            else:
                p_dict = all_palettes[self.select_colormap.value]
                numbers = list(p_dict.keys())
                n = np.max(numbers)
                p = p_dict[n]
                self.color_mapper.palette = p
                self.umap_scatter.nonselection_glyph.fill_color = 'lightgray'

        return callback


    def register_callbacks(self):
        self.register_reset_on_double_tap_event(self.umap_figure)
        self.register_reset_on_double_tap_event(self.data_figure)
        for i in range(len(self.stacks_figures)):
            self.register_reset_on_double_tap_event(self.stacks_figures[i])


        self.show_anomalies.on_click(self.show_anomalies_callback)
        self.next_button.on_click(self.next_stack_index)
        self.prev_button.on_click(self.prev_stack_index)        

        self.select_score.on_change('value', self.update_color())
        self.select_umap.on_change('value', self.update_umap_figure())

        self.search_galaxy.on_change('value', self.search_galaxy_callback())
        self.select_galaxy.on_change('value', self.select_galaxy_callback())
        self.select_spectrum_plot_type.on_change('value', self.select_galaxy_callback())
        self.select_nof_stacks.on_change('value', self.select_nof_stacks_callback())
        self.select_stack_by.on_change('value', self.select_stack_by_callback())

        self.select_colormap.on_change('value', self.select_colormap_callback())

        self.umap_figure.on_event(PanEnd, self.reset_stack_index)
        self.umap_figure.on_event(PanEnd, self.select_stacks_callback())

        self.selected_galaxies_source.selected.js_on_change('indices',
                                                            CustomJS(
                                                                args=dict(s1=self.selected_galaxies_source,
                                                                          sg=self.select_galaxy),
                                                                code="""
                                                                    var inds = s1.attributes.selected['1d'].indices
                                                                    if (inds.length > 0) {
                                                                        sg.value = String(s1.data.index[inds[0]]);
                                                                    }
                                                                    console.log(s1);
                                                                    console.log('selected_galaxies_source_js')
                                                                    """))

        self.umap_source_view.selected.js_on_change('indices', CustomJS(
            args=dict(s1=self.umap_source_view, s2=self.selected_galaxies_source, s3=self.selected_objects, s4=self.galaxy_links, s5=self.object_ids), code="""
                var inds = s1.attributes.selected['1d'].indices
                var d1 = s1.data;
                var d2 = s2.data;
                var d3 = s3.data;
                d2.index = []
                d2.score = []
                d2.order = []
                d2.info_id = []
                d2.object_id = []
                d3.index = []
                d3.score = []
                d3.order = []
                d3.info_id = []
                d3.object_id = []
                for (var i = 0; i < inds.length; i++) {
                    d2.index.push(d1['names'][inds[i]])
                    d2.score.push(d1['color_data'][inds[i]])
                    d2.order.push(0.0)
                    d2.info_id.push(s4[d1['names'][inds[i]]])
                    d2.object_id.push(s5[d1['names'][inds[i]]])
                    d3.index.push(d1['names'][inds[i]])
                    d3.score.push(d1['color_data'][inds[i]])
                    d3.order.push(0.0)
                    d3.info_id.push(s4[d1['names'][inds[i]]])
                    d3.object_id.push(s5[d1['names'][inds[i]]])
                }
                console.log('umap_source_view_js')
                s2.change.emit();
                s3.data = d3;
                s3.change.emit();
            """))
        self.select_score_table.js_on_change('value', CustomJS(
            args=dict(s1=self.umap_source_view, s2=self.selected_galaxies_source, s4=self.galaxy_links, s5=self.object_ids), code="""
                    var inds = s1.attributes.selected['1d'].indices
                    var d1 = s1.data;
                    var d2 = s2.data;
                    d2.index = []
                    d2.score = []
                    d2.order = []
                    d2.info_id = []
                    d2.object_id = []
                    for (var i = 0; i < inds.length; i++) {
                        d2.index.push(d1['names'][inds[i]])
                        d2.score.push(d1['color_data'][inds[i]])
                        d2.order.push(0.0)
                        d2.info_id.push(s4[d1['names'][inds[i]]])
                        d2.object_id.push(s5[d1['names'][inds[i]]])
                    }
                    console.log(d2.index)
                    console.log('select_score_table_js')
                    s2.change.emit();
                """))

        ### !!!NOTE!!! careful with semicolons here, all vars except last need one!
        self.update_table.js_on_change('value', CustomJS(
            args=dict(s1=self.umap_source_view, s2=self.selected_galaxies_source, s3=self.selected_objects), code="""
                    var d2 = s2.data;
                    console.log(s3.attributes.data.index);
                    var selected_objects = s3.attributes.data.index;
                    var score = s3.attributes.data.score;
                    var order = s3.attributes.data.order;
                    var info_id = s3.attributes.data.info_id;
                    var object_id = s3.attributes.data.object_id
                    d2.index = []
                    d2.score = []
                    d2.order = []
                    d2.info_id = []
                    d2.object_id = []
                    var inds = []
                    console.log(selected_objects);
                    for (var i = 0; i < selected_objects.length; i++) {
                        inds.push(i)
                        d2.index.push(selected_objects[i])
                        d2.score.push(score[i])
                        d2.order.push(order[i])
                        d2.info_id.push(info_id[i])
                        d2.object_id.push(object_id[i])
                    }
                    s1.attributes.selected['1d'].indices = inds
                    s1.attributes.selected.attributes.indices = inds
                    console.log(s1)
                    console.log('update_table_js')
                    s2.change.emit();
                    s1.change.emit();
                """))

        self.internal_reset.js_on_change('value', CustomJS(
            args=dict(p=self.umap_figure), code="""
                        p.reset.emit()
                        """))

        self.umap_figure.on_event(PanEnd, self.update_umap_filter_event())
        self.umap_figure.on_event(Reset, self.update_umap_filter_reset())


def get_astro_session(doc):
    sess = astro_web(data_type, data_path)
    return sess(doc)


# This is necessary for displaying image properly
def process_image(im):
    alpha = np.full((im.shape[0], im.shape[1], 1), 255)
    im = np.concatenate((im, alpha), axis=2)
    return im.astype(np.uint8)


# a = astro_web(data_type, data_path)
if __name__ == '__main__':
    lh = 5006
    websockets = ['104.248.124.15','104.248.124.15:{}'.format(lh), f'localhost:{lh}', 'weirdgalaxi.es'] #my digitalocean server
    print('Opening Bokeh application on {}'.format(websockets))
    server = Server({'/': get_astro_session}, num_procs=1, #0 autodetects # cores
                    allow_websocket_origin=websockets, show=False,
                    port=lh, prefix='/')

    #print('Opening Bokeh application on http://localhost:{}/'.format(lh))
    #websockets = ['localhost:{}'.format(lh)]
    #server = Server({'/galaxies': get_astro_session}, num_procs=0,
    #                allow_websocket_origin=websockets, show=False)
    server.start() # this line doesn't seem to do anything, but also doesn't hurt...
    # KSF: found this code here, but not sure what it's doing https://riptutorial.com/bokeh/example/29716/local-bokeh-server-with-console-entry-point
    # server.io_loop.add_callback(server.show, "/") # this was commented out
    server.io_loop.start()
