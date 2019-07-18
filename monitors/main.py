### TODO ###
# History length option in config for variables
# Supplimentary Plots (open another file(s) and plot it. ie. profiles, contours...)
# Pushover notifications
# Grayscale mode
# Night mode
# Mobile Mode ?
############

## Plotting
from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, Range1d, Select, CustomJS
from bokeh.models.widgets import CheckboxButtonGroup, Panel, Tabs
from bokeh.palettes import Category10_10 as palette, Dark2_8 as palette_2
from bokeh.core.enums import LegendLocation
import itertools 

## Data Parsing
from fluent_tui.utils import parse_outfile, parse_residuals, fluent_connect
import pandas as pd
import numpy as np
import re

## Config Parsing
import configparser
import ast
import os
from fractions import Fraction
from shutil import copyfile

def config_parse(cwd):
    if 'monitors.ini' not in os.listdir(cwd):
        copyfile('monitors.ini', cwd+'\\monitors.ini')
        os.startfile(cwd+'\\monitors.ini')
        input('Press any key to continue when finished editing the config file');

    cfg = configparser.ConfigParser()
    cfg.read(cwd+'\\monitors.ini')
    try:
        mon_fname = cwd+'\\'+cfg.get('SETUP','mon_fname')
    except:
        mon_fname = cwd+'\\data.out'
    try:
        tra_fname = cwd+'\\'+cfg.get('SETUP','tra_fname')
    except:
        tra_fname = cwd+'\\transcript.out'
    try:
        img_folder = cwd+'\\'+cfg.get('SETUP','img_folder')
        images = ast.literal_eval(cfg.get('SETUP','images'))
        if isinstance(images,str): images=[images]
    except:
        img_folder = None
        images = None
    variables = ast.literal_eval(cfg.get('SETUP','variables'))
    try:
        var_ylims = ast.literal_eval(cfg['SETUP']['var_ylims'])
    except:
        var_ylims = None
    try:
        resid_len = cfg.getint('SETUP','resid_len')
    except:
        resid_len = None
    try:
        width,height = cfg.getint('GRIDSPEC','width'),cfg.getint('GRIDSPEC','height')
    except:
        width,height = 1440,720
    sizes = ast.literal_eval(re.sub(r'([\d.]+)/([\d.]+)', r'(\1,\2)',cfg.get('GRIDSPEC','sizes')))
    sizes = np.array([[Fraction(*x) if isinstance(x,tuple) else x for x in s] for s in sizes])
    sizes = (sizes*np.array([width,height])).astype(int)
    try:
        polling_int = cfg.getint('SETUP','polling_int')
    except:
        polling_int = 500
    return dict(
        mon_fname=mon_fname,
        tra_fname=tra_fname,
        img_folder=img_folder,
        variables=variables,
        images=images,
        var_ylims=var_ylims,
        resid_len=resid_len,
        sizes=sizes,
        width=width,
        height=height,
        polling_int=polling_int,
    )

def var_name_cleaner(vars):
    output = []
    for col in vars:
        if isinstance(col,str):
            output.append(col.replace('-','_').replace(' ','_'))
        else:
            output.append([var.replace('-','_').replace(' ','_') for var in col])
    return output
    
def get_res(tra_fname,resid_len=500):
    df = parse_residuals(tra_fname)
    if df is None:
        return
    df = df[[c for c in df.columns if c not in ['time','iter-left',
        'converged','relative-total-time','relative-time-step']]]
    df.columns = var_name_cleaner(df.columns)
    if resid_len is None:
        return df
    else:
        return df.iloc[-resid_len::]

def get_vars(mon_fname):
    df = parse_outfile(cfg['mon_fname'])
    if df is None:
        return
    df.columns = var_name_cleaner(df.columns)
    return df

def get_prog(tra_fname):
    steps = []
    lines = []
    converged = False
    with open(tra_fname,'r') as file:
        for line in file:
            lines.append(line)
            if len(lines) > 50:
                lines.pop(0);
            if 'more time step' in line.lower():
                steps.append(int(line.split(' ')[0]))
        if len(steps) == 0:
            df = parse_residuals(tra_fname) 
            steps = df['iter-left'].values
        if 'solution is converged' in ' '.join(lines):
            if 'time steps' not in ' '.join(lines):
                converged = True
            
    steps = np.array(steps)
    try:
        ind = np.where(np.diff(steps)>0)[0][-1]+1
    except IndexError:
        if len(steps)>0:
            ind = 0
        else:
            raise
    total_steps = steps[ind]+1
    steps_left = steps[-1]-1
    steps_complete = total_steps-steps_left
    text = '{0:>2d}%'.format(int(100*steps_complete//total_steps))
    if converged:
        return dict(steps_left=[0],
            steps_complete=[1], text=['CONVERGED'])
    else:
        return dict(steps_left=[steps_left/total_steps],
            steps_complete=[steps_complete/total_steps], text=[text])
        
def get_img_data(cfg):
    try:
        static = __file__.split('\\')[-2]+'/static/'
        img_folder = cfg['img_folder']
        img_basenames = cfg['images']
        images = [[fname for fname in os.listdir(img_folder) if name in fname].pop() for name in img_basenames]
        # Clean Static Folder
        [os.remove(static+fname) for fname in os.listdir(static)]
        d = {}
        for fname,ibn in zip(images,img_basenames):
            copyfile(img_folder+'\\'+fname, static+fname)
            if fname[-4::] != '.png':
                from PIL import Image
                with Image.open(static+fname) as tmp_img:
                    tmp_img.save(static+fname[0:-4]+'.png')
                os.remove(static+fname)
                fname = fname[0:-4]+'.png'
                d.update({ibn:[static+fname]})
        d.update(dict(x=[0], y=[1], w=[1], h=[1]))
        return d
    except:
        return dict(x=[],y=[],w=[],h=[])
    
def monitors(doc):
    ## Streaming Update Function
    def update():
        global cfg
        global s1
        global s2
        global s3
        
        df = get_vars(cfg['mon_fname'])
        s1.stream(df.to_dict(orient='list'),rollover=len(df))
        
        df = get_res(cfg['tra_fname'],cfg['resid_len'])
        if cfg['resid_len'] is None:
            s2.stream(df.to_dict(orient='list'),rollover=len(df))
        else:
            s2.stream(df.to_dict(orient='list'),rollover=cfg['resid_len'])
        s3.stream(get_prog(cfg['tra_fname']),rollover=1)
        
    def update_images():
        global cfg
        global s4
        s4.stream(get_img_data(cfg),rollover=1)
        
    ## Backgroud Color Update Function
    def night_mode():
        if (pd.datetime.now().hour >= 19) or (pd.datetime.now().hour < 6):
            curdoc().theme = 'dark_minimal'
            #curdoc().template = 'dark.html'
            
        else:
            curdoc().theme = 'light_minimal'
            #curdoc().template = 'index.html'
            # THIS NEEDS TO BE A JINJA TYPE?
        ## Not sure why this doesn't work...    
        #curdoc().template_variables.update(background='#BA55D3')
        
    global cfg
    
    ## Plotting Setup
    do_not_plot =['iteration','time','iter_left','converged',
        'relative_total_time','relative_time_step']
    sizes = cfg['sizes'].tolist()
    colors = itertools.cycle(palette)
    colors_res = itertools.cycle(palette)
    hover = HoverTool()
    res_hover = HoverTool()
    if 'flow_time' in s1.data.keys():
        # Unsteady Flow (transient)
        x_opts = [('Flow Time','@flow_time s'),('Time Step','@time_step')]
        x_var = 'flow_time'
    else:
        # Steady Flow
        x_opts = [('Iteration','@iteration')]
        x_var = 'iteration'
    hover.tooltips = [('Variable','$name'),('Value','$y'),*x_opts]
    res_hover.tooltips = [('Variable','$name'),('Value','$y'),('Iteration','@iteration')]
    
    ## Plot Residuals
    w,h = sizes.pop(0)
    res_plots = figure(tools="pan,wheel_zoom,box_zoom,undo,reset,save",
        plot_width=w, plot_height=h,y_axis_type='log')
    res_plots.add_tools(res_hover)
    
    for var,color in zip(s2.data.keys(),colors_res):
        if var not in do_not_plot:
            res_plots.line('iteration', var, line_width=2, source=s2,
                color=color,legend=var.replace('_',' ').title(),
                          name=var.replace('_',' ').title())
            
    res_plots.legend.location = "top_left"
    res_plots.legend.click_policy="hide"
    
    ## Progress Bar
    prog_bar = figure(plot_width=300, plot_height=25, toolbar_location=None,x_range=[0,1])
    prog_bar.grid.visible = False
    prog_bar.axis.visible = False
    prog_bar.hbar_stack(['steps_complete', 'steps_left'], y=1, height=0.8,color=("limegreen", "lightgrey"), source=s3)
    prog_bar.text(0.5, 1, text='text',
           text_baseline="middle", text_align="center", text_color="white",text_font_style="bold",source=s3)

    ## Plot Monitor Variables
    variables = var_name_cleaner(cfg['variables'])
    if cfg['var_ylims']:
        ylims = cfg['var_ylims']
    else:
        ylims = len(variables)*[None]
        
    plots = [res_plots]
    for var,ylim,color in zip(variables,ylims,colors):
        w,h = sizes.pop(0)
        p = figure(tools="pan,wheel_zoom,box_zoom,undo,reset,save",plot_width=w, plot_height=h)
        # Link x ranges
        try:
            p.x_range=plots[1].x_range
        except: pass
        if ylim is not None:
            print(ylim)
            p.y_range=Range1d(*ylim)
            
        if isinstance(var,str):
            p.line(x_var, var, line_width=2, source=s1, color=color,
                legend=var.replace('_',' ').title(),
                name=var.replace('_',' ').title())
        elif isinstance(var,list) or isinstance(var,tuple):
            for y,color in zip(var,itertools.cycle(palette_2)):
                p.line(x_var, y, line_width=2, source=s1, color=color,
                    legend=y.replace('_',' ').title(),
                    name=y.replace('_',' ').title())
            
        p.add_tools(hover)
        p.legend.click_policy="hide"
        p.legend.location = "top_left"
        plots.append(p)
    
    ## Legend Locations
    select = Select(value="top_left", options=list(LegendLocation), width=100, height=25)
    [select.js_link('value', f.legend[0], 'location') for f in plots[1::]];
    
    ## Legend Visibility
    def legend_visibility(active):
        for n,plot in enumerate(plots):
            if n in active:
                plot.legend[0].visible = True
            else:
                plot.legend[0].visible = False
        
    legend_button_group = CheckboxButtonGroup(
        labels=['Legend '+str(n) for n in range(0,len(plots))],
        active=[n for n in range(0,len(plots))], width=cfg['width']-400, height=25)
    legend_button_group.on_click(legend_visibility)
    
    ## Turn off all active tools (helps for mobile)
    #https://stackoverflow.com/questions/49282688/how-do-i-set-default-active-tools-for-a-bokeh-gridplot
    # THIS DOESNT @!#$#@ WORK
    for p in plots:
        p.toolbar.active_drag = None
        p.toolbar.active_scroll = None
        p.toolbar.active_tap = None
   
    ## Initialize Image Plots
    if cfg['img_folder']:
        try:
            img_basenames = cfg['images']
            tabs = [Panel(child=plots[0], title='Residuals')]
            for img in img_basenames:
                p = figure(plot_width=plots[0].width, plot_height=plots[0].height, x_range=(0, 1), y_range=(0,1))
                p.image_url(url=img, x='x', y='y', w='w', h='h', source=s4)
                tabs.append(Panel(child=p, title=img))
            plots[0] = Tabs(tabs=tabs)
        except:
            print('Cannot add images')

    ## Create Layout
    # This will split the flat arry where sizes sum to the width of the doc
    sizes = cfg['sizes']
    splits = 1+np.where(np.cumsum(sizes[:,0])%cfg['width'] == 0)[0]
    layout = [x.tolist() for x in np.split(np.array(plots),splits) if len(x)>0]
    
    ## Build the Document
    #doc.template = 'dark.html'
    night_mode()
    doc.add_periodic_callback(update, cfg['polling_int'])
    if cfg['img_folder']:
        doc.add_periodic_callback(update_images, 1000*30)
    doc.add_periodic_callback(night_mode, 1000*60*10)
    doc.title='Fluent Monitors'
    
    #Panel(child=p, title=fname)
    #gps = gridplot([[prog_bar,select,legend_button_group],[layout]], toolbar_location=None)
    #doc.add_root(gps)
    
    doc.add_root(gridplot([[prog_bar,select,legend_button_group]], toolbar_location=None))
    [doc.add_root(gridplot([row], toolbar_location='right')) for row in layout]

#########################
## Pretty Print
pprint = lambda string: print('\n'+120*'='+'\n|{0:-^118s}|\n'.format('  '+string+'  ')+120*'=')
def tprint(key,value):
    key = key.replace('_',' ').title()+':'
    if not isinstance(value,str): value = str(value)
    value = value.replace('\n',' ')
    if len(value) > 100:
        value = value[0:47]+'...'+value[-50::]
    print('{0:.<20}{1:.>100}'.format(key,value))
def desc_source(source):
    df=source.to_df()
    print(df.head(5),'\n......\n',df.tail(5))
    print(df.describe(include='all'))
    
## Get the Fluent Working Directory
pprint('Connecting to Fluent')
cwd = fluent_connect(verbose=True)
## Parse Config
pprint('Reading Configuration File')
cfg = config_parse(cwd)
[tprint(k,v) for k,v in cfg.items()];

if not os.path.exists(cfg['mon_fname']):
    #from time import sleep
    print('Waiting for data file:\n',cfg['mon_fname'])
    #n = 1
    #while not os.path.exists(cfg['mon_fname']):
    #    sleep(1)
    #    print((n%10)*'.', end='')
    hold = input('Should I wait or exit?[wait]: ') or 'wait'
    if hold.lower() == 'wait':
        input('Press any key once fluent is iterating...')
## Initialize Data
s1 = ColumnDataSource(get_vars(cfg['mon_fname']).to_dict(orient='list'))
try:
    #s2 = ColumnDataSource(get_res(cfg['tra_fname'],cfg['resid_len']).to_dict(orient='list'))
    d2 = get_res(cfg['tra_fname'],cfg['resid_len']).to_dict(orient='list')        
    s2 = ColumnDataSource({k:[] for k,v in d2.items()})
except AttributeError as e:
    if 'to_dict' in str(e):
        #No iterations have run, already messaged console
        hold = input('Should I wait or exit?[wait]: ') or 'wait'
        if hold.lower() == 'wait':
            input('Press any key once fluent is iterating...')
            s1 = ColumnDataSource(get_vars(cfg['mon_fname']).to_dict(orient='list'))
            #s2 = ColumnDataSource(get_res(cfg['tra_fname'],cfg['resid_len']).to_dict(orient='list'))
            d2 = get_res(cfg['tra_fname'],cfg['resid_len']).to_dict(orient='list')        
            s2 = ColumnDataSource({k:[] for k,v in d2.items()})
        else:
            os._exit(0)
    else:
        raise e

s3 = ColumnDataSource(get_prog(cfg['tra_fname']))

if cfg['img_folder']:
    s4 = ColumnDataSource(get_img_data(cfg))
pprint('Residuals DataFrame')
desc_source(s2)
pprint('Monitors DataFrame')
desc_source(s1)
## Run the application
pprint('Starting Monitor Application')
monitors(curdoc())