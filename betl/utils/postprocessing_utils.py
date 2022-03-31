from PIL import Image
import glob
import json
import os
import matplotlib.colors as mc
import colorsys


def removeImages(img_path):
    for f in glob.glob(img_path):
        os.remove(f)


def createGIF(img_path, save_path, delete_imgs=False, duration=200):
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    all_imgs = sorted(glob.glob(img_path))
    img, *imgs = [Image.open(f) for f in all_imgs]
    img.save(fp=save_path, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=0)
    if delete_imgs:
        removeImages(img_path)


def initialize_plot(purpose):
    path = '.'
    color_json = glob.glob(path + '/**/RWTHcolors.json', recursive=True)
    with open(color_json[0]) as json_file:
        c = json.load(json_file)

    if purpose is 'CDC_paper':
        plot_params = {
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            'text.latex.preamble': r'\usepackage{amsfonts}',
            "font.size": 9
        }
    else:
        plot_params = {}

    return c, plot_params


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])