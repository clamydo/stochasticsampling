import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from DataStreamer import Streamer
import DataStreamer
import argparse

parser = argparse.ArgumentParser(description='Generate initial condition.')
parser.add_argument('data',
                    help='Path to data file in CBOR format.')
args = parser.parse_args()


ds = Streamer(args.data)
sim_settings = ds.get_metadata()
bs, gs, gw = DataStreamer.get_bs_gs_gw(sim_settings)


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
p = plt.imshow(np.zeros((gs[0], gs[1])), origin='lower')

axslice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

sslice = Slider(axslice, 'Timestep', 1, len(ds.index) - 1,
                valinit=0, valfmt='%0.0f', dragging=False)


def update(val):
    i = int(val)
    try:
        data = ds[i]
        c = DataStreamer.dist_to_concentration(
                DataStreamer.data_to_dist(data, gs),
                gw
            )

        ax.set_title('Timestep: {}'.format(data['timestep']))

        p.set_data(c.T)
        p.autoscale()
        fig.canvas.draw_idle()
    except:
        ax.set_title('Failed to read timestep')


# register arrow keys
def press(event):
    step = 10

    if event.key == 'left':
        if sslice.val > 0:
            sslice.set_val(sslice.val - 1)
    if event.key == 'right':
        if sslice.val < len(ds.index) - 1:
            sslice.set_val(sslice.val + 1)
    if event.key == 'down':
        if sslice.val - step >= 0:
            sslice.set_val(sslice.val - step)
    if event.key == 'up':
        if sslice.val + step <= len(ds.index) - 1:
            sslice.set_val(sslice.val + step)


sslice.on_changed(update)
fig.canvas.mpl_connect('key_press_event', press)


update(1)
plt.show()
