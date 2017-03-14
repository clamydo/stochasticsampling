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

print(sim_settings)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
p = plt.imshow(np.zeros((gs[0], gs[1])), origin='lower')

axslice = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

sslice = Slider(axslice, 'Slice',
                0, 1, valinit=0,
                dragging=False)


def update_title(i, timestep):
    ax.set_title('Slice: {}/{}, Timestep: {}'.format(
        i + 1, len(ds.index), timestep))


def update(val):
    i = int(val * (len(ds.index) - 1))
    try:
        data = ds[i]
        c = DataStreamer.dist_to_concentration(
                DataStreamer.data_to_dist(data, gs),
                gw
            )

        update_title(i, data['timestep'])

        p.set_data(c.T)
        p.autoscale()
        fig.canvas.draw_idle()
    except:
        ax.set_title('Failed to read timestep')


# register arrow keys
def press(event):
    step = 10

    max = len(ds.index) - 1

    if event.key == 'left':
        if sslice.val - 1/max >= 0:
            sslice.set_val(sslice.val - 1/max)
    if event.key == 'right':
        if sslice.val + 1/max <= 1:
            sslice.set_val(sslice.val + 1/max)
    if event.key == 'down':
        if sslice.val - step/max >= 0:
            sslice.set_val(sslice.val - step/max)
    if event.key == 'up':
        if sslice.val + step/max <= 1:
            sslice.set_val(sslice.val + step/max)
    if event.key == 'home':
        sslice.set_val(0)
    if event.key == 'end':
        sslice.set_val(1)
    if event.key == 'f5':
        i = int(sslice.val * (len(ds.index) - 1))

        ds.rebuild_index()
        print('Reloaded file. Now have {} slices'.format(len(ds.index)))

        sslice.set_val(1)

        fig.canvas.draw_idle()


sslice.on_changed(update)
fig.canvas.mpl_connect('key_press_event', press)


update(1)
plt.show()
