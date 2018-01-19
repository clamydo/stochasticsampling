#!/usr/bin/env python3

# This is a hacked up viewer for simulation data in a concatinated CBOR format.
# Sorry for the ugliness.

import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from DataStreamer import Streamer
import DataStreamer
import argparse
import json
from pathlib import Path
from scipy.fftpack import fftn
from scipy import special
import DataStreamer
from DataStreamer import Streamer

parser = argparse.ArgumentParser(description='Generate initial condition.')
parser.add_argument('data',
                    help='Path to data file in CBOR format.')
args = parser.parse_args()


index_file = Path(args.data).with_suffix(".index")

if index_file.is_file():
    index = np.fromfile(str(index_file), dtype=np.uint64)
    ds = Streamer(args.data, index=index)
else:
    print('Build index...')
    ds = Streamer(args.data)
    index = ds.index


sim_settings = ds.get_metadata()
if sim_settings['parameters']['diffusion']['rotational'] != 0:
    kappa = sim_settings['parameters']['magnetic_reorientation'] / \
        sim_settings['parameters']['diffusion']['rotational']
else:
    kappa = 10000000

bs, gs, gw = DataStreamer.get_bs_gs_gw(sim_settings)

print(json.dumps(sim_settings, indent=1))

fig, axs = plt.subplots(2, 2)
# plt.subplots_adjust(left=0.25, bottom=0.25)

p = axs[1, 0].imshow(np.zeros((gs['x'], gs['z'])).T, origin='lower')
# fig.colorbar(p, ax=axs[1])

v_x = np.linspace(gw['x'] / 2., bs['x'] - gw['x'] / 2, gs['x'])
v_z = np.linspace(gw['y'] / 2., bs['z'] - gw['z'] / 2, gs['z'])
v_X, v_Z = np.meshgrid(v_x, v_z)

ang_x = np.linspace(gw['phi'] / 2, 2 * np.pi - gw['phi'] / 2, gs['phi'], endpoint=True)

fft_plot = axs[0, 1].imshow(np.zeros((gs['x'], gs['z'])).T, origin='lower')
axs[0, 1].set_title('DFT')

ang_dist_th_plot, = axs[0, 0].plot([], [])
ang_dist_plot, = axs[0, 0].plot([], [], '.')
axs[0, 0].set_title('angular distribution')

quiv = axs[1, 1].quiver(v_X, v_Z,
                        np.ones((gs['x'], gs['z'])), np.ones((gs['x'], gs['z'])),
                        scale=None)

axslice = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor='lightgoldenrodyellow')
sslice = Slider(axslice, 'Slice',
                0, 1, valinit=0,
                dragging=False)


for ax in axs.flatten():
    ax.set_aspect(1)


def update_title(i, timestep):
    axs[1, 0].set_title('Slice: {}/{}, time: {}'.format(
        i,
        len(ds.index) - 1,
        timestep * sim_settings['simulation']['timestep']))


def psi(x, kappa):
    return np.exp(kappa * np.sin(x)) / np.abs(special.iv(0, kappa)) / 2 / np.pi


def update(val):
    i = int(val * (len(ds.index) - 2))
    #i = int(val)
    data = ds[i]
    d = DataStreamer.data_to_dist(data)
    c = DataStreamer.dist_to_concentration3d(d, gw)

    update_title(i, data['timestep'])

    dft = fftn(c, axes=(0, 1, 2))
    mag = np.abs(dft)
    mag = mag / mag[0, 0]
    mag[0, 0] = 0

    fft_plot.set_data(mag[:, 0, :].T)
    fft_plot.autoscale()

    # th = psi(ang_x, kappa)
    # ang_dist_th_plot.set_data(ang_x, th)
    # ang_dist = np.mean(d, axis=(0, 1))
    # ang_dist_plot.set_data(ang_x, ang_dist)
    # axs[0, 0].set_xlim([0, 2 * np.pi])
    # axs[0, 0].set_ylim([0, max(np.max(ang_dist), np.max(th)) * 1.05])
    # axs[0, 0].set_aspect('auto')

    mean_c = np.mean(c, axis=1)
    p.set_data(mean_c.T)
    p.autoscale()

    ff = data['flowfield']
    if ff is None:
        axs[1, 1].set_title('no flowfield availabe')
        quiv.set_UVC(np.ones((gs['x'], gs['z'])), np.ones((gs['x'], gs['z'])))
    else:
        ff = DataStreamer.data_to_flowfield(data)
        axs[1, 1].set_title('flowfield')
        # quiv.set_UVC(ff[0].T, ff[1].T)
        axs[1, 1].cla()
        x = np.arange(gs['x'])
        z = np.arange(gs['z'])
        axs[1, 1].streamplot(x, z, ff[0, :, 0, :].T, ff[1, :, 0, :].T)

    fig.canvas.draw_idle()


# register arrow keys
def press(event):
    step = 10

    max = len(ds.index)

    if event.key == 'left':
        if sslice.val - 1 / max >= 0:
            sslice.set_val(sslice.val - 1 / max)
        else:
            sslice.set_val(0)
    if event.key == 'right':
        if sslice.val + 1 / max <= 1:
            sslice.set_val(sslice.val + 1 / max)
        else:
            sslice.set_val(1)
    if event.key == 'down':
        if sslice.val - step / max >= 0:
            sslice.set_val(sslice.val - step / max)
        else:
            sslice.set_val(0)
    if event.key == 'up':
        if sslice.val + step / max <= 1:
            sslice.set_val(sslice.val + step / max)
        else:
            sslice.set_val(1)
    if event.key == 'home':
        sslice.set_val(0)
    if event.key == 'end':
        sslice.set_val(1)
    if event.key == 'f5':
        if index_file.is_file():
            index = np.fromfile(str(index_file), dtype=np.uint64)
            ds.set_index(index)
        else:
            index = ds.rebuild_index()

        print('Reloaded file. Now have {} slices'.format(len(ds.index) - 1))

        sslice.set_val(1)

        fig.canvas.draw_idle()


sslice.on_changed(update)
fig.canvas.mpl_connect('key_press_event', press)


update(1)
plt.show()
