import bitstring
import cbor
from io import SEEK_CUR
import numpy as np


class Streamer(object):
    """Slicable object, representing all blobs in data file,
    streaming directly from the disk.
    """

    def __init__(self, source_fn, index=None):
        self.source_fn = source_fn
        if index is None:
            self.index = self.build_index()
        else:
            self.index = index
        self.__file = open(source_fn, 'rb')

    def __del__(self):
        self.__file.close()

    def __getitem__(self, given):
        if isinstance(given, slice):
            data = []

            for i in self.index[(given.start + 1):given.stop:given.step]:
                self.__file.seek(int(i / 8))  # convert bit to byte position
                data.append(cbor.load(self.__file).value)

            return data
        else:
            self.__file.seek(int(self.index[given] / 8))
            return cbor.load(self.__file).value

    def build_index(self):
        """Builds up an index of CBOR objects by searching for CBORTag in file.
        Returns list of bit offset for the blobs.
        """
        cbortag = bitstring.BitArray(b'\xd9\xd9\xf7')

        filestream = bitstring.ConstBitStream(filename=self.source_fn)
        return list(filestream.findall(cbortag, bytealigned=True))

    def get_metadata(self):
        with open(self.source_fn, 'rb') as f:
            sim_settings = cbor.load(f).value

        return sim_settings


# def filter_index(index):
#     """ Naivly removes metadata blob, initial value blob and all blobs, that are
#     smaller than the average blob size with three standard deviations. For some
#     reason, sometimes are false positives in the index...
#     WARNING: For now, it also kicks out the last blob. Would need to take
#     filesize into account.
#     WARNING: Only works, if false tag is more at the end of the blob.
#     """
#     avg = np.average(np.diff(index[2:]))
#     return np.array(index[2:-1])[np.diff(index[2:]) > 0.5 * avg + 3 * std]


def sim_output_gen(source_file, index, start=0, step=1, stop=None):
    """Generator that yields simulation output"""

    with open(source_file, 'rb') as f:

        for i in index[start:stop:step]:
            f.seek(int(i / 8))  # convert bit to byte position
            yield cbor.load(f).value


def data_to_dist(data, gs):
    """Takes data dictonary and returns numpy array of sampled
    distribution in the correct shape, with (x, y, angle).
    """
    return np.array(data['distribution']['dist']['data']).reshape(*gs)


def dist_to_concentration(dist, gw):
    """Takes an distribution array and returns a concentration
    field by naive integraton of orientation.
    """
    return np.sum(dist, axis=2) * gw[2]


def get_bs_gs_gw(sim_settings):
    bs = sim_settings['simulation']['box_size']
    gs = sim_settings['simulation']['grid_size']

    gw = [
        sim_settings['simulation']['box_size'][0] / gs[0],
        sim_settings['simulation']['box_size'][1] / gs[1],
        2 * np.pi / gs[2]
    ]

    return bs, gs, gw


def get_scaling(source_fn, start=0, step=1, stop=None):
    sout = sim_output_gen(source_fn, index, start, step, stop)

    vmax = 0

    for data in sout:
        dist = dist_to_concentration(data_to_dist(data, gs), gw)
        m = np.max(dist)

        if m > vmax:
            vmax = m

    return m
