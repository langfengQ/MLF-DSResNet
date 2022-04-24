import os
import h5py
import struct
import glob
from DVS_dataload.events_timeslices import *


TimeStep = 40
ds = [4, 4]
dt = 30 * 1000
size=[2, 32, 32]


def gather_aedat(directory, start_id, end_id, filename_prefix='user'):
    fns = []
    for i in range(start_id, end_id):
        search_mask = directory + os.sep + \
            filename_prefix + "{0:02d}".format(i) + '*.aedat'
        glob_out = glob.glob(search_mask)
        if len(glob_out) > 0:
            fns += glob_out
    return fns
# read aedat


def aedat_to_events(filename):
    # read label
    label_filename = filename[:-6] + '_labels.csv'
    labels = np.loadtxt(label_filename,
                        skiprows=1,
                        delimiter=',',
                        dtype='uint32')

    events = []
    with open(filename, 'rb') as f:

        for i in range(5):
            _ = f.readline()

        while True:
            data_ev_head = f.read(28)
            if len(data_ev_head) == 0:
                break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsource = struct.unpack('H', data_ev_head[2:4])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventoffset = struct.unpack('I', data_ev_head[8:12])[0]
            eventtsoverflow = struct.unpack('I', data_ev_head[12:16])[0]
            eventcapacity = struct.unpack('I', data_ev_head[16:20])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]
            eventvalid = struct.unpack('I', data_ev_head[24:28])[0]

            if (eventtype == 1):
                event_bytes = np.frombuffer(f.read(eventnumber * eventsize),
                                            'uint32')
                event_bytes = event_bytes.reshape(-1, 2)

                x = (event_bytes[:, 0] >> 17) & 0x00001FFF
                y = (event_bytes[:, 0] >> 2) & 0x00001FFF
                p = (event_bytes[:, 0] >> 1) & 0x00000001
                t = event_bytes[:, 1]
                events.append([t, x, y, p])

            else:
                f.read(eventnumber * eventsize)

    events = np.column_stack(events)
    events = events.astype('uint32')

    clipped_events = np.zeros([4, 0], 'uint32')

    for l in labels:
        start = np.searchsorted(events[0, :], l[1])
        end = np.searchsorted(events[0, :], l[2])
        clipped_events = np.column_stack([clipped_events,
                                          events[:, start:end]])

    return clipped_events.T, labels


# build hdf5
def sample(times, addrs, T=40, dt=30000, size=[2, 32, 32], ds=[4,4], is_train_Enhanced=False):

    tbegin = times[0]
    tend = np.maximum(0, times[-1] - T * dt)

    start_time = np.random.randint(tbegin, tend) if is_train_Enhanced else 0

    data = get_tmad_slice(times[()],
                          addrs[()],
                          start_time,
                          T * dt)
    data[:, 0] -= data[0, 0]

    t_start = data[0][0]
    ts = range(t_start, t_start + T * dt, dt)
    re = np.zeros([len(ts)] + size, dtype='int8')
    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        idx_end += find_first(data[idx_end:, 0], t + dt)
        if idx_end > idx_start:
            data_temp = data[idx_start:idx_end, 1:]
            pol, y, x = data_temp[:, 2], (data_temp[:, 0] // ds[0]).astype(np.int),\
                        (data_temp[:, 1] // ds[1]).astype(np.int)
            np.add.at(re, (i, pol, x, y), 1)
        idx_start = idx_end
    return re


def create_hdf5(path, save_path):
    print('processing train data...')
    save_path_train = os.path.join(save_path, 'DvsGesture_train_40step_downsample')
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)

    fns_train = gather_aedat(path, 1, 24)
    index_data_save = 0
    for i in range(len(fns_train)):
        print('processing training data: {}/{}, {:.1f} %'.format(i+1, len(fns_train), 100.*(i+1)/len(fns_train)))
        data, labels_starttime = aedat_to_events(fns_train[i])
        tms = data[:, 0]
        ads = data[:, 1:]
        lbls = labels_starttime[:, 0]
        start_tms = labels_starttime[:, 1]
        end_tms = labels_starttime[:, 2]
        for lbls_idx in range(len(lbls)):
            s_ = get_slice(tms, ads, start_tms[lbls_idx], end_tms[lbls_idx])
            data = sample(s_[0], s_[1], T=TimeStep, dt=dt, size=size, ds=ds, is_train_Enhanced=False)
            index_data_save += 1

            with h5py.File(save_path_train + os.sep + 'DVS-Gesture-train' + str(index_data_save) + '.hdf5',
                           'w') as f:
                f.create_dataset('data', data=data, dtype=np.int8)
                f.create_dataset('label', data=lbls[lbls_idx] - 1, dtype=np.int8)
    print('Training data processing completed')

    print('processing test data...')
    save_path_test = os.path.join(save_path, 'DvsGesture_test_40step_downsample')
    if not os.path.exists(save_path_test):
        os.makedirs(save_path_test)

    fns_test = gather_aedat(path, 24, 30)
    index_data_save = 0

    for i in range(len(fns_test)):
        print('processing testing data: {}/{}, {:.1f} %'.format(i+1, len(fns_test), 100.*(i+1)/len(fns_test)))
        data, labels_starttime = aedat_to_events(fns_test[i])
        tms = data[:, 0]
        ads = data[:, 1:]
        lbls = labels_starttime[:, 0]
        start_tms = labels_starttime[:, 1]
        end_tms = labels_starttime[:, 2]
        for lbls_idx in range(len(lbls)):
            s_ = get_slice(tms, ads, start_tms[lbls_idx], end_tms[lbls_idx])
            data = sample(s_[0], s_[1], T=TimeStep, dt=dt, size=size, ds=ds, is_train_Enhanced=False)
            index_data_save += 1
            with h5py.File(save_path_test + os.sep + 'DVS-Gesture-test' + str(index_data_save) + '.hdf5',
                           'w') as f:
                f.create_dataset('data', data=data, dtype=np.uint8)
                f.create_dataset('label', data=lbls[lbls_idx] - 1, dtype=np.uint8)

    print('Test data processing completed')


if __name__ == '__main__':
    path = os.getcwd() + os.sep + 'data' + os.sep + 'DVS_Gesture'
    create_hdf5(os.path.join(path, 'source_DvsGesture'), path)
