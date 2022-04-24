import os
import glob
import h5py
from DVS_dataload.events_timeslices import *

TimeStep = 10
ds = [3.04, 3.04]
dt = 10 * 1000
size = [2, 42, 42]

def aedat_to_events(path):
    events = []
    with open(path, 'rb') as f:
        for _ in range(5):
            f.readline()

        event_bytes = np.frombuffer(f.read(), '>I')
        allAddr = event_bytes[0::2]
        allTs = event_bytes[1::2]

        x = 128 - 1 - ((allAddr & 0x000000FE) >> 1)
        y = (allAddr & 0x7f00) >> 8
        p = allAddr & 0x00000001
        t = allTs
        events.append([t, x, y, p])

        events = np.column_stack(events)
        events = events.astype('uint32')

        return events.T


def gather_classpath_list(path):
    classpath_list = glob.glob(os.path.join(path, '*.zip'))
    for i in range(len(classpath_list)):
        classpath_list[i] = classpath_list[i][:-4]
    return classpath_list


def gather_aedat(path, train_list, test_list):
    list = glob.glob(os.path.join(path, '*.aedat'))
    train_list += list[0:900]
    test_list += list[900:1000]

    return None


def sample(times, addrs, T, dt, size, ds, is_train_Enhanced=False):
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
            pol, x, y = data_temp[:, 2], (data_temp[:, 0] // ds[0]).astype(np.int),\
                        (data_temp[:, 1] // ds[1]).astype(np.int)
            np.add.at(re, (i, pol, x, y), 1)
        idx_start = idx_end
    return re


def create_hdf5(path, save_path):
    classpath_list = gather_classpath_list(path)

    train_list = []
    test_list = []
    train_label = []
    test_label = []

    for i in range(len(classpath_list)):
        train_label += [i for _ in range(900)]
        test_label += [i for _ in range(100)]
        gather_aedat(classpath_list[i], train_list, test_list)

    #  trian file creat
    print('processing train data...')
    save_path_train = os.path.join(save_path, 'DVS_CIFAR10_train_10ms_10step')
    if not os.path.exists(save_path_train):
        os.makedirs(save_path_train)
    for i in range(len(train_list)):
        print('processing training data: {}/{}, {:.1f} %'.format(i+1, len(train_label), 100.*(i+1)/len(train_label)))
        label = train_label[i]
        data = aedat_to_events(train_list[i])
        tms = data[:, 0]
        ads = data[:, 1:]
        data = sample(tms, ads, T=TimeStep, dt=dt, size=size, ds=ds, is_train_Enhanced=False)

        with h5py.File(save_path_train + os.sep + 'DVS-CIFAR10-train' + str(i) + '.hdf5', 'w') as f:
            f.create_dataset('data', data=data, dtype=np.int8)
            f.create_dataset('label', data=label, dtype=np.int8)

    print('Training data processing completed')

    #  test file creat
    print('processing test data...')
    save_path_test = os.path.join(save_path, 'DVS_CIFAR10_test_10ms_10step')
    if not os.path.exists(save_path_test):
        os.makedirs(save_path_test)
    for i in range(len(test_list)):
        print('processing testing data: {}/{}, {:.1f} %'.format(i+1, len(test_list), 100.*(i+1)/len(test_list)))
        label = test_label[i]
        data = aedat_to_events(test_list[i])
        tms = data[:, 0]
        ads = data[:, 1:]
        data = sample(tms, ads, T=TimeStep, dt=dt, size=size, ds=ds, is_train_Enhanced=False)

        with h5py.File(save_path_test + os.sep + 'DVS-CIFAR10-test' + str(i) + '.hdf5', 'w') as f:
            f.create_dataset('data', data=data, dtype=np.int8)
            f.create_dataset('label', data=label, dtype=np.int8)
    print('Testing data processing completed')


if __name__ == '__main__':
    save_path = os.getcwd() + os.sep + 'data' + os.sep + 'DVS_CIFAR10'
    read_path = save_path + os.sep + 'source_DvsCIFAR10'
    train_list = create_hdf5(read_path, save_path)
