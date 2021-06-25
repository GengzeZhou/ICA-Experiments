# -*- coding: utf-8 -*-
import numpy as np
import wave
import struct
import matplotlib.pyplot as plt


def load_file(filename):
    '''
    load data into numpy.array
    :param filename: data file name
    :return: S: signal
            T: time sequence
            param: audio parameters
    '''
    # read audio signal
    mix1 = wave.open(filename, 'r')

    # load parameters of audio
    # nchannels: number of channel；
    # sampwidth digitalizing bit :byte；
    # framerate: sample frequency；
    # nframes: sampling number
    params = mix1.getparams()
    [nchannels, sampwidth, framerate, nframes] = params[:4]

    # Read audio in string format
    str_data = mix1.readframes(nframes)
    waveData = np.frombuffer(str_data, dtype=np.int16)

    # close file
    mix1.close()

    # normalize signal
    Source = waveData / (max(abs(waveData)))
    # T = 1/f time sequence
    Time = np.arange(0, nframes) / framerate

    return Source, Time, params


def save_file(filename, params, Source):
    '''
    save all result
    :param filename: result file name
    :param params: parameter of audio
    :param Source: signal of audio
    :return: None
    '''
    # Create a new file for the audio data
    out_wave = wave.open(filename, 'w')
    # Set the parameters of the file
    out_wave.setparams((params))
    # normalize signal
    Source = Source / (max(abs(Source)))
    # Writes data to a file frame by frame
    # Source:int16，-32767~32767
    for i in Source:
        out_wave.writeframes(struct.pack('h', int(i * 64000 / 2)))
    out_wave.close()

def plot_source(T, S, graph_title):
    '''
    show plot of signal
    :param T: time sequence
    :param S: signal
    :param graph_title: title of graph
    :return: figure
    '''
    n = S.shape[1]
    figure, axes = plt.subplots(n, 1, constrained_layout=True)
    figure.suptitle(graph_title)
    for i in range(n):
        axes[i].set_title('Source {}'.format(i + 1))
        axes[i].plot(T, S[:, i], color='b')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')
    plt.show()


def plot_k_history(k_history):
    '''
    just for test
    '''
    plt.plot([i for i in range(len(k_history))], k_history)
    plt.xlabel('iterate time')
    plt.ylabel('kurt')
    plt.title('Curve of kurt history')
    plt.show()
