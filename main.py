from FastICA import negentropy_fastICA, Infomax_nature_ICA, kurtosis_ICA
from dataProcessing import load_file, save_file, plot_source, plot_k_history
import numpy as np
import argparse
from sklearn.decomposition import FastICA


def cal_similarity(origin_s, seperate_s):
    '''
    :param origin_s: original_signal (feature_num, component_num)
    :param seperate_s: seperate_signal (feature_num, component_num)
    :return: similarity matrix (component_num, component_num)
    '''
    feature_num, component_num = origin_s.shape
    similarity = np.zeros((component_num, component_num))
    for i in range(component_num):
        for j in range(component_num):
            temp = abs(np.sum(origin_s[:, i] @ seperate_s[:, j].T)) / np.sqrt(
                np.sum(np.power(origin_s[:, i], 2)) * np.sum(np.power(seperate_s[:, j], 2)))
            similarity[i, j] = temp
    return similarity


def cal_pi(G):
    '''
    calculate performance index
    :param G: w @ A (sample_number, sample_number)
    :return: pi : float
    '''
    n = G.shape[0]
    sum_total = 0

    for i in range(n):
        sum_col = 0
        sum_row = 0
        for j in range(n):
            sum_col += (abs(G[i, j]) / max(np.abs(G[i, :])))
            sum_row += (abs(G[j, i]) / max(np.abs(G[:, i])))
        sum_total = sum_total + sum_row + sum_col - 2
    pi = sum_total / (n * (n - 1))
    return pi


if __name__ == '__main__':
    # get argument from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='ICA_mix | Speech ')
    parser.add_argument('--ICA_model', type=str, default='All',
                        help='All | kurtosis_ICA | negentropy_ICA | infomax_ICA')
    parser.add_argument('--fast', type=str, default='False',
                        help='True | False, only for kurtosis_ICA')
    parser.add_argument('--eval', type=str, default='False',
                        help='True | False, calculate the evaluation functions for each result')
    parser.add_argument('--nonlinear', type=str, default='logconsh',
                        help='logconsh | cube | exp')
    parser.add_argument('--max_iterate', type=int, default=500,
                        help='max number of iterate')
    parser.add_argument('--k_eta', type=float, default=0.1,
                        help='eta for kurtosis ICA')
    parser.add_argument('--i_eta', type=float, default=0.0001,
                        help='eta for infomax ICA')
    parser.add_argument('--i_eta_gamma', type=float, default=0.3,
                        help='eta_gamma for infomax ICA')
    opt = parser.parse_args()

    if opt.fast == 'True':
        opt.fast = True
    else:
        opt.fast = False
    if opt.eval == 'True':
        opt.eval = True
    else:
        opt.eval = False

    # decide sataset
    filename = []
    if opt.dataset == 'ICA_mix':
        for i in range(1, 4):
            filename.append(f'data/ICA mix {i}.wav')
        Source_1, Time_1, params_1 = load_file(filename[0])
        Source_2, Time_2, params_2 = load_file(filename[1])
        Source_3, Time_3, params_3 = load_file(filename[2])
        data = np.vstack((Source_1[np.newaxis, :], Source_2[np.newaxis, :]))
        data = np.vstack((data, Source_3[np.newaxis, :]))
        print('Finish loading ICA_mix')
    elif opt.dataset == 'Speech':
        mix_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
        filename.append(f'data/FemaleSpeech-16-4-mono-20secs.wav')
        filename.append(f'data/MaleSpeech-16-4-mono-20secs.wav')
        Source_1, Time_1, params_1 = load_file(filename[0])
        Source_2, Time_2, params_2 = load_file(filename[1])
        data = np.vstack((Source_1[np.newaxis, :], Source_2[np.newaxis, :]))
        origin_data = data.copy()
        data = mix_matrix @ data
        # Plot input data
        plot_source(Time_1, origin_data.T, 'Input data')
        print('Finish loading Speech')
    else:
        print(opt.dataset + 'dataset is not exist!')
    # Plot input data
    plot_source(Time_1, data.T, 'Input data')
    # run algorithm
    if opt.ICA_model == 'All':
        print('Start separate signal')
        S1, k_history, w1 = kurtosis_ICA(data.T, eta=opt.k_eta, fast=opt.fast, max_iterate=opt.max_iterate)
        S2, w2 = negentropy_fastICA(data.T, model=opt.nonlinear, max_iterate=opt.max_iterate)
        S3, w3 = Infomax_nature_ICA(data.T, max_iterate=opt.max_iterate, miu=opt.i_eta, miu_gamma=opt.i_eta_gamma)
        print('Finish separate signal')

        print('Start plotting')
        x = 'with' if opt.fast else 'without'
        plot_source(Time_1, S1, 'Result of kurtosis ICA ' + x + ' fast')
        plot_source(Time_1, S2, 'Result of fast ICA')
        plot_source(Time_1, S3, 'Result of Infomax natural gradient descent ICA')
        print('Finish plotting')
        print('Saving result')
        for i in range(1, data.shape[0]):
            save_file(f'result/{opt.dataset}_kurtosis_ICA_result_{i}_{x}_fast.wav', params_1, S1[:, i - 1])
            save_file(f'result/{opt.dataset}_fast_ICA_result_{i}.wav', params_1, S2[:, i - 1])
            save_file(f'result/{opt.dataset}_Infomax_nature_ICA_result_{i}.wav', params_1, S3[:, i - 1])

    elif opt.ICA_model == 'kurtosis_ICA':
        S1, k_history, w1 = kurtosis_ICA(data.T, eta=opt.k_eta, fast=opt.fast, max_iterate=opt.max_iterate)
        S2 = None
        S3 = None
        print('Finish separate signal')

        print('Start plotting')
        x = 'with' if opt.fast else 'without'
        plot_source(Time_1, S1, 'Result of kurtosis ICA ' + x + ' fast')
        print('Finish plotting')
        print('Saving result')
        for i in range(1, data.shape[0]):
            save_file(f'result/{opt.dataset}_kurtosis_ICA_result_{i}_{x}_fast.wav', params_1, S1[:, i - 1])

    elif opt.ICA_model == 'negentropy_ICA':
        S1 = None
        S2, w2 = negentropy_fastICA(data.T, model=opt.nonlinear, max_iterate=opt.max_iterate)
        S3 = None
        print('Finish separate signal')
        print('Start plotting')
        plot_source(Time_1, S2, 'Result of fast ICA')
        print('Finish plotting')

        print('Saving result')
        for i in range(1, data.shape[0]):
            save_file(f'result/{opt.dataset}_fast_ICA_result_{i}.wav', params_1, S2[:, i - 1])

    elif opt.ICA_model == 'infomax_ICA':
        S1 = None
        S2 = None
        S3, w3 = Infomax_nature_ICA(data.T, max_iterate=opt.max_iterate, miu=opt.i_eta, miu_gamma=opt.i_eta_gamma)
        print('Finish separate signal')

        print('Start plotting')
        plot_source(Time_1, S3, 'Result of Infomax natural gradient descent ICA')
        print('Finish plotting')
        print('Saving result')
        for i in range(1, data.shape[0]):
            save_file(f'result/{opt.dataset}_Infomax_nature_ICA_result_{i}.wav', params_1, S3[:, i - 1])


    else:
        print(opt.ICA_model + ' model is not exist!')

    if opt.eval:
        print('Start evaluation')
        transformer = FastICA(n_components=data.shape[0])
        X_transformed = transformer.fit_transform(data.T)
        plot_source(Time_1, X_transformed, 'Result of inbuilt Fast ICA')
        if opt.dataset == 'Speech':
            if opt.ICA_model == 'All':
                ''' only for test
                if not opt.fast:
                   plot_k_history(k_history)'''
                similarity1 = cal_similarity(origin_data.T, S1)
                print('Similarity of kurtosis ICA and original source:\n', similarity1)

                similarity2 = cal_similarity(origin_data.T, S2)
                print('Similarity of negentropy ICA and original source:\n', similarity2)

                similarity3 = cal_similarity(origin_data.T, S3)
                print('Similarity of infomax ICA and original source:\n', similarity3)
                pi1 = cal_pi(mix_matrix @ w1)
                pi2 = cal_pi(mix_matrix @ w2)
                pi3 = cal_pi(mix_matrix @ w3)
                print('pi result of model kurtosis_ICA is: ' + str(pi1))
                print('pi result of model negentropy_ICA is: ' + str(pi2))
                print('pi result of model infomax_ICA is: ' + str(pi3))

            elif opt.ICA_model == 'kurtosis_ICA':
                similarity = cal_similarity(origin_data.T, S1)
                print('Similarity of kurtosis ICA and original source:\n', similarity)
                pi1 = cal_pi(mix_matrix @ w1)
                print('pi result of model kurtosis_ICA is: ' + str(pi1))

            elif opt.ICA_model == 'negentropy_ICA':
                similarity = cal_similarity(origin_data.T, S2)
                print('Similarity of negentropy ICA and original source:\n', similarity)
                pi2 = cal_pi(mix_matrix @ w2)
                print('pi result of model negentropy_ICA is: ' + str(pi2))

            else:
                similarity = cal_similarity(origin_data.T, S3)
                print('Similarity of infomax ICA and original source:\n', similarity)

                pi3 = cal_pi(mix_matrix @ w3)
                print('pi result of model infomax ICA is: ' + str(pi3))
        else:
            if opt.ICA_model == 'All':
                similarity1 = cal_similarity(X_transformed, S1)
                print('Similarity of kurtosis ICA and original source:\n', similarity1)

                similarity2 = cal_similarity(X_transformed, S2)
                print('Similarity of negentropy ICA and original source:\n', similarity2)

                similarity3 = cal_similarity(X_transformed, S3)
                print('Similarity of infomax ICA and original source:\n', similarity3)

            elif opt.ICA_model == 'kurtosis_ICA':
                similarity = cal_similarity(X_transformed, S1)
                print('Similarity of kurtosis ICA and original source:\n', similarity)

            elif opt.ICA_model == 'negentropy_ICA':
                similarity = cal_similarity(X_transformed, S2)
                print('Similarity of negentropy ICA and original source:\n', similarity)

            else:
                similarity = cal_similarity(X_transformed, S3)
                print('Similarity of infomax ICA and original source:\n', similarity)
    else:
        pass
    print('All done!')
