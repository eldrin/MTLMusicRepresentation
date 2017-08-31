import os
import numpy as np
from math import factorial
import argparse

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))

    except ValueError, msg:
        raise ValueError('window_size and order have to be of type of int')
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError('window_size must be a positive odd number')
    if window_size < order + 2:
        raise TypeError('window_size is too small for the polynomials order')
    order_range = range(order+1)
    half_window = (window_size-1)//2
    # precompute coefficients
    b = np.mat(
        [
            [k**i for i in order_range]\
            for k in range(-half_window, half_window+1)
        ]
    )
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals,y,lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def parse_log(log_file,keywords):
    """parse log file by given keywords

    Inputs

    log_file : file-like object
    file-like objects for log file

    keywords : list of string
    list of keywords which is visualized

    """

    lines = map(
        lambda x:x.replace('\n',''),
        log_file.readlines()
    )

    try:
        curves = {}
        for kw in keywords:
            kw = kw + ':'
            curves[kw] = np.array(
                map(
                    lambda x:
                    float(
                        x.split(kw)[1].split(' ')[0]
                    ),
                    filter(
                        lambda x:
                        kw in x,
                        lines
                    )
                )
            )

            if len(curves[kw])==0:
                raise ValueError
    except ValueError:
        raise ValueError('At least one value must exist to visualize')

    except Exception as e:
        print(e)

    return curves


def is_valid_file(parser,arg):
    if not os.path.exists(arg):
        parser.error("The file {} does not exist!".format(arg))
    else:
        return open(arg,'r')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'logfile',help='path of log file to visualize',metavar="FILE",
        type=lambda x:is_valid_file(parser,x))
    parser.add_argument(
        '--keywords',nargs='+',
        help='list of keywords to visualize')
    parser.add_argument(
        '-o','--outfile',type=str,default=None,
        help='visualzation out file path')
    parser.add_argument(
        '-m','--merge',type=str,default=None,
        help='indicating limited merging for candidate curves {"sum"}')

    parser.add_argument(
        '--split',dest='split',action='store_true')
    parser.add_argument(
        '--no-split',dest='split',action='store_false')
    parser.set_defaults(split=True)

    args = parser.parse_args()

    import matplotlib
    if args.outfile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    curves = parse_log(args.logfile,args.keywords)

    if args.merge is not None:
        if args.merge == 'sum':
            new_curves = {}
            new_curves['sum_curve'] = np.sum(
                curves.values(),axis=0
            )
            curves = new_curves

    # for now, first cost and acc omitted
    # TODO:later, it would be controlled by user with argument

    # plot
    if args.split:
        fig,axs = plt.subplots(len(curves.keys()),1)
        smooth_curves = {}
        j = 0
        for k,v in curves.iteritems():
            smooth_curves[k] = savitzky_golay(
                v,window_size=31,order=3
            )

            if len(curves)==1:
                axs.plot(v,alpha=0.3)
                axs.plot(smooth_curves[k],label=k)
                axs.legend()
            else:
                axs[j].plot(v,alpha=0.3)
                axs[j].plot(smooth_curves[k],label=k)
                axs[j].legend()
                j+=1
    else:
        fig,axs = plt.subplots(1,1)
        smooth_curves = {}
        j = 0
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        for k,v in curves.iteritems():
            smooth_curves[k] = savitzky_golay(
                v,window_size=31,order=3
            )

            axs.plot(v,alpha=0.3,c=colors[j%10])
            axs.plot(smooth_curves[k],c=colors[j%10],label=k)
            axs.legend()
            j+=1

    plt.grid(True)

    if args.outfile is None:
        plt.show()
    else:
        plt.savefig(args.outfile)

    # terminate app
    args.logfile.close()
