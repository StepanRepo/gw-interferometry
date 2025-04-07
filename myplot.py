import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

def save_image(filename, pad_inches = 0, tight = False, *args, **kwargs):
	# PdfPages is a wrapper around pdf
	# file so there is no clash and create
	# files with no error.
    with PdfPages(filename) as p:

        # get_fignums Return list of existing
        # figure numbers
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]

        # iterating over the numbers in list
        for fig in tqdm(figs):

            # and saving the files
            if tight:
                fig.savefig(p, format='pdf', 
                            bbox_inches='tight', 
                            pad_inches = pad_inches,
                            *args,
                            **kwargs)
            else:
                fig.savefig(p, format='pdf',
                            pad_inches = pad_inches,
                            *args,
                            **kwargs)

            plt.close(fig)

def set_tex():

    try:
        plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": "serif",
        "font.size"   : 12
        })
    except:
        print("TeX is not availible on the system")
