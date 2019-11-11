import argparse
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os


import lsst.afw.image as afw_image
import lsst.geom as lsst_geom
from lsst.eo_utils.sflat.file_utils import get_sflat_files_run



def analyze_profile(xarr, xval, yval):
    analyzed = 0
    swing_15 = 0
    swing_10 = 0
    swing_5 = 0
    xmin = min(xarr)
    xmax = max(xarr)
    
    if xmax - xmin > 1300:
        yval_array = np.array(yval)
        analyzed = 1
        std = (xmax - xmin) / 6
        analyze = np.logical_and(xval >= xmin + 1.1*std, xval <= xmax - 1.1*std)
        analyze_yvals = yval_array[analyze]
        ymax = max(analyze_yvals)
        ymin = min(analyze_yvals)
        if ymax < 0.55 and ymin > 0.45:
            swing_5 = 1
        if (ymax >= 0.55 or ymin <= 0.45) and (ymax < 0.60 and ymin > 0.40):
            swing_10 = 1
        if (ymax >= 0.60 or ymin <= 0.40) and (ymax < 0.65 and ymin > 0.35):
            swing_15 = 1
        if np.sum(np.array([swing_5, swing_10, swing_15])) > 1:
            print('Your logic is flawed')
    return analyzed, swing_15, swing_10, swing_5 



def get_bbox(keyword, dxmin=0, dymin=0, dxmax=0, dymax=0):
    """
    Parse an NOAO section keyword value (e.g., DATASEC = '[1:509,1:200]') from the
    FITS header and return the corresponding bounding box for sub-image retrieval.
    """
    xmin, xmax, ymin, ymax = [val - 1 for val in eval(keyword.replace(':', ','))]
    bbox = lsst_geom.Box2I(lsst_geom.Point2I(xmin + dxmin, ymin + dymin),
                          lsst_geom.Point2I(xmax + dxmax, ymax + dymax))
    return bbox



def make_master_dict(runs, imtype):
#This function is an abomination
    master_dict = {}
    runs_RTM_dict = {}
    RTM_runs_dict = {}
    SNNs = ['S00', 'S01', 'S02', 'S10', 'S11', 'S12', 'S20', 'S21', 'S22']
    for run in runs:
        runs_RTM_dict[run] = get_sflat_files_run(run)
        for RTM in runs_RTM_dict[run].keys():
            RTM_runs_dict[RTM] = []

    for run, RTM_dict in runs_RTM_dict.items():
        for RTM in RTM_dict.keys():
            RTM_runs_dict[RTM].append(run)

    for RTM, run_list in RTM_runs_dict.items():
        master_dict[RTM] = {}
        for SNN in SNNs:
            master_dict[RTM][SNN] = {'SFLAT': {}}
            for run in run_list:
                master_dict[RTM][SNN]['SFLAT'][run] = []

    for RTM, SNN_dict in master_dict.items():
        for SNN, eotype_dict in SNN_dict.items():
            for eotype, run_dict in eotype_dict.items():
                for run, imfile_list in run_dict.items():
                    for imfile in runs_RTM_dict[run][RTM][SNN][eotype]:
                        if imtype in imfile:
                            imfile_list.append(imfile)
    return master_dict



def make_working_dict(runs_dict, RTMnum, SNNnum, hdunum, imtypestr):
    working_dict = {'RTM': RTMnum, 'SNN': SNNnum, 'hdu': '{:02d}'.format(hdunum), 'imtype': imtypestr, 'arrays': {}, 'sensors': {}}

    for run, files_list in runs_dict.items():
        imarray = im_to_array(files_list[0], hdunum)
        im_md = afw_image.readMetadata(files_list[0], 0)

        imarray, low, high = scipy.stats.sigmaclip(imarray, low = 3.0, high = 3.0)
        CCD_name = im_md.get('LSST_NUM')

        working_dict['arrays'][run] = imarray
        working_dict['sensors'][run] = CCD_name

    return working_dict



def mkProfile(ax, xarr, yarr, p_label, nx=100, xmin=0., xmax=1.0, ymin=0., ymax=1.0, retPlot=True):

    dx = (xmax-xmin) / nx
    bins = np.arange(xmin, xmax, dx)
    ind = np.digitize(xarr, bins)
    xval = []
    xerr = []
    yval = []
    yerr = []
    for i in range(len(bins)-1):

        here = (ind==i+1)
        ygood = np.logical_and(yarr>=ymin, yarr<=ymax)
        ok = np.logical_and(ygood,here)
        yinthisbin = yarr[ok]
        yhere = np.array(yinthisbin)
        n = len(yinthisbin)
        if n>0:
            xval.append(0.5*(bins[i+1]+bins[i]))
            xerr.append(0.5*(bins[i+1]-bins[i]))
            yval.append(yhere.mean())
            yerr.append(yhere.std()/n)

    analyzed, swing_15, swing_10, swing_5 = analyze_profile(xarr, xval, yval)
    if retPlot:
        profile = ax.errorbar(xval,yval,xerr=xerr,yerr=yerr, label = p_label)
        return profile, analyzed, swing_15, swing_10, swing_5
    else:
        return xval, yval, xerr, yerr



def make_bitprof_plot(w_dicts, saveto):
    su = '_'
    sp = ' '
    prof_bin_size = 2**7
    shade_bit = 512
    total_ps1 = 0
    swing_15s1 = 0
    swing_10s1 = 0
    swing_5s1 = 0
    total_ps2 = 0
    swing_15s2 = 0
    swing_10s2 = 0
    swing_5s2 = 0
    w_dicts_keys = sorted(w_dicts)

    profs, p_axs = plt.subplots(8, 2, sharey=True, sharex=False, figsize=(30, 20))
    p_axs = p_axs.ravel()

    for p_ax, wkey in zip(p_axs, w_dicts_keys):
        working_dict = w_dicts[wkey]
        array_mins = []
        array_maxs = []
        raftname = working_dict['RTM']
        sensor = working_dict['SNN']
        hdu = working_dict['hdu']
        imtype = working_dict['imtype']

        for run, array in working_dict['arrays'].items():
            array_mins.append(min(array))
            array_maxs.append(max(array))

        prof_xmin = min(array_mins)
        prof_xmin = prof_xmin - prof_xmin % prof_bin_size
        prof_xmax = max(array_maxs)
        prof_xmax = prof_xmax - prof_xmax % prof_bin_size
        prof_nx = (prof_xmax - prof_xmin) / prof_bin_size

        for pbit in range(3):
            prof_bit = 2**pbit
            for run, array in working_dict['arrays'].items():
                bit_on = (array & prof_bit) // prof_bit
                profile, total, swing_15, swing_10, swing_5 = mkProfile(p_ax, array, bit_on, '{} {} {}s bit'.format(run, working_dict['sensors'][run], prof_bit), nx = prof_nx, xmin = prof_xmin, xmax = prof_xmax) 
                if prof_bit == 1:
                    total_ps1 += total
                    swing_15s1 += swing_15 
                    swing_10s1 += swing_10 
                    swing_5s1 += swing_5 

                if prof_bit == 2:
                    total_ps2 += total
                    swing_15s2 += swing_15 
                    swing_10s2 += swing_10 
                    swing_5s2 += swing_5 


        p_ax.grid()
        p_ax.set_title('amp{}'.format(hdu))
        p_ax.set_xlim([prof_xmin, prof_xmax])
        p_ax.set_ylim([0.35, 0.65])
        
        p_range = np.arange(prof_xmin, prof_xmax + 1)
        bar_lims = []
        
        bit_flips = (p_range % shade_bit) == 0
        bar_lims = p_range[bit_flips].tolist()

        if len(bar_lims) == 0:
            if ((p_range[0] & shade_bit) // shade_bit) == 1:
                bar_lims.append(p_range[0])
                bar_lims.append(p_range[-1])
        else:
            if ((min(bar_lims) & shade_bit) // shade_bit) == 0:
                bar_lims.append(p_range[0])
            if ((max(bar_lims) & shade_bit) // shade_bit) == 1:
                bar_lims.append(p_range[-1])

        bar_color = 'g'
        bar_alpha = 0.10
        
        if len(bar_lims) > 0:
            bar_lims.sort()
            for i in range(len(bar_lims) // 2):
                p_ax.axvspan(bar_lims[2 * i], bar_lims[2 * i + 1], color = bar_color, alpha = bar_alpha)
    
    profs.suptitle(sp.join([raftname, sensor, imtype, 'Least Significant Bit Profiles']), fontsize = 16)
    profs.text(0.5, 0.0, 'Signal (adu)', ha = 'center')
    profs.text(0.0, 0.5, 'Bit Frequency in Bin', va = 'center', rotation = 'vertical')
    handles, labels = p_ax.get_legend_handles_labels()
    profs.legend(handles, labels, 'upper right', fontsize='xx-large')
    profs.tight_layout()
    profs.subplots_adjust(top = 0.96)

    fname = su.join([raftname, sensor, imtype, 'LS_bit_profiles_bin{}.png'.format(prof_bin_size)])
    plt.savefig(os.path.join(saveto, fname))
    plt.close()
    return total_ps1, swing_15s1, swing_10s1, swing_5s1, total_ps2, swing_15s2, swing_10s2, swing_5s2
 
 


def im_to_array(imfile, hdu):
    image = afw_image.ImageF(imfile, hdu)
    mask = afw_image.Mask(image.getDimensions())
    masked_image = afw_image.MaskedImageF(image, mask)
    md = afw_image.readMetadata(imfile, hdu)
    imsec = masked_image.Factory(masked_image, get_bbox(md.get('DATASEC')))
    values, mask, var = imsec.getArrays()
    values = values.flatten()
    values = values.astype(int)
    
    return values



def make_modplot(working_dict, mod, saveto):
    su = '_'
    sp = ' '
    raftname = working_dict['RTM']
    sensor = working_dict['SNN']
    hdu = working_dict['hdu']
    imtype = working_dict['imtype']

    for run in working_dict['arrays'].keys():
        im_array = working_dict['arrays'][run]
        values_mod = im_array % mod
        
        plt.figure(figsize = (12, 3))
        h_mod = plt.hist(values_mod, bins = mod)
        plt.ylabel('Counts')
        plt.xlabel(sp.join(['Signal mod', str(mod), '(adu)']))
        plt.title(sp.join([raftname, sensor, 'run', run, 'amp{}'.format(hdu), imtype, 'Pixel Value Distrubution Mod', str(mod)]))
        fname = su.join([raftname, run, sensor, imtype, 'amp{}_pixel_distributions_mod{}.png'.format(hdu, mod)]) 
        plt.savefig(os.path.join(saveto, fname))
        plt.close()



def make_amp_histogram(w_dicts, saveto):
    su = '_'
    sp = ' '
    for key, working_dict in w_dicts.items():
        raftname = working_dict['RTM']
        sensor = working_dict['SNN']
        hdu = working_dict['hdu']
        imtype = working_dict['imtype']
    
        for run, im_array in working_dict['arrays'].items():
            h_binr = int(max(im_array) - min(im_array))
            
            plt.figure(figsize = (16, 8))
            plt.xlabel('Signal (adu)')
            plt.ylabel('Counts')
            plt.hist(im_array, bins = h_binr)
            
            fname = su.join([raftname, run, sensor, imtype, 'amp{}_pixel_distributions.png'.format(hdu)])
            plt.title(sp.join([raftname, sensor, run, 'amp{} '.format(hdu), imtype, 'Distribution']))
            plt.savefig(os.path.join(saveto, fname))
            plt.close()



def sflat_ADC_analysis(runs, imtype, saveto):
    master_dict = make_master_dict(runs, imtype)
    total_profs1 = 0
    nswing1_15 = 0
    nswing1_10 = 0
    nswing1_5 = 0
    total_profs2 = 0
    nswing2_15 = 0
    nswing2_10 = 0
    nswing2_5 = 0


    for key_rtm, SNN_dict in master_dict.items():
    
        # Loop over sensors
        for key_SNN, sflat_dict in SNN_dict.items():
            for key_eotype, runs_dict in sflat_dict.items():
                
                # Directory to save the plots to
                save_dir = os.path.join(saveto, key_rtm, key_SNN, key_eotype)
                os.makedirs(save_dir, exist_ok = True)

                w_dicts = {}
    
                # Loop over amplifiers 
                for hdu in range(1, 17):
                    w_dicts['amp{:02d}'.format(hdu)] = make_working_dict(runs_dict, key_rtm, key_SNN, hdu, imtype)
    
                        
                    #for imod in range(4, 7):
                    #    mod = 2**imod
    
                    #    make_modplot(working_dict, mod, save_dir)

                #make_amp_histogram(w_dicts, save_dir)
    
                total_ps1, swing1_15, swing1_10, swing1_5, total_ps2, swing2_15, swing2_10, swing2_5  = make_bitprof_plot(w_dicts, save_dir)
                total_profs1 += total_ps1
                nswing1_15 += swing1_15
                nswing1_10 += swing1_10
                nswing1_5 += swing1_5
                total_profs2 += total_ps1
                nswing2_15 += swing2_15
                nswing2_10 += swing2_10
                nswing2_5 += swing2_5


                print('Wrote plots for {} {}'.format(key_rtm, key_SNN))

    print('Total 1s bit profiles analyzed: {}\nProfiles with swing between .1 and .15: {}\nProfiles with swing between .05' \
    ' and .1: {}\nProfiles with swing less than .05: {}'.format(total_profs1, nswing1_15, nswing1_10, nswing1_5))
    print('Total 2s bit profiles analyzed: {}\nProfiles with swing between .1 and .15: {}\nProfiles with swing between .05' \
    ' and .1: {}\nProfiles with swing less than .05: {}'.format(total_profs2, nswing2_15, nswing2_10, nswing2_5))




if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Analyze sflat files for given runs at amplifier level')
    parser.add_argument('runs', nargs="+", help='A list of runs (as strings) to analyze')
    parser.add_argument('--imtype', default='flat_H', help='image type to analyze. E.g. flat1, flat2, flat_H, flat_L. Default: \'flat_h\'')
    parser.add_argument('--saveto', default='/u/ec/adriansh/lsst/analysis/plots/adc_analysis', help='Where to ' \
    'save the plots. Default: \'/u/ec/adriansh/lsst/analysis/plots/adc_analysis\'')
    args = parser.parse_args()

    if args.imtype == 'flat_H' or args.imtype == 'flat_L':
        sflat_ADC_analysis(args.runs, args.imtype, args.saveto)

    else:
        print('invalid imtype')
        exit()
