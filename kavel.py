#!/usr/bin/env python
#
# Kavel - audio manipulation tool
#
# this file is released under Public Domain
#
#


import sys
import soundfile as sf

from numpy import *
from optparse import OptionParser, OptionGroup
from datetime import datetime

class Kavel:
    '''
    Kavel - audio manipulation tool
    'by Erik Johansson @ 2017 http://github.com/opengd/kavel'
    '''

    manipulation_orders = {
        "b": "braid",
        "r": "reverse",
        "rl": "reverse left channel",
        "rr": "reverse right channel",
        "ro": "reverse on frame",
        "mx": "max on frame slice",
        "mi": "min on frame slice",
        "av": "average on frame slice",
        "me": "median on frame slice",
        "c": "chop frames",
        "d": "duplicate frames",
        "dc": "duplicate counter",
    }

    default_manipulation_order = "b_ro_r_rl_rr_mx_mi_av_me_c_d_dc"

    @staticmethod
    def get_mainpulation_order(manipulation_order):
        '''
        Get manipulation order list from order string
        '''
        if manipulation_order == "":
            manipulation_order = Kavel.default_manipulation_order

        orders = []

        for order in manipulation_order.split('_'):
            if order in Kavel.manipulation_orders.keys():
                orders.append(order)

        if len(orders) < 1:
            orders = Kavel.default_manipulation_order.split('_')
        
        return orders

    @staticmethod
    def load_frames_from_file(
        filename, 
        start_frame, 
        end_frame, 
        reverse_input, 
        reverse_input_left, 
        reverse_input_right, 
        braid, 
        braid_on, 
        reverse_on,
        max_on_slice,
        min_on_slice,
        average_on_slice,
        median_on_slice,
        chop_frames,
        duplicate_frames, 
        duplicate_counter,
        manipulation_order
        ):
        '''
        Try to load frames from audio file and run manipulations
        '''
        try:
            print('\n#Input\n')
            print(sf.info(filename), True)
            
            sample_data, samplerate = sf.read(filename, start=start_frame, stop=end_frame)
        except:
            print('Error loading file: {!s} {!r}'.format(filename, sys.exc_info()[0]))
            return None
        else:
            print('frames read: {}'.format(len(sample_data)))
            print('start_frame: {}'.format(start_frame))
            print('end_frame: {}'.format(end_frame or len(sample_data)))

            orders = Kavel.get_mainpulation_order(manipulation_order)
                        
            print('Manipulation order: {!s}'.format('_'.join(orders)))

            print('Correcting input data layout: {} frames'.format(len(sample_data)))
            
            frames = Kavel.get_two_track_frame_list_from_pairs(sample_data)

            output_decorations = ''

            for order in orders:
                if order == "b" and braid:
                    output_decorations = output_decorations + '_b{}'.format(braid_on)
                    frames = Kavel.braid_frames(frames, braid_on)

                elif order == "ro" and reverse_on > 0:
                    output_decorations = output_decorations + '_ro{}'.format(reverse_on)
                    frames = Kavel.reverse_on_frame(frames, reverse_on)

                elif order == "r" and reverse_input:
                    output_decorations = output_decorations + '_r'
                    frames = Kavel.reverse_frames(frames)

                elif order == "rl" and reverse_input_left:
                    output_decorations = output_decorations + '_rl'
                    frames = Kavel.reverse_frames_left_channel(frames)

                elif order == "rr" and reverse_input_right:
                    output_decorations = output_decorations + '_rr'
                    frames = Kavel.reverse_frames_right_channel(frames)
                
                elif order == "mx" and max_on_slice > 0:
                    output_decorations = output_decorations + '_mx{}'.format(max_on_slice)
                    frames = Kavel.max_on_slice_frame(frames, max_on_slice)
                
                elif order == "mi" and min_on_slice > 0:
                    output_decorations = output_decorations + '_mi{}'.format(min_on_slice)
                    frames = Kavel.min_on_slice_frame(frames, min_on_slice)
                
                elif order == "av" and average_on_slice > 0:
                    output_decorations = output_decorations + '_av{}'.format(average_on_slice)
                    frames = Kavel.average_on_slice_frame(frames, average_on_slice)
                
                elif order == "me" and median_on_slice > 0:
                    output_decorations = output_decorations + '_me{}'.format(median_on_slice)
                    frames = Kavel.median_on_slice_frame(frames, median_on_slice)
                
                elif order == "c" and chop_frames > 0:
                    output_decorations = output_decorations + '_c{}'.format(chop_frames)
                    frames = Kavel.chop_every_on_frame(frames, chop_frames)
                
                elif order == "d" and duplicate_frames > 0 and duplicate_counter > 0:
                    output_decorations = output_decorations + '_d{}_dc{}'.format(duplicate_frames, duplicate_counter)
                    frames = Kavel.duplicate_every_on_slize_frame(frames, duplicate_frames, duplicate_counter)

            sample_data = array(frames)

            print('{} frames corrected'.format(len(sample_data[0])))

            return (samplerate, sample_data, output_decorations)
    
    @staticmethod
    def get_two_track_frame_list_from_pairs(frames):
        two_track_frame_list = [[], []]

        for a in frames:
            two_track_frame_list[0].append(a[0])
            two_track_frame_list[1].append(a[1])
        
        return two_track_frame_list
    
    @staticmethod
    def get_pair_frame_list_from_two_track(frames):
        pair_frame_list = []
        for i in range(len(frames[0])):
            pair_frame_list.append([frames[0][i], frames[0][i]])
        
        return pair_frame_list

    @staticmethod
    def max_on_slice_frame(frames, max_on_slice):
        print('Max on slice of {} frame'.format(max_on_slice))
        r_buf = [[], []]
        slice_counter = 0
        i = 0
        l = []
        r = []
        while i < len(frames[0]):
            l.append(frames[0][i])
            r.append(frames[1][i])
            i = i + 1
            slice_counter = slice_counter + 1
            if (i == len(frames[0])) or slice_counter == max_on_slice:
                max_frame_l = max(l)
                max_frame_r = max(r)
                for f in range(len(l)):
                    l[f] = max_frame_l
                    r[f] = max_frame_r

                r_buf[0].extend(l)
                r_buf[1].extend(r)
                l = []
                r = []
                slice_counter = 0

        return r_buf
    
    @staticmethod
    def min_on_slice_frame(frames, min_on_slice):
        print('Min on slice of {} frame'.format(min_on_slice))
        r_buf = [[], []]
        slice_counter = 0
        i = 0
        l = []
        r = []
        while i < len(frames[0]):
            l.append(frames[0][i])
            r.append(frames[1][i])
            i = i + 1
            slice_counter = slice_counter + 1
            if (i == len(frames[0])) or slice_counter == min_on_slice:
                min_frame_l = min(l)
                min_frame_r = min(r)
                for f in range(len(l)):
                    l[f] = min_frame_l
                    r[f] = min_frame_r

                r_buf[0].extend(l)
                r_buf[1].extend(r)
                l = []
                r = []
                slice_counter = 0

        return r_buf
    
    @staticmethod
    def average_on_slice_frame(frames, average_on_slice):
        print('Average on slice of {} frame'.format(average_on_slice))
        r_buf = [[], []]
        slice_counter = 0
        i = 0
        l = []
        r = []
        while i < len(frames[0]):
            l.append(frames[0][i])
            r.append(frames[1][i])
            i = i + 1
            slice_counter = slice_counter + 1
            if (i == len(frames[0])) or slice_counter == average_on_slice:
                average_frame_l = sum(l) / len(l)
                average_frame_r = sum(r) / len(r)
                for f in range(len(l)):
                    l[f] = average_frame_l
                    r[f] = average_frame_r

                r_buf[0].extend(l)
                r_buf[1].extend(r)
                l = []
                r = []
                slice_counter = 0

        return r_buf

    @staticmethod
    def median_on_slice_frame(frames, on_slice):
        print('Median on slice of {} frame'.format(on_slice))
        r_buf = [[], []]
        slice_counter = 0
        i = 0
        l = []
        r = []
        while i < len(frames[0]):
            l.append(frames[0][i])
            r.append(frames[1][i])
            i = i + 1
            slice_counter = slice_counter + 1
            if (i == len(frames[0])) or slice_counter == on_slice:
                calc_frame_l = median(l)
                calc_frame_r = median(r)
                for f in range(len(l)):
                    l[f] = calc_frame_l
                    r[f] = calc_frame_r

                r_buf[0].extend(l)
                r_buf[1].extend(r)
                l = []
                r = []
                slice_counter = 0

        return r_buf
    
    @staticmethod
    def chop_every_on_frame(frames, chop_frames):
        print('Chop on slice of {} frame'.format(chop_frames))
        r_buf = [[], []]
        chop_counter = 0
        for i in range(len(frames[0])):
            if chop_counter != chop_frames:
                r_buf[0].append(frames[0][i])
                r_buf[1].append(frames[1][i])
                chop_counter += 1
            else:
                chop_counter = 0
        
        return r_buf

    @staticmethod
    def duplicate_every_on_slize_frame(frames, duplicate_frames, duplications = 1):
        print('Duplicate on slice of {} frame, do {} duplications'.format(duplicate_frames, duplications))
        r_buf = [[], []]
        dup_buf = [[], []]
        duplicate_counter = 0
        for i in range(len(frames[0])):
            r_buf[0].append(frames[0][i])
            r_buf[1].append(frames[1][i])
            dup_buf[0].append(frames[0][i])
            dup_buf[1].append(frames[1][i])
            if i == len(frames[0]) or duplicate_counter == duplicate_frames:
                for r in range(duplications):
                    r_buf[0].extend(dup_buf[0])
                    r_buf[1].extend(dup_buf[1])

                dup_buf = [[], []]
                duplicate_counter = 0
            else:
                duplicate_counter += 1
        
        return r_buf

    @staticmethod
    def braid_frames(frames, braid_on):
        print('Braid frames, change on every {} frame'.format(braid_on))
        ar = [[], []]
        i = 0
        b_c_l = 0
        b_c_r = 1
        #for a in frames:
        for f in range(len(frames[0])):
            #ar[b_c_l].append(a[0])
            #ar[b_c_r].append(a[1])
            ar[b_c_l].append(frames[0][f])
            ar[b_c_r].append(frames[1][f])
            i = i + 1
            if i == braid_on:
                if b_c_l == 0:
                    b_c_l = 1
                    b_c_r = 0
                else:
                    b_c_l = 0
                    b_c_r = 1
                i = 0
        return ar
    
    @staticmethod
    def reverse_on_frame(frames, reverse_on):
        print('Reverse frames, change on every {} frame'.format(reverse_on))
        r_buf = [[], []]
        r_o = 0
        i = 0
        do_revers = False
        l = []
        r = []
        while i < len(frames[0]):
            l.append(frames[0][i])
            r.append(frames[1][i])
            i = i + 1
            r_o = r_o + 1
            if (i == len(frames[0])) or r_o == reverse_on:
                if do_revers:
                    l = l[::-1]
                    r = r[::-1]
                    do_revers = False
                else:
                    do_revers = True
                
                r_buf[0].extend(l)
                r_buf[1].extend(r)
                l = []
                r = []
                r_o = 0

        return r_buf
    
    @staticmethod
    def reverse_frames(frames):
        print('Reverse input file')
        frames[0] = frames[0][::-1]
        frames[1] = frames[1][::-1]
        return frames
    
    @staticmethod
    def reverse_frames_left_channel(frames):
        print('Reverse left channel on input file')
        frames[0] = frames[0][::-1]
        return frames
    
    @staticmethod
    def reverse_frames_right_channel(frames):
        print('Reverse right channel on input file')
        frames[1] = frames[1][::-1]
        return frames

class Paulstretch:
    '''
    This is Paulstretch , Python version
    by Nasca Octavian PAUL, Targu Mures, Romania
    http://www.paulnasca.com/

    Requirements: Numpy, Scipy

    Original version with GUI: 
    http://hypermammut.sourceforge.net/paulstretch/

    For start, I recomand to use "paulstretch_stereo.py".

    The "paulstretch_mono.py" is a very simple test implementation of the Paulstretch algorithm.
    The "paulstretch_newmethod.py" is a extended/slower Paulstretch algorithm which has onset detection.

    "paulstretch_steps.png" describes the steps of Paulstretch algorithm graphically.

    The Paulstretch algorithm is released under Public Domain.
    '''
    @staticmethod
    def optimize_windowsize(n):
        orig_n = n
        while True:
            n = orig_n
            while (n%2) == 0:
                n /= 2
            while (n%3) == 0:
                n /= 3
            while (n%5) == 0:
                n /= 5

            if n < 2:
                break
            orig_n += 1
        return orig_n
    
    @staticmethod
    def paulstretch(samplerate, smp, stretch, windowsize_seconds, onset_level, outfilename):
        
        nchannels = smp.shape[0]

        print('output file: {!s}\nsamplerate: {} Hz\nchannels: {}\n'.format(outfilename, samplerate, nchannels))
        outfile = sf.SoundFile(outfilename, 'w', samplerate, nchannels)

        #make sure that windowsize is even and larger than 16
        windowsize = int(windowsize_seconds*samplerate)
        if windowsize < 16:
            windowsize = 16
        windowsize = Paulstretch.optimize_windowsize(windowsize)
        windowsize = int(windowsize/2) * 2
        half_windowsize = int(windowsize/2)

        #correct the end of the smp
        nsamples = smp.shape[1]
        end_size = int(samplerate*0.05)
        if end_size < 16:
            end_size = 16

        smp[:,nsamples-end_size:nsamples] *= linspace(1, 0, end_size)

        
        #compute the displacement inside the input file
        start_pos = 0.0
        displace_pos = windowsize * 0.5

        #create Hann window
        window = 0.5 - cos(arange(windowsize, dtype='float')*2.0* pi/(windowsize-1)) * 0.5

        old_windowed_buf = zeros((2, windowsize))
        hinv_sqrt2 = (1+sqrt(0.5)) * 0.5
        hinv_buf = 2.0 *(hinv_sqrt2-(1.0-hinv_sqrt2) * cos(arange(half_windowsize, dtype='float')*2.0*pi/half_windowsize))/hinv_sqrt2

        freqs = zeros((2, half_windowsize+1))
        old_freqs = freqs

        num_bins_scaled_freq = 32
        freqs_scaled = zeros(num_bins_scaled_freq)
        old_freqs_scaled = freqs_scaled

        displace_tick = 0.0
        displace_tick_increase = 1.0/stretch
        if displace_tick_increase > 1.0:
            displace_tick_increase = 1.0
        extra_onset_time_credit = 0.0
        get_next_buf = True
        start_time = datetime.now()
        
        while True:
            if get_next_buf:
                old_freqs = freqs
                old_freqs_scaled = freqs_scaled

                #get the windowed buffer
                istart_pos =int(floor(start_pos))
                buf = smp[:, istart_pos:istart_pos+windowsize]
                if buf.shape[1] < windowsize:
                    buf = append(buf, zeros((2, windowsize-buf.shape[1])), 1)
                buf= buf * window
        
                #get the amplitudes of the frequency components and discard the phases
                freqs = abs(fft.rfft(buf))

                #scale down the spectrum to detect onsets
                freqs_len = freqs.shape[1]
                if num_bins_scaled_freq < freqs_len:
                    freqs_len_div = freqs_len // num_bins_scaled_freq
                    new_freqs_len = freqs_len_div * num_bins_scaled_freq
                    freqs_scaled = mean(mean(freqs, 0)[:new_freqs_len].reshape([num_bins_scaled_freq, freqs_len_div]), 1)
                else:
                    freqs_scaled = zeros(num_bins_scaled_freq)


                #process onsets
                m = 2.0*mean(freqs_scaled-old_freqs_scaled)/(mean(abs(old_freqs_scaled))+1e-3)
                if m < 0.0:
                    m = 0.0
                if m > 1.0:
                    m = 1.0
                if m > onset_level:
                    displace_tick = 1.0
                    extra_onset_time_credit += 1.0

            cfreqs = (freqs*displace_tick) + (old_freqs*(1.0-displace_tick))

            #randomize the phases by multiplication with a random complex number with modulus=1
            ph = random.uniform(0, 2*pi, (nchannels, cfreqs.shape[1]))*1j
            cfreqs = cfreqs*exp(ph)

            #do the inverse FFT 
            buf = fft.irfft(cfreqs)

            #window again the output buffer
            buf *= window

            #overlap-add the output
            output = buf[:, 0:half_windowsize]+old_windowed_buf[:, half_windowsize:windowsize]
            old_windowed_buf = buf

            #remove the resulted amplitude modulation
            output *= hinv_buf
            
            #clamp the values to -1..1 
            output[output>1.0] = 1.0
            output[output<-1.0] = -1.0

            #write the output to wav file
            d = int16(output.ravel(1)*32767.0)
            d = array_split(d, len(d)/2)

            outfile.write(d)

            if get_next_buf:
                start_pos += displace_pos

            get_next_buf=False

            if start_pos >= nsamples:
                print("100 %")
                break
            sys.stdout.write('{} % \r'.format(int(100.0*start_pos/nsamples)))
            sys.stdout.flush()

            
            if extra_onset_time_credit <= 0.0:
                displace_tick += displace_tick_increase
            else:
                credit_get = 0.5*displace_tick_increase #this must be less than displace_tick_increase
                extra_onset_time_credit -= credit_get
                if extra_onset_time_credit < 0:
                    extra_onset_time_credit = 0
                displace_tick += displace_tick_increase-credit_get

            if displace_tick >= 1.0:
                displace_tick =displace_tick % 1.0
                get_next_buf = True

        outfile.close()
        print('Stretching completed in: {}'.format(datetime.now() - start_time))

########################################

def print_inputfile_stats(filename):
    try:
        wavedata, samplerate = sf.read(filename)
        print(sf.info(filename), True)
        print('frames: {}'.format(len(wavedata)))
    except:
        print('Error loading file: {!s} {!r}'.format(filename, sys.exc_info()[0]))
        return None

def get_output_file_name(input_file, output_file, output_decorations, output_name):
    '''
    Return a suiting output file name from user input 
    or manipulation
    '''
    if output_file == "":
        infile_split = input_file.split(".")
        if output_name == "":
            outputfile = infile_split[0]
        outputfile = outputfile + output_name + output_decorations + '.' + infile_split[len(infile_split)-1]
    else:
        return output_file

    return outputfile

class MyOptionGroup(OptionGroup):
    def format_description(self, formatter):
        '''
        text_width = max(self.width - self.current_indent, 11)
        indent = " "*self.current_indent
        return textwrap.fill(self.description,
                             text_width,
                             initial_indent=indent,
                             subsequent_indent=indent)
        '''
        return self.description

if __name__ == "__main__":
    print('Kavel - audio manipulation tool')
    print('by Erik Johansson @ 2017 http://github.com/opengd/kavel')

    parser = OptionParser(usage="usage: %prog [options] input_file output_file(optional)")

    paulstrech_options = OptionGroup(parser, 'Paulstrech Options')
    paulstrech_options.add_option("-s", "--stretch", action="store_true", dest="stretch", help="Stretch using Paul's Extreme Sound Stretch (Paulstretch)", default=False)
    paulstrech_options.add_option("--stretch_amount", dest="stretch_amount", help="stretch amount (1.0 = no stretch), above 0.0", type="float", default=8.0)
    paulstrech_options.add_option("--window_size", dest="window_size", help="window size (seconds), above 0.001", type="float", default=0.25)
    paulstrech_options.add_option("--onset", dest="onset", help="onset sensitivity (0.0=max,1.0=min)", type="float", default=10.0)
    parser.add_option_group(paulstrech_options)

    man_ordes = ''
    '''
    for cmd, desc in Kavel.manipulation_orders.items():
        man_ordes = '{}\n{} - {}'.format(man_ordes, cmd, desc)
    '''
    input_mani_options = MyOptionGroup(parser, 'Input File Manipulation Options', man_ordes)
    input_mani_options.add_option("-t", "--start_frame", dest="start_frame", help="Start read on frame", type="int", default=0)
    input_mani_options.add_option("-e", "--end_frame", dest="end_frame", help="End read on frame", type="int", default=None)
    input_mani_options.add_option("-r", "--reverse", action="store_true", dest="reverse_input", help="Reverse input file", default=False)
    input_mani_options.add_option("--reverse_left", action="store_true", dest="reverse_input_left", help="Reverse left channel on input file", default=False)
    input_mani_options.add_option("--reverse_right", action="store_true", dest="reverse_input_right", help="Reverse right channel on input file", default=False)
    input_mani_options.add_option("-b","--braid", action="store_true", dest="braid", help="Braid right and left channels by frame", default=False)
    input_mani_options.add_option("--braid_on", dest="braid_on", help="Braid on frame (default=1), must be used with braid option", type="int", default=1)
    input_mani_options.add_option("--reverse_on", dest="reverse_on", help="Reverse on frame (default=1)", type="int", default=0)
    input_mani_options.add_option("--max_on_slice", dest="max_on_slice", help="Change all frames on slice to the max value in frames", type="int", default=0)
    input_mani_options.add_option("--min_on_slice", dest="min_on_slice", help="Change all frames on slice to the min value in frames", type="int", default=0)
    input_mani_options.add_option("--average_on_slice", dest="average_on_slice", help="Change all frames on slice to the average value in frame slice", type="int", default=0)
    input_mani_options.add_option("--median_on_slice", dest="median_on_slice", help="Change all frames on slice to the median value in frame slice", type="int", default=0)
    input_mani_options.add_option("--chop", dest="chop_frames", help="Chop frames", type="int", default=0)
    input_mani_options.add_option("--duplicate", dest="duplicate_frames", help="Duplicate frames, value is slice size", type="int", default=0)
    input_mani_options.add_option("--duplicate_counter", dest="duplicate_counter", help="Do number of duplications frames, default is 1", type="int", default=1)
    parser.add_option_group(input_mani_options)

    parser.add_option(
        "-m", 
        "--manipulation_order", 
        dest="manipulation_order", 
        type="string", 
        help="Manipulation order, default is {!r}".format(Kavel.default_manipulation_order), 
        default=Kavel.default_manipulation_order
        )
    parser.add_option("--show_manipulation_order_list", action="store_true", dest="show_manipulation_order_list", help="Print manipulation order helper list and then exit", default=False)
    parser.add_option("-i", "--input_file_stat", action="store_true", dest="input_file_stat", help="Print inputfile stat and then exit", default=False)
    parser.add_option("-l", "--list_supported_types", action="store_true", dest="list_supported_types", help="List all supported input file types and then exit", default=False)
    parser.add_option("--output_name", dest="output_name", help="Output name added to decorations if not on command line", type="string", default="")

    (options, args) = parser.parse_args()

    if options.show_manipulation_order_list:
        print('\nManipulation order list:\n')
        for key, desc in Kavel.manipulation_orders.items():
            print('{} - {}'.format(key, desc))
        
        print('\nDefault order: {}'.format(Kavel.default_manipulation_order))
        sys.exit(0)

    elif options.list_supported_types:
        print('Supported file types:\n')
        for file_type, desc in sf.available_formats().items():
            print('{} - {}'.format(file_type, desc))
        sys.exit(0)

    if len(args) > 0 and options.input_file_stat:
        print_inputfile_stats(args[0])
        sys.exit(0) 

    if len(args) < 1:
        print("Error in command line parameters. Run this program with --help for help.")
        sys.exit(1)
    elif len(args) == 1:
        outputfile = ""
    else:
        outputfile = args[1]

    (samplerate, smp, output_decorations) = Kavel.load_frames_from_file(
        args[0], 
        options.start_frame, 
        options.end_frame, 
        options.reverse_input, 
        options.reverse_input_left, 
        options.reverse_input_right, 
        options.braid,
        options.braid_on,
        options.reverse_on,
        options.max_on_slice,
        options.min_on_slice,
        options.average_on_slice,
        options.median_on_slice,
        options.chop_frames,
        options.duplicate_frames,
        options.duplicate_counter,
        options.manipulation_order
        )

    # Add a kavel signature to ouputfile if non is in args
    output_decorations = output_decorations + '_k'

    if options.stretch and (options.stretch_amount > 0.0) and (options.window_size > 0.001):
        # Stretch audio data using paulstretch
        output_decorations = output_decorations + '_s{}_w{}_o{}'.format(options.stretch_amount, options.window_size, options.onset)
        
        print('\n#Streching\n')

        print("Paul's Extreme Sound Stretch (Paulstretch) - Python version 20141220")
        print("by Nasca Octavian PAUL, Targu Mures, Romania\n")
        
        print('stretch amount = {}'.format(options.stretch_amount))
        print('window size = {} seconds'.format(options.window_size))
        print('onset sensitivity = {}'.format(options.onset))
        
        outputfile = get_output_file_name(args[0], outputfile, output_decorations, options.output_name)

        Paulstretch.paulstretch(samplerate, smp, options.stretch_amount, options.window_size, options.onset, outputfile)
    else:
        # Output file after manipulation
        frames = Kavel.get_pair_frame_list_from_two_track(smp)

        outputfile = get_output_file_name(args[0], outputfile, output_decorations, options.output_name)

        sf.write(outputfile, frames, samplerate, 'PCM_16')
    
    print('\n#Output\n')
    print(sf.info(outputfile), True)
