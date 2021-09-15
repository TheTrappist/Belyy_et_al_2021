# -*- coding: utf-8 -*-
"""

This module contains a collection of functions and tools for performing
two-color correlation analysis on single-particle trajectories.

Written by Vladislav Belyy (UCSF).

"""
import os
import numpy as np
from scipy import stats
import pandas as pd
import warnings
import json
import matplotlib.pylab as plt
import seaborn as sns
import parse_trackmate as pt

def read_analysis_params (settings_file, save_dir_reports=None,
                                                print_summary=False):
    """Read analysis parameters from JSON settings file.

    Args:
        settings_file (str): Path to the settings file, either relative to the
            location of the settings file or absolute.
        save_dir_reports (str): Base directory where reports should be saved.
            This is used to generate individual subfolders using the relative
            paths stored in the settings file. If None, no subfolders are
            generated. Defaults to None.
        print_summary (bool): print some of the key parameters loaded from the
            file. Defaults to False.

    Returns:
        conditions (dict):
            'condition' (array of str): All paths to data folders corresponding
                to the corresponding condition.
        params (dict): Analysis parameters taken from the JSON settings file.
            'do_corr_analysis' (bool): Perform correlation analysis if True.
            'do_int_analysis' (bool): Perform intensity analysis if True.
            'corr_settings' (dict): Correlation analysis settings.
            'int_settings' (dict): Intensity analysis settings.
            'window' (int): In frames, length of sliding window for correlation
                analysis. Only used in correlation analysis.
            'pcc_cutoff' (float): stringency of the pearson's correlation
                coefficient. Only used in correlation analysis.
            'save_dir' (path): Base directory for saving results.
            'save_dir_data' (path): Base directory for saving filtered tracks.
            'raw_json' (dict): the entire contents of the json file with no
                processing.
    """

    # Read the json settings file
    with open(settings_file) as fd:
        json_data = json.load(fd)

    base_dir = os.path.split(settings_file)[0]
    base_dir = os.path.abspath(base_dir) # Convert relative path to absolute

    # Build lists of folders containing tracks for each condition
    conditions = {}
    for condition, folders in json_data['conditions'].items():
        if type(folders) is not list: folders = [folders]
        paths = []
        for folder in folders:
            full_path = os.path.join(base_dir, folder)
            paths = paths + [os.path.normpath(full_path)]
        conditions[condition] = paths

    # Build paths for output files
    params = {}
    if save_dir_reports:
        save_dir = os.path.join(save_dir_reports, json_data['save_dir_reports'])
        params['save_dir'] = os.path.normpath(save_dir)
    save_dir_data = os.path.join(base_dir, json_data['save_dir_filt_data'])
    params['save_dir_data'] = os.path.normpath(save_dir_data)

    # Load the analysis settings
    params['do_corr_analysis'] = False
    params['do_int_analysis'] = False
    params['corr_settings'] = None
    params['int_settings'] = None
    params['window'] = None
    params['pcc_cutoff'] = None

    settings_loaded = False
    if "correlation_analysis" in json_data:
        corr_settings = json_data['correlation_analysis']
        params['do_corr_analysis'] = corr_settings['do_correlation_analysis']
        params['window'] = corr_settings['window']
        params['pcc_cutoff'] = corr_settings['pcc_cutoff']
        params['corr_settings'] = corr_settings
        settings_loaded = True
    if "intensity_analysis" in json_data:
        int_settings = json_data['intensity_analysis']
        params['do_int_analysis'] = int_settings['do_intensity_analysis']
        params['int_settings'] = int_settings
        settings_loaded = True
    if not settings_loaded: # Backwards compatibility with older settings files
        params['do_corr_analysis'] = True
        params['window'] = json_data['window']
        params['pcc_cutoff'] = json_data['pcc_cutoff']

    params['raw_json'] = json_data

    if print_summary:
        params_to_print = dict(params)
        params_to_print.pop('raw_json')
        print(params_to_print)

    return conditions, params

def overlap_mask_2channel (C1_min, C1_max, C2_min, C2_max, min_overlap=0.0):
    """Compute overlap matrix for arbitrary data from two channels (e.g. time).

    Args:
        C1_min (array-like): Minimum values in channel 1.
        C2_min (array-like): Minimum values in channel 2.
        C1_max (array-like): Maximum values in channel 1.
        C2_max (array-like): Maximum values in channel 2.
        min_overlap (float): stringency of overlap requirement. If greater than
            0, tracks must overlap by at least this much for the overlap to
            count (useful, e.g., for temporal overlap requiring multiple
            frames). If less than zero, allows tracks to be shifted by at most
            this much and still count as overlapping (e.g., for less stringent
            spatial overlap calculations). Defaults to 0.

    Returns:
        overlap_out (dict):
            'mask' (bool array): n-by-m matrix (n being the number of tracks
                in channel 1 and m the number of tracks in channel 2), where
                True indicates positive overlap between the corresponding
                tracks.
            'duration' (numpy array): n-by-m matrix, where each cell indicates
                the duration of the overlap between the corresponding tracks.
            'idx' (numpy array): 2-by-x matrix, where x is the total number
                of positive pairwise overlaps between tracks in channel 1 and
                channel 2. For each pair of overlapping tracks, the original
                index of the track in channel 2 is stored.
    """

    n = len(C1_min)
    m = len(C2_min)

    # build matrices of start and end track times for finding overlap
    min_arr_C1 = np.transpose(np.broadcast_to(C1_min, (m, n)))
    min_arr_C2 = np.broadcast_to(C2_min, (n, m))
    max_arr_C1 = np.transpose(np.broadcast_to(C1_max, (m, n)))
    max_arr_C2 = np.broadcast_to(C2_max, (n, m))

    overlap_min_raw = np.maximum(min_arr_C1, min_arr_C2)
    overlap_max_raw = np.minimum(max_arr_C1, max_arr_C2)
    overlap_length_raw = np.subtract(overlap_max_raw, overlap_min_raw)

    # apply minimum overlap cutoff
    overlap_mask = np.greater_equal(overlap_length_raw, min_overlap)

    overlap_length = np.multiply(overlap_length_raw, overlap_mask)

    # Find indices of tracks that pass the overlap threshold
    overlap_idx = np.nonzero(overlap_mask)

    overlap_out = {'mask' : overlap_mask,
                   'duration' : overlap_length,
                   'idx' : overlap_idx}

    return overlap_out


def get_overlap_in_time (df_C1, df_C2, min_overlap_frames=3):
    """Compute temporal overlap between two sets of particle tracks.

    Assume that data come from channels 1 and 2, with n and m trajectories,
    respectively.

    Args:
        df_C1 (Pandas dataframe): track data from channel 1.
        df_C2 (Pandas dataframe): track data from channel 2.
        min_overlap (int): minimum number of frames to count tracks as
            overlapping in time. Defaults to 3.

    Returns:
        overlap_t (dict):
            'mask' (bool array): n-by-m matrix where True indicates positive
                overlap between the corresponding tracks.
            'duration' (numpy array): n-by-m matrix, where each cell indicates
                the duration of the overlap between the corresponding tracks.
            'idx' (numpy array): 2-by-x matrix, where x is the total number
                of positive pairwise overlaps between tracks in channel 1 and
                channel 2. For each pair of overlapping tracks, the original
                index of the track in channel 2 is stored.
    """

    frames_C1 = get_track_start_end_frames(df_C1)
    frames_C2 = get_track_start_end_frames(df_C2)

    overlap_t = overlap_mask_2channel (frames_C1['t_start'], frames_C1['t_end'],
                                       frames_C2['t_start'], frames_C2['t_end'],
                                       min_overlap=min_overlap_frames)

    return overlap_t

def get_overlap_in_space (df_C1, df_C2, min_overlap=0.0):
    """Compute temporal overlap between two sets of particle tracks.

    Assume that data come from channels 1 and 2, with n and m trajectories,
    respectively.

    Args:
        df_C1 (Pandas dataframe): track data from channel 1.
        df_C2 (Pandas dataframe): track data from channel 2.
        min_overlap (float): stringency of overlap requirement. If greater than
            0, tracks must overlap by at least this much for the overlap to
            count. If less than zero, allows tracks to be shifted by at most
            this much and still count as overlapping (e.g., for less stringent
            spatial overlap calculations). Defaults to 0.

    Returns:
        overlap_x (dict): overlap along the x-coordinate, with these keys:
            'mask' (bool array): n-by-m matrix where True indicates positive
                overlap between the corresponding tracks.
            'duration' (numpy array): n-by-m matrix, where each cell indicates
                the duration of the overlap between the corresponding tracks.
            'idx' (numpy array): 2-by-x matrix, where x is the total number
                of positive pairwise overlaps between tracks in channel 1 and
                channel 2. For each pair of overlapping tracks, the original
                index of the track in channel 1 and 2 is stored.
        overlap_y (dict): same as overlap_y, but for the y-coordinate.
        p_vs_t (dict): position vs. time matrices for each channel. Keys:
            'C1_t_x' (numpy array): n-by-t matrix of x-positions for channel 1
            'C1_t_y' (numpy array): n-by-t matrix of y-positions for channel 1
            'C2_t_x' (numpy array): n-by-t matrix of x-positions for channel 2
            'C2_t_y' (numpy array): n-by-t matrix of y-positions for channel 2


    """
    # Arrange all tracks in a n-by-t matrix with NaNs, where t is
    # the total number of time points and n is the number of tracks
    num_frames = int(max(df_C1['t'].max(), df_C2['t'].max()))+1
    C1_t_x, C1_t_y = build_pos_vs_time_matrix(df_C1, num_frames)
    C2_t_x, C2_t_y = build_pos_vs_time_matrix(df_C2, num_frames)

    # Calculate minimum and maximum x and y values of each track
    min_x_C1 = np.nanmin(C1_t_x, axis=1)
    min_x_C2 = np.nanmin(C2_t_x, axis=1)
    min_y_C1 = np.nanmin(C1_t_y, axis=1)
    min_y_C2 = np.nanmin(C2_t_y, axis=1)

    max_x_C1 = np.nanmax(C1_t_x, axis=1)
    max_x_C2 = np.nanmax(C2_t_x, axis=1)
    max_y_C1 = np.nanmax(C1_t_y, axis=1)
    max_y_C2 = np.nanmax(C2_t_y, axis=1)

    overlap_x = overlap_mask_2channel(min_x_C1, max_x_C1, min_x_C2, max_x_C2,
                                      min_overlap=min_overlap)
    overlap_y = overlap_mask_2channel(min_y_C1, max_y_C1, min_y_C2, max_y_C2,
                                      min_overlap=min_overlap)

    p_vs_t = {'C1_t_x' : C1_t_x,
              'C1_t_y' : C1_t_y,
              'C2_t_x' : C2_t_x,
              'C2_t_y' : C2_t_y}

    return overlap_x, overlap_y, p_vs_t

def build_pos_vs_time_matrix (df_tracks, num_frames, plot=False):
    """Return x- and y- coordinates of n tracks as n-by-t matrices with nans.

    Useful for computing spatial overlap and displacements. Builds one matrix
    for each coordinate (x and y), where each track in df_track is a row and
    each frame of the movie is a column. The matrix is populated with position
    values that are available; whenever a track does not extend into a given
    frame, NaN is assigned instead.

    Args:
        df_tracks (Pandas dataframe): track data.
        num_frames (int): Number of frames on the t-axis. Not simply taking
            the maximum frame in df_tracks because this function is often used
            to compare two sets of tracks, where the output arrays must have
            the same time dimension.
        plot (bool): Plot the output track matrix for debugging. Defaults to
            False.

    Returns:
        track_x_by_frame (numpy array): n-by-t matrix of track x-coordinates.
        track_y_by_frame (numpy array): n-by-t matrix of track y-coordinates.
    """

    track_starts = np.diff(df_tracks['track_ID'], prepend=-1).astype(bool)
    track_starts[0] = False # Make sure the index of the first track is 0
    tracks = np.cumsum(track_starts)
    num_tracks = tracks[-1] + 1

    track_x_by_frame = np.full((num_tracks, num_frames), np.nan)
    track_y_by_frame = np.copy(track_x_by_frame)
    track_x_by_frame[tracks, df_tracks['t']] = df_tracks['x']
    track_y_by_frame[tracks, df_tracks['t']] = df_tracks['y']

    # Plot to test for sanity
    if plot:
        fig, ax = plt.subplots(1,1)
        fig.tight_layout(pad=2)
        for i in range(num_tracks):
            ax.plot(track_x_by_frame[i,:], track_y_by_frame[i,:])


    return track_x_by_frame, track_y_by_frame

def windowed_pearson (c1, c2, window):
    """Calculate Pearson's correlation coeffient with a sliding window.

    Compares pairwise position trajectories (e.g. along the x- or y-axis)
    between two channels by computing the Pearson's correlation coefficient
    with a sliding window. The inputs, c1 in c2, contain position data of
    aligned particle tracks (i.e. each row in c1 is a track corresponding to
    an track in c2). Returns an n-by-m matrix of Pearson's correlation
    coefficients, where n is the number of trajectories in each channel and m
    is the number of windows that fit into the length of the trajectory (e.g.
    for a 5-frame trajectory, there are three possible window positions and the
    second dimension of the output will therefore be equal to 3). The sliding
    window always moves by 1 frame. The function handles NaNs (missing frame
    data) in input tracks gracefully; the output only contains non-NaN values
    in locations where neither of the two tracks being compared contain NaNs in
    the entire window.

    Pearson's correlation coefficient for paired data {(x1, y1)...(xn, yn)}:
    corr_pearson = sum(xi-mean(x)*(yi-mean(y))) / sqrt(sum((xi-mean(x))^2) *
    sum((yi-mean(y))^2)).

    Args:
        c1 (numpy array): position data from channel 1, n-by-t array (n is
           number of trajectories and t is number of frames). Points for which
           positions are not available should be filled with NaNs.
        c2 (numpy array): position data from channel 2. Must have the same
           dimensions as c1.
        window (int) : size (in frames) of the sliding window

    Returns:
        corr_pearson (numpy array): n-by-m matrix of Pearson's correlation
            coefficients, where n is the number of trajectories in each channel
            and m is the number of windows that fit into the length of the
            trajectory. Only contains real values in positions where the
            correlation coefficient is well-defined (i.e. no missing data in
            either of the two tracks throughout the entire window). Contains
            NaNs elsewhere.
    """

    def normalize_sliding_window (a, window):
        """Efficiently calculate normalized sliding window trajectories"""

        t = np.shape(a)[1] # number of frames in data array
        # prepare fancy array indices
        indexer_window = np.arange(window).reshape(1,-1)
        indexer_frame = np.arange(t-window+1).reshape(-1,1)
        indexer = indexer_window + indexer_frame
        indexer2 = np.zeros_like(indexer_window) + indexer_frame

        # recast data into a 3-dimensional array, with dimensions n,m,t
        # (n: # of trajectories, m: # of windows, t: frames per window)
        windowed = a[:,indexer]

        # ignore warnings from calculating means on slices of all nan's
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            wind_means = np.nanmean(windowed, axis=2)

        # subtract mean values to normalize "a"
        expanded_means = wind_means[:, indexer2]
        wind_norm = windowed - expanded_means
        return wind_norm

    c1_norm = normalize_sliding_window(c1, window)
    c2_norm = normalize_sliding_window(c2, window)

    prod = np.multiply (c1_norm, c2_norm)

    sum_c1_sq = np.sum(np.square(c1_norm), axis=2)
    sum_c2_sq = np.sum(np.square(c2_norm), axis=2)

    # Calculate Pearson's correlation coefficient
    numerator = np.sum(prod, axis=2)
    denominator = np.sqrt(np.multiply(sum_c1_sq, sum_c2_sq))
    corr_pearson = np.divide(numerator, denominator,
                             out=np.full_like(numerator, np.nan),
                             where=(denominator!=0))

    return corr_pearson

def compute_track_intensities (df_tracks, int_channel, analysis_settings={},
                               min_frames=4, last_frame=50):
    """Compute fluorescence intensities in a desired channel based on tracks.

    Args:
        df_tracks (Pandas dataframe): track data (can come from channel of
            either color)
        int_channel (str): which channel(s) to use for intensity calculation?
            Valid values are 'C1', 'C2', or 'both'
        analysis_settings (dict): intensity analysis settings
        min_frames (int): minimum number of frames in track for it to count
        last_frame (int): last frame (within the given movie) that the track
            can span into. This is to remove late tracks where molecules have
            mostly photobleached.

    Returns:
        df_ints_by_track (Pandas dataframe): result
    """

    #Remove all tracks that are shorter than the minimum duration
    times = get_track_start_end_frames(df_tracks)
    times['duration'] = times['t_end'] - times['t_start']
    long = times[times['duration'] >= min_frames]
    long_early = long[long['t_end'] <= last_frame]

    # remove short tracks
    df_tracks = df_tracks[df_tracks['track_ID'].isin(long_early['track_ID'])]

    df_ints_by_track = pd.DataFrame()

    #determine which intensity field names to use
    intensity_fields = []
    bkgnd_corr_type = analysis_settings["bkgnd_correction_type" ]
    if (int_channel == 'C1') or (int_channel == 'both'):
        if bkgnd_corr_type == "local" or bkgnd_corr_type == "both":
            intensity_fields.append('INT_C1_CORR_LOC')
        if bkgnd_corr_type == "global" or bkgnd_corr_type == "both":
            intensity_fields.append('INT_GLOBAL_C1')

    if (int_channel == 'C2') or (int_channel == 'both'):
        if bkgnd_corr_type == "local" or bkgnd_corr_type == "both":
            intensity_fields.append('INT_C2_CORR_LOC')
        if bkgnd_corr_type == "global" or bkgnd_corr_type == "both":
            intensity_fields.append('INT_GLOBAL_C2')

    track_ints = df_tracks[['track_ID']+intensity_fields]
    df_ints_by_track = track_ints.groupby(['track_ID']).mean().reset_index()

    # Add track standard deviations
    track_stds = track_ints.groupby(['track_ID']).std().reset_index()
    int_std = [x + '_std' for x in intensity_fields]
    df_ints_by_track[int_std] = track_stds[intensity_fields]

    if int_channel == 'both': # Compute total C1 + C2 intensities
        i1, i2 = intensity_fields[0], intensity_fields[1]
        df_ints_by_track['C1_C2'] = df_ints_by_track[i1] + df_ints_by_track[i2]

    return df_ints_by_track

def filt_tracks_by_intensities (df_tracks, df_ints_by_track, int_type, bounds):
    """Filter tracks based on their intensities (both minimum and maximum).

    Args:
        df_tracks (Pandas dataframe): track data (can come from channel of
            either color)
        df_ints_by_track (Pandas dataframe): intensities by track (typically
            this would be the output of the compute_track_intensities function)
        int_type (string): column name in df_ints_by_track storing the
            intensity values will be used for track filtering
        bounds (tuple): lower and upper bounds of the intensities

    Returns:
        df_ints_filt (Pandas dataframe): filtered df_ints_by_track with only
            the desired tracks remaining.
        df_tracks_filt (Pandas dataframe): filtered df_tracks with only the
            desired tracks remaining.
    """
    ibt, it = df_ints_by_track, int_type
    df_ints_filt = ibt.loc[(ibt[it]>bounds[0]) & (ibt[it]<bounds[1])].copy()
    filt_IDs = df_ints_filt['track_ID'].unique()

    df_tracks_filt = df_tracks[df_tracks['track_ID'].isin(filt_IDs)].copy()

    return df_ints_filt, df_tracks_filt


def get_track_start_end_frames(df_tracks):
    """Assemble a list of start and end frames for every track.

    Args:
        df_tracks (Pandas dataframe): track data. Must contain columns
            labeled 't' (frame; must be sorted for each track) and
            'track_name', which uniquely identifies each track.
    Returns:
        df_out (Pandas dataframe): Contains five columns: 'track_ID',
            'idx_start', 'idx_end', 't_start', 't_end'. 't_start' and 't_end'
            two store the first and last frame for each track, while the
            'idx_' columns store the indices of the start and end frames in
            the input dataframe df_tracks.

    """
    # Find first and last frame of each track
    track_IDs = df_tracks['track_ID']
    track_starts = np.diff(track_IDs, prepend=-1).astype(bool)
    track_ends = np.diff(track_IDs, append=-1).astype(bool)

    track_start_idx = df_tracks.index[track_starts]
    frame_start_C1 = df_tracks[['track_ID','t']].iloc[track_start_idx]

    track_end_idx = df_tracks.index[track_ends]
    frame_end_C1 = df_tracks[['t']].iloc[track_end_idx]

    df_out = pd.concat([frame_start_C1.reset_index(drop=False),
                        frame_end_C1['t'].reset_index(drop=False)], axis=1)
    df_out.columns = ['idx_start','track_ID', 't_start', 'idx_end', 't_end']
    df_out = df_out[['track_ID', 'idx_start', 'idx_end', 't_start', 't_end']]

    return df_out

def compute_corr (df_C1, df_C2, overlap_dist=0, window=3,
                  pcc_cutoff=0.95, plot=False):
    """Compute correlations between tracks in two channels (different colors).

    Args:
        df_C1 (Pandas dataframe): track data from channel 1.
        df_C2 (Pandas dataframe): track data from channel 2.
        overlap_dist (float): stringency of overlap requirement. If greater
            than 0, tracks must overlap by at least this much for the overlap
            to count. If less than zero, allows tracks to be shifted by at most
            this much and still count as overlapping (e.g., for less stringent
            spatial overlap calculations). Defaults to 0.
        window (int): number of frames to use for sliding window correlation.
            Defaults to 3.
        pcc_cutoff (float): minimum Pearson's correlation coefficient value
            that makes a given window count as correlated. Defaults to 0.95.
        plot (bool): create a figure with a graphical summary of the results.
            Can be useful for debugging and sanity checks. Defaults to False.

    Returns:
        df_corr (Pandas dataframe): pairwise correlations between the channels.
    """

    # Find tracks that overlap in time
    overlap_t = get_overlap_in_time(df_C1, df_C2,
                                    min_overlap_frames=window)

    # Find tracks that overlap in space
    overlap_x, overlap_y, p_vs_t = get_overlap_in_space (df_C1, df_C2,
                                                 min_overlap=overlap_dist)

    # Combine space and time overlap criteria to make a combined mask
    overlap_all = np.logical_and.reduce((overlap_x['mask'], overlap_y['mask'],
                                        overlap_t['mask']))

    # Find indices of all candidate overlapping tracks
    overlap_all_idx = np.nonzero(overlap_all)
    C1_idx = overlap_all_idx[0]
    C2_idx = overlap_all_idx[1]

    # Assemble all overlapping tracks into NaN-padded n-by-t arrays, where n
    # is the number of pairwise track-track overlaps and t is number of frames
    num_frames = int(max(df_C1['t'].max(), df_C2['t'].max()))+1
    num_overlap_pairs = np.shape(overlap_all_idx)[1]

    ovlp_x_C1 = np.full((num_overlap_pairs, num_frames), np.nan)
    ovlp_x_C2 = np.copy(ovlp_x_C1)
    ovlp_y_C1 = np.copy(ovlp_x_C1)
    ovlp_y_C2 = np.copy(ovlp_x_C1)

    ovlp_x_C1[:,:] = p_vs_t['C1_t_x'][C1_idx, :]
    ovlp_y_C1[:,:] = p_vs_t['C1_t_y'][C1_idx, :]
    ovlp_x_C2[:,:] = p_vs_t['C2_t_x'][C2_idx, :]
    ovlp_y_C2[:,:] = p_vs_t['C2_t_y'][C2_idx, :]

    # Compute pairwise track-track correlations with a sliding window
    corr_pearson_x = windowed_pearson(ovlp_x_C1, ovlp_x_C2, window)
    corr_pearson_y = windowed_pearson(ovlp_y_C1, ovlp_y_C2, window)

    corr_cutoff_x = np.nan_to_num(corr_pearson_x) > pcc_cutoff
    corr_cutoff_y = np.nan_to_num(corr_pearson_y) > pcc_cutoff
    corr_cutoff = np.logical_and(corr_cutoff_x, corr_cutoff_y)
    windows_over_cutoff = np.sum(corr_cutoff, axis=1)

    pos_overlap = np.nonzero(windows_over_cutoff)[0]

    # Plot results as a sanity check
    if plot:
        fig, axarr = plt.subplots(2,2, sharex=True, sharey=True)
        fig.tight_layout(pad=2)
        for position in pos_overlap:
            track_id_C1= overlap_all_idx[0][position]
            track_id_C2= overlap_all_idx[1][position]

            axarr[0,0].plot(p_vs_t['C1_t_x'][track_id_C1,:],
                 p_vs_t['C1_t_y'][track_id_C1,:])
            #axarr[0].scatter(ovlp_x_C1[i,:], ovlp_y_C1[i,:], s=0.5,
            #     c=range(num_frames), edgecolors='face', linewidths=1)
            axarr[0,1].plot(p_vs_t['C2_t_x'][track_id_C2,:],
                 p_vs_t['C2_t_y'][track_id_C2,:])

        # Plot all tracks that are at least as long as the window
        for i in range(len(p_vs_t['C1_t_x'])):
            if np.count_nonzero(~np.isnan(p_vs_t['C1_t_x'][i,:])) >= window:
                axarr[1,0].plot(p_vs_t['C1_t_x'][i,:], p_vs_t['C1_t_y'][i,:])
        for i in range(len(p_vs_t['C2_t_x'])):
            if np.count_nonzero(~np.isnan(p_vs_t['C2_t_x'][i,:])) >= window:
                axarr[1,1].plot(p_vs_t['C2_t_x'][i,:], p_vs_t['C2_t_y'][i,:])

    if plot:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(221)
        plt.imshow(overlap_x['mask'])
        ax = fig.add_subplot(222)
        plt.imshow(overlap_y['mask'])
        ax = fig.add_subplot(223)
        plt.imshow(overlap_t['mask'])
        ax = fig.add_subplot(224)
        plt.imshow(overlap_all)
        ax.set_aspect('equal')
        plt.plot()


    df_corr = pd.DataFrame(
            {'n_corr_windows' : windows_over_cutoff[pos_overlap],
             'Track_ID_C1' : overlap_all_idx[0][pos_overlap],
             'Track_ID_C2' : overlap_all_idx[1][pos_overlap]})

    return df_corr


def plot_trajectory_2color (data, movie_ID):
    """Separately plot two channel trajectories of a given movie side by side.

    Args:
        data (pandas dataframe): dataframe containing parsed data. At the very
            least, must contain the columns 'color', 'movie_ID', and 'parsed'.
        movie_ID (str): unique ID of a movie in data. Must have a match in the
            'movie_ID' column of data.
    Returns:
        fig (matplotlib figure): A plot of the output.

    """
    # Get indices of the movies of both colors, denoted by C1 and C2
    movie_idx = data['movie_ID'] == movie_ID
    idx_C1 = data.index[movie_idx & (data['color'] == 'C1')].tolist()[0]
    idx_C2 = data.index[movie_idx & (data['color'] == 'C2')].tolist()[0]

    tracks_C1 = data.at[idx_C1, 'parsed']
    tracks_C2 = data.at[idx_C2, 'parsed']

    max_frame = max(max(tracks_C1['t']), max(tracks_C2['t']))+1
    t_C1 = build_pos_vs_time_matrix(tracks_C1, max_frame)
    t_C2 = build_pos_vs_time_matrix(tracks_C2, max_frame)

    # Set up the plots
    fig, axarr = plt.subplots(1,2, sharex=True, sharey=True)
    fig.tight_layout(pad=2)
    axarr[0].set_title('All tracks - C1')
    axarr[0].set_xlabel('Position (μm)')
    axarr[0].set_ylabel('Position (μm)')
    axarr[1].set_title('All tracks - C2')
    axarr[1].set_xlabel('Position (μm)')
    axarr[1].set_ylabel('Position (μm)')

    # Iterate over tracks and plot each one
    for i in range(len(t_C1[0])):
        axarr[0].plot(t_C1[0][i], t_C1[1][i])
    for i in range(len(t_C2[0])):
        axarr[1].plot(t_C2[0][i], t_C2[1][i])

    return fig

def get_frac_corr (df_corr, df_C1, df_C2, window, corr_by_window=False):
    """Pull out numbers of total and correlated tracks or windows.

    Args:
        df_corr (Pandas dataframe): pairwise correlations between the channels.
            This is typically the output of the compute_corr function.
        df_C1 (Pandas dataframe): track data from channel 1.
        df_C2 (Pandas dataframe): track data from channel 2.
        window (int): length of the window (in frames) that was used for
            computing the sliding window correlation. All tracks shorter than
            this value are discarded from analysis because they can't
            physically meet the correlation criterion.
        corr_by_window (bool) : normalize correlations by number of windows as
            stored in df_corr. If False, correlation is done on a track by
            track basis (i.e. any two tracks that have at least one
            corrlated window are considered to be correlated). If True, each
            sliding window is treated as a separate "track". Defaults to False.

    Returns:
        frac_corr (dict): Three important values for each channel (C1 and C2):
            'num_corr_Cx': Number of tracks or windows in that channel that are
            found to be correlated with a track or window in another channel,
            'num_tot_Cx': Total number of valid tracks or windos in that
            channel that pass the length cutoff imposed by window,
            'frac_corr_Cx': Fraction of valid tracks or windows in that channel
            that have a correlated track in the other channel.
    """

    #Remove all tracks that are shorter than the correlation window
    times_C1 = get_track_start_end_frames(df_C1)
    times_C1['duration'] = times_C1['t_end'] - times_C1['t_start']
    long_C1 = times_C1[times_C1['duration'] >= window]

    times_C2 = get_track_start_end_frames(df_C2)
    times_C2['duration'] = times_C2['t_end'] - times_C2['t_start']
    long_C2 = times_C2[times_C2['duration'] >= window]


    if corr_by_window: # compute correlations by window
        window_nums_C1 = long_C1['duration'] - window + 1
        num_corr_C1 = df_corr['n_corr_windows'].sum()
        num_tot_C1 = window_nums_C1.sum()

        window_nums_C2 = long_C2['duration'] - window + 1
        num_corr_C2 = df_corr['n_corr_windows'].sum()
        num_tot_C2 = window_nums_C2.sum()


    else: # compute correlations by track
        num_corr_C1 = len(df_corr['Track_ID_C1'].unique())
        num_tot_C1 = len(long_C1)


        num_corr_C2 = len(df_corr['Track_ID_C2'].unique())
        num_tot_C2 = len(long_C2)

    frac_corr_C1 = num_corr_C1 / num_tot_C1
    frac_corr_C2 = num_corr_C2 / num_tot_C2

    frac_corr ={'num_corr_C1' : num_corr_C1,
                'num_tot_C1' : num_tot_C1,
                'frac_corr_C1' : frac_corr_C1,
                'num_corr_C2' : num_corr_C2,
                'num_tot_C2' : num_tot_C2,
                'frac_corr_C2' : frac_corr_C2,}

    return frac_corr

def get_movie_data_both_channels (data, movie_ID):
    """Pull out track data in both channels for one movie out of a dataframe.

    Args:
        data (Pandas dataframe): Parsed tracks for each movie, as created by the
            read_2color_data function of the parse_trackmate module. Contains
            the following columns:
                'parsed', which is the tracks_df dataframe with the actual data,
                'color', numerical value of the channel (typically 1 or 2),
                'movie_ID', full name of the source movie minus channel prefix,
                'file_name' and 'file_path', self-explanatory.
        'movie_ID': movie for which data needs to be pulled

    Returns:
        df_C1 (Pandas dataframe): Parsed tracks for channel 1.
        df_C2 (Pandas dataframe): Parsed tracks for channel 2.
        names_paths (dict): filenames and paths of all associated movies.
    """

    print("Processing movie:", movie_ID)

    # Get indices of the movies of both colors, denoted by C1 and C2
    movie_idx = data['movie_ID'] == movie_ID
    idx_C1 = data.index[movie_idx & (data['color'] == 'C1')]
    idx_C2 = data.index[movie_idx & (data['color'] == 'C2')]

    # Locate data for each color of the current movie
    df_C1 = data.loc[idx_C1, 'parsed'].iat[0]
    df_C2 = data.loc[idx_C2, 'parsed'].iat[0]

    # Locate corresponding file names and paths
    filename_C1 = data.loc[idx_C1, 'file_name'].iat[0]
    file_path_C1 = data.loc[idx_C1, 'file_path'].iat[0]
    filename_C2 = data.loc[idx_C2, 'file_name'].iat[0]
    file_path_C2 = data.loc[idx_C2, 'file_path'].iat[0]

    names_paths = {'name_C1' : filename_C1, 'path_C1' : file_path_C1,
                    'name_C2' : filename_C1, 'path_C2' : file_path_C2}

    return df_C1, df_C2, names_paths

def corr_analysis (data, window=3, pcc_cutoff=0.95, save_dir_data=None,
                    corr_by_window=False, plot_all_tracks=False):
    """Run correlation analysis for all movies in a given condition.

    Args:
        data (Pandas dataframe): Parsed tracks for each movie, as created by the
            read_2color_data function of the parse_trackmate module. Contains
            the following columns:
                'parsed', which is the tracks_df dataframe with the actual data,
                'color', numerical value of the channel (typically 1 or 2),
                'movie_ID', full name of the source movie minus channel prefix,
                'file_name' and 'file_path', self-explanatory.
        window (int): number of frames to use for sliding window correlation.
            Defaults to 3.
        pcc_cutoff (float): minimum Pearson's correlation coefficient value
            that makes a given window count as correlated. Defaults to 0.95.
        save_dir_data (str): Location to save TrackMate files that only include
            trajectories that were found to be correlated.
        corr_by_window (bool): If True, calculates fraction of all valid windows
            that were found to be correlated. If False, calculates fraction of
            all tracks that were found to be correlated. Defaults to False.
        plot_all_tracks (bool): create separate figures showing every track.
            Takes forever and should only be used for debugging. Defaults to
            False.


    Returns:
        result (Pandas dataframe): Fractions of correlated trajectories per
            movie.
    """
    result = pd.DataFrame()

    for movie_ID in data['movie_ID'].unique():

        df_C1, df_C2, names_paths = get_movie_data_both_channels (data,
                                                                    movie_ID)

        # Test for correlation
        df_corr = compute_corr(df_C1, df_C2,
            pcc_cutoff=pcc_cutoff, window=window, plot=plot_all_tracks)

        out = get_frac_corr(df_corr, df_C1, df_C2, window=window,
                                corr_by_window=corr_by_window)
        out['movie_ID'] = movie_ID
        result = result.append(out, ignore_index=True)

        # Save filtered TrackMate files, if desired
        if save_dir_data:
            file_path_out_C1 = os.path.join(save_dir_data,
                            'Filtered'+names_paths['name_C1']+'.xml')
            file_path_out_C2 = os.path.join(save_dir_data,
                            'Filtered'+names_paths['name_C2']+'.xml')

            pt.write_tracks_to_trackmate_file (names_paths['path_C1'],
                file_path_out_C1, df_C1, df_corr['Track_ID_C1'])
            pt.write_tracks_to_trackmate_file (names_paths['path_C2'],
                file_path_out_C2, df_C2, df_corr['Track_ID_C2'])

    return result

def intensity_analysis (data, int_settings, save_dir_data=None):
    """INCOMPLETE: Run intensity analysis for all movies in a given condition.

    Args:
        data (Pandas dataframe): Parsed tracks for each movie, as created by the
            read_2color_data function of the parse_trackmate module. Contains
            the following columns:
                'parsed', which is the tracks_df dataframe with the actual data,
                'color', numerical value of the channel (typically 1 or 2),
                'movie_ID', full name of the source movie minus channel prefix,
                'file_name' and 'file_path', self-explanatory.
        int_settings (dict): settings for intensity analysis.
        save_dir_data (str): Location to save TrackMate files that only include
            trajectories that were found to be correlated. Defaults to None.

    Returns:
        result (Pandas dataframe): Fractions of correlated trajectories per
            movie.
    """

    result = pd.DataFrame()

    for movie_ID in data['movie_ID'].unique():

        df_C1,df_C2,names_paths = get_movie_data_both_channels (data, movie_ID)

        track_ints = pt.compute_track_intensities(df_C1, 'both', int_settings)
        out, tracks_filt = pt.filt_tracks_by_intensities (df_C1, track_ints,
                                                    "INT_C2_CORR_LOC", (50,500))
        out['movie_ID'] = movie_ID

        result = result.append(out, ignore_index=True)

        # Save filtered TrackMate files, if desired
        if save_dir_data:
            file_path_out_C1 = os.path.join(save_dir_data,
                            'Filt_Intensity'+names_paths['name_C1']+'.xml')
            pt.write_tracks_to_trackmate_file (names_paths['path_C1'],
                file_path_out_C1, df_C1, out['track_ID'])

    return result


def plot_correlation_results (corr_results, plot_1='frac_corr_C1',
                                plot_2='num_tot_C1'):
    """Create nice plots showing fractions of correlated trajectories.

    Args:
        corr_results (Pandas dataframe): correlation results, output of the
            corr_analysis function.
        plot_1 (str): condition to plot in the first plot. Must be a valid key
            in corr_results. Defaults to 'frac_corr_C1'.
        plot_2 (str): condition to plot in the second plot. Must be a valid key
            in corr_results. Defaults to 'num_tot_C1'.

    Returns:
        fig (matplotlib figure): main data figure.
        fig_linreg: second figure showing the linear fit of plot_2 vs plot_1
    """
    num_conditions = len(corr_results['Condition'].unique().tolist())

    fig,ax = plt.subplots(ncols=3)

    # Make the first plot2
    plot = sns.barplot(x="Condition", y=plot_1, data=corr_results, ci=95,
                        color="lightsteelblue", capsize=0.1, ax=ax[0])
    sns.stripplot(x="Condition", y=plot_1, data=corr_results, size=3,
                        jitter=0.2, color=".6", ax=ax[0])

    # Make the second plot
    plot2 = sns.barplot(x="Condition", y=plot_2, data=corr_results, ci=95,
                        color="lightsteelblue", capsize=0.1, ax=ax[1])
    sns.stripplot(x="Condition", y=plot_2, data=corr_results, size=3,
                        jitter=0.2, color=".6", ax=ax[1])

    # Make a scatterplot of condition 2 vs condition 1
    marker_list = ['o'] * num_conditions
    sns.scatterplot(x=plot_2, y=plot_1,
        hue="Condition", style="Condition", data=corr_results,
                        markers=marker_list, ax=ax[2])

    #Rotate x-labels to avoid overlap
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45,
                        horizontalalignment='right')
    plot2.set_xticklabels(plot.get_xticklabels(), rotation=45,
                        horizontalalignment='right')

    fig_linreg = sns.lmplot(x=plot_2, y=plot_1,
        hue="Condition", data=corr_results);
    ax[0].set_ylim(0,)
    ax[1].set_ylim(0,)

    return fig, fig_linreg

def pairwise_t_tests (corr_results, statistic_to_test='frac_corr_C1'):
    """Calculate pairwise t-test statistics between all conditions.

    Args:
        corr_results (Pandas dataframe): Correlation results, output of the
            corr_analysis function.
        statistic_to_test (str): Parameter to be tested. Must be a valid key in
            corr_results. Defaults to 'frac_corr_C1'.
    Returns:
        t_test_summaries (str): Human-readable summary of all pairwise
            comparisons.
    """

    all_conditions = corr_results['Condition'].unique().tolist()
    c_list = all_conditions.copy()
    t_test_summaries = []
    for cond1 in c_list:
        all_conditions.remove(cond1)
        for cond2 in all_conditions:
            st=statistic_to_test
            data_in_c1 = corr_results[corr_results['Condition'] == cond1][st]
            data_in_c2 = corr_results[corr_results['Condition'] == cond2][st]
            t_test = str(stats.ttest_ind(data_in_c1, data_in_c2,
                                        equal_var=False))
            t_test_string = " ".join(['T-test between conditions',cond1,'and',
                                        cond2,':'])
            t_test_summaries.append(" ".join([t_test_string, t_test, '\n']))

    return t_test_summaries


def test_pearson_corr (window=3, pcc_cutoff=0.95):
    """Debug use only - runs correlation analysis on a pair of dummy tracks.

    Args:
        window (int): number of frames to use for sliding window correlation.
            Defaults to 3.
        pcc_cutoff (float): minimum Pearson's correlation coefficient value
            that makes a given window count as correlated. Defaults to 0.95.

    Returns:
        None. (results are displayed as text)
    """
    ovlp_x_C1 = np.full((3,8), np.nan)
    ovlp_x_C2 = np.copy(ovlp_x_C1)
    ovlp_x_C1[0,2] = 3
    ovlp_x_C1[1,2:6] = [5,6,7,2]
    ovlp_x_C1[2,1:6] = [4,5.5,6.2,7.1,2.2]
    ovlp_x_C2[1,1:6] = [4.1,5.2,5.1,7.5,1.2]
    ovlp_x_C2[2,4:8] = [1,3,2,3]
    ovlp_y_C1 = np.copy(ovlp_x_C1)
    ovlp_y_C2 = np.copy(ovlp_x_C2)
    ovlp_y_C2[1,3] = 6

    #print(np.shape(ovlp_x_C1))
    print(ovlp_x_C1)
    print(ovlp_x_C2)
    print(ovlp_y_C1)
    print(ovlp_y_C2)


    corr_pearson_x = windowed_pearson(ovlp_x_C1, ovlp_x_C2, window)
    corr_pearson_y = windowed_pearson(ovlp_y_C1, ovlp_y_C2, window)

    corr_cutoff_x = np.nan_to_num(corr_pearson_x) > pcc_cutoff
    corr_cutoff_y = np.nan_to_num(corr_pearson_y) > pcc_cutoff
    corr_cutoff = np.logical_and(corr_cutoff_x, corr_cutoff_y)
    windows_over_cutoff = np.sum(corr_cutoff, axis=1)

    print(corr_pearson_x)
    print(corr_pearson_y)
    print(windows_over_cutoff)

    return
