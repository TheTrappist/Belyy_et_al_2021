# -*- coding: utf-8 -*-
"""

This module contains a collection of functions for reading and parsing xml
files output by the TrackMate ImageJ plugin.

Written by Vladislav Belyy (UCSF)

"""
import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pylab as plt
from lxml import etree
import glob



def parse_trackmate_file (file, load_all_attr=False, spot_num_attr=[]):
    """(Reasonably) efficiently parse a TrackMate file into tracks using lxml.

    This is a significant speed-up of my previous approach that was using
    minidom. Taking a single output file from the TrackMate plugin as an input,
    this function parses the spots and edges in the file and returns nicely
    formatted pandas dataframes containing the individual spots, edges, and
    tracks.

    Args:
        file (str): Full path to TrackMate xml file.
        load_all_attr (bool): If True, loads all attributes associated with the
            spots and edges (such as radius, quality, etc...) This is slower
            but can be more informative, depending on what the data are used
            for. If False, only loads spot and track IDs, and x-y-t
            coordinates. Defaults to False.
        spot_num_attr (list of str): If provided, loads additional numerical
            spot attributes from indicated fields. If spots don't have one of
            the specified attributes, execution fails. Defaults to empty list.

    Returns:
        tracks_df (pandas dataframe): Dataframe of spot positions and spot IDs,
            sorted by track ID first and frame second. Only contains the spots
            that are associated with a track.
        edges_df (pandas dataframe): Dataframe of track edges, sorted by track
            ID first and frame second.
        spots_df (pandas dataframe): Dataframe of all spots, regardless of
            whether the spot is associated with a track or not. Sorted by spot
            ID.
    """
    # Parse the datafile
    data_etree = etree.parse(file)

    ### First, build a dataframe of spots #####################################
    spots_by_frames = data_etree.xpath('/TrackMate/Model/AllSpots')[0]

    # These are the required minimal spot attributes that need to be loaded
    main_attr_spot = ['ID','FRAME','POSITION_X','POSITION_Y']

    # Get keys of the spot attributes to be pulled
    if load_all_attr:
        spot_attributes = spots_by_frames[0][0].keys()
    elif spot_num_attr:
        spot_attributes = main_attr_spot + spot_num_attr
    else:
        spot_attributes = main_attr_spot

    # Iterate over all spots, pull the required attributes of each spot
    spot_rows = []
    for spots in spots_by_frames:
        for spot in spots:
            spot_attrib = {}
            for attrib in spot_attributes:
                spot_attrib[attrib]= spot.get(attrib)
            spot_rows.append(spot_attrib)

    # Assemble spot data into a nicely formatted and sorted dataframe, spots_df
    spots_df = pd.DataFrame(spot_rows)
    spots_df[main_attr_spot] = spots_df[main_attr_spot].apply(pd.to_numeric)
    if spot_num_attr:
        spots_df[spot_num_attr] = spots_df[spot_num_attr].apply(pd.to_numeric)
    spots_df['FRAME'] = spots_df['FRAME'].astype('int32')
    spots_df.sort_values(by='ID', inplace=True)
    spots_df.reset_index(drop=True, inplace=True)


    ### Next, build a dataframe of edges ######################################
    tracks = data_etree.xpath('/TrackMate/Model/AllTracks')[0]

    # Just as with spots, get keys of the edge attributes to be pulled
    main_attr_edge = ['SPOT_SOURCE_ID', 'SPOT_TARGET_ID', 'EDGE_TIME']
    if load_all_attr:
        edge_attributes = tracks[0][0].keys()
    else:
        edge_attributes = main_attr_edge

    # Iterate over edges, pull the required attributes of each edge.
    # Additionally, save track ID and track name associated with each edge.
    edge_rows = []
    for track in tracks:
        for edge in track.iter('Edge'):
            edge_attrib = {}
            edge_attrib['track_name'] = track.get('name')
            edge_attrib['track_ID'] = track.get('TRACK_ID')
            for attrib in edge_attributes:
                edge_attrib[attrib]= edge.get(attrib)
            edge_rows.append(edge_attrib)

    # Assemble edge data into a nicely formatted dataframe, edges_df
    edges_df = pd.DataFrame(edge_rows)
    edges_df[main_attr_edge] = edges_df[main_attr_edge].apply(pd.to_numeric)
    edges_df['track_ID'] = edges_df['track_ID'].astype('int64')

    # Add info about the source spot for each edge in the dataframe
    spot_idx = spots_df['ID'].searchsorted(edges_df['SPOT_SOURCE_ID'])
    edges_df['spot_source_ID'] = spots_df['ID'].iloc[spot_idx].values
    edges_df['spot_source_frame'] = spots_df['FRAME'].iloc[spot_idx].values

    # Sort the edges dataframe by tracks and time. Note that source spot frame
    # is used because for some reason TrackMate's "edge time" attribute is not
    # reliable for sorting and does not always correspond to the source spots'
    # order. This probably has something to do with gap closing (?).

    edges_df.sort_values(by=['track_ID','spot_source_frame'], inplace=True)
    edges_df.reset_index(drop=True, inplace=True)

    ### Assemble data into tracks #############################################

    # First, pull out IDs of all but the last spot of each track (since each
    # edge contains a source and target spot, the length of a spot list for a
    # track is always 1 greater than the length of the corresponding edge list)
    tracks_df = pd.DataFrame()
    tracks_df['track_ID'] = edges_df['track_ID']
    tracks_df['track_name'] = edges_df['track_name']
    tracks_df['spot_ID'] = edges_df['spot_source_ID']
    tracks_df['t'] = edges_df['spot_source_frame']

    # Find and append the last spot of each track
    last_spots = pd.DataFrame()
    track_ends = np.diff(edges_df['track_ID'], append=-1).astype(bool)
    end_spot_idx = edges_df['SPOT_TARGET_ID'].iloc[track_ends]
    track_end_idx = spots_df['ID'].searchsorted(end_spot_idx)
    last_spots['track_ID'] = edges_df['track_ID'].iloc[track_ends].values
    last_spots['track_name'] = edges_df['track_name'].iloc[track_ends].values
    last_spots['spot_ID'] = spots_df['ID'].iloc[track_end_idx].values
    last_spots['t'] = spots_df['FRAME'].iloc[track_end_idx].values
    tracks_df = tracks_df.append(last_spots, ignore_index=True)

    # Populate the tracks dataframe with spot coordinates
    spot_idx_all = spots_df['ID'].searchsorted(tracks_df['spot_ID'])
    tracks_df['x'] = spots_df['POSITION_X'].iloc[spot_idx_all].values
    tracks_df['y'] = spots_df['POSITION_Y'].iloc[spot_idx_all].values
    tracks_df['t'] = spots_df['FRAME'].iloc[spot_idx_all].values

    # populate the tracks dataframe with any custom spot attributes
    for spot_attr in spot_num_attr:
        tracks_df[spot_attr] = spots_df[spot_attr].iloc[spot_idx_all].values

    # Sort tracks by track ID and frame, yielding a nicely formatted track_df
    tracks_df.sort_values(by=['track_ID','t'], inplace=True)
    tracks_df.reset_index(drop=True, inplace=True)

    return tracks_df, edges_df, spots_df

def write_tracks_to_trackmate_file (file_in, file_out, tracks_df, track_IDs,
                                    path_to_movie=None):
    """ Make a copy of a TrackMate xml file containing only selected tracks.

    Args:
        file_in (str): Path to source TrackMate xml file.
        file_out (str): Path to source TrackMate xml file.
        tracks_df (pandas dataframe): Dataframe of spot positions and spot IDs,
            sorted by track ID first and frame second. Only contains the spots
            that are associated with a track.
        track_IDs (pandas series): Track_ID values of tracks to be included in
            the output file. Other tracks, as well as spots not associated with
            one of these tracks, will be removed.
        path_to_movie (str): full path to a movie on which the filtered tracks
            are to be overlaid. Useful, e.g., for taking trajectories from a
            single-color movie and overlaying them on a 2-color movie. This
            value replaces the Settings>ImageData>filename and >folder fields
            of the output xml. If set to None, these fields remain unchanged.
            Note that this must contain the full filename as well as the path.
            Defaults to None.

    Returns:
        None.
    """
    # Parse the datafile
    data_etree = etree.parse(file_in)

    # Remove AllTracks that aren't in 'track_ID'
    track_ID_vals = track_IDs.values

    tracks = data_etree.xpath('/TrackMate/Model/AllTracks')[0]
    for track in tracks:
        track_ID = int(track.get('TRACK_ID'))
        if track_ID not in track_ID_vals: # Remove this track
            tracks.remove(track) # remove the track itself

    # Remove FilteredTracks that aren't in 'track_ID'
    filt_tracks = data_etree.xpath('/TrackMate/Model/FilteredTracks')[0]
    for track in filt_tracks:
        track_ID = int(track.get('TRACK_ID'))
        if track_ID not in track_ID_vals: # Remove this track
            filt_tracks.remove(track) # remove the track itself

    # Remove spots not associated with one of the tracks in 'track_ID'
    spots_by_frames = data_etree.xpath('/TrackMate/Model/AllSpots')[0]


    tracks_kept = tracks_df[tracks_df['track_ID'].isin(track_ID_vals)]
    spots_kept = tracks_kept['spot_ID'].values
    # Iterate over all spots removing any that aren't associated with a track
    for spots_in_frame in spots_by_frames:
        for spot in spots_in_frame:
            spot_ID = int(spot.get('ID'))
            if spot_ID not in spots_kept:
                spots_in_frame.remove(spot)

    if path_to_movie: # Update the ImageData fields
        img_data = data_etree.xpath('/TrackMate/Settings/ImageData')[0]
        abs_path = os.path.abspath(path_to_movie)
        head_tail = os.path.split(abs_path)
        img_data.set('filename', head_tail[1])
        img_data.set('folder', head_tail[0])

    # write updated tree to file
    data_etree.write(file_out)

    return


def read_2color_data (data_dirs, do_int_analysis=False, int_settings=None):
    """Read 2-color TrackMate xml data from files in one or more directories.

    Args:
        data_dirs (list of str): Paths to all directories containing the xml
            TrackMate files for the current condition. Note that the files
            should be separated into channels using ImageJ's standard "C1_" and
            "C2_" prefixes.
        do_int_analysis (bool): will these tracks be used for spot intensity
            analysis? If true, background-corrected spot intensities are loaded.
            Defaults to False.
        'int_settings' (dict): Intensity analysis settings. Defaults to None.

    Returns:
        data (Pandas dataframe): Parsed tracks for each movie. Contains the
            following columns:
                'parsed', which is the tracks_df dataframe with the actual data,
                'color', numerical value of the channel (typically 1 or 2),
                'movie_ID', full name of the source movie minus channel prefix,
                'file_name' and 'file_path', self-explanatory.
    """

    # Prepare the "data" dataframe
    if type(data_dirs) is not list: data_dirs = [data_dirs] # for a single dir
    data_files, file_names = [],[]
    for data_dir in data_dirs:
        curr_files = sorted(glob.glob(os.path.join(data_dir,'**/*.xml'),
                                                                recursive=True))
        stripped_names = [os.path.basename(f) for f in curr_files]
        curr_names = [os.path.splitext(fn)[0] for fn in stripped_names]
        data_files = data_files + curr_files
        file_names = file_names + curr_names
    data = pd.DataFrame({'file_name' : file_names, 'file_path' : data_files})
    try:
        data['color'] = data['file_name'].str.slice(0,2)
        data['movie_ID'] = data['file_name'].str.slice(3,)
    except AttributeError:
        print('Error: at least one file path is invalid. See info below.')
        print(data_dirs)
        print(data)

    # Determine which (if any) numerical spot attributes to parse
    spot_num_attr = []
    if do_int_analysis:
        if int_settings["bkgnd_correction_type"] == "local":
            spot_num_attr = ['INT_C1_CORR_LOC', 'INT_C2_CORR_LOC']
        elif int_settings["bkgnd_correction_type"] == "global":
            spot_num_attr = ['INT_GLOBAL_C1', 'INT_GLOBAL_C2']
        else: # load all
            spot_num_attr = ['INT_C1_CORR_LOC', 'INT_C2_CORR_LOC',
                             'INT_GLOBAL_C1', 'INT_GLOBAL_C2']
    # Parse tracking data
    data['parsed'] = ""
    for idx in data.index:
        a,_,_ = parse_trackmate_file(data['file_path'].loc[idx],
                                       spot_num_attr=spot_num_attr)
        data.at[idx, 'parsed'] = a

    return data
