# This script scans through a directory of 2-color movies, separately
# tracks particles in both channels using the TrackMate plugin with
# user-specified custom settings, and saves the resulting TrackMate
# output files to a user-specified directory. The user settings are specified
# in a JSON file (see example file to see how it should be formatted).

# Written by Vladislav Belyy at UCSF. Code is heavily borrowed
# from TrackMate and ImageJ scripting examples on the respective
# Wiki pages.


#@ File(label='Directory for source files', style='directory') import_dir
#@ String(label='File types', value='tif;png') file_types
#@ File(label='Input settings file, leave blank to use defaults', value='*.json', style='open') settings_in
#@ String(label='Output settings file', value='TrackMate_2color_settings') settings_out
#@ String(label='Filter', value='') filters
#@ Boolean(label='Recursive search', value=True) do_recursive
#@ Boolean(label='Preserve subdirectory structure', value=True) preserve_subdir_struct
#@ File(label='Directory to save TrackMate output to', style='directory') out_dir


import sys
import json
import os
import math
from java.io import File
from ij import IJ, WindowManager
from ij.plugin import ChannelSplitter
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking import LAPUtils
from fiji.plugin.trackmate.io import TmXmlWriter
from fiji.plugin.trackmate.Dimension import INTENSITY, LENGTH
import fiji.plugin.trackmate.features.ModelFeatureUpdater as ModelFeatureUpdater
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import fiji.plugin.trackmate.features.track.TrackDurationAnalyzer as TrackDurationAnalyzer
import fiji.plugin.trackmate.features.spot.SpotIntensityAnalyzerFactory as SpotIntensityAnalyzerFactory


### Load tracking settings from file, or populate with defaults if no file is specified ###

if isinstance(settings_in, File): # Convert to string path
	settings_in = settings_in.getAbsolutePath()

if os.path.isfile(settings_in): # Load settings from JSON file
	with open(settings_in) as fd:
		json_data_in = json.load(fd)

	detector_settings_C1 = json_data_in[0]['detector_settings_C1']
	detector_settings_C2 = json_data_in[0]['detector_settings_C2']
	tracker_settings = json_data_in[0]['tracker_settings']
	min_track_duration = json_data_in[0]['min_track_duration']

	# Load optional settings
	try:
		bkgnd_corr_settings = json_data_in[0]['bkgnd_corr_settings']
	except:
		bkgnd_corr_settings = None
		print("Settings file contains no background correction settings; reverting to defaults")

	print("Succesfully loaded data from JSON file")

else: # Load default settings
	print("Settings file is invalid or not specified; using default settings")
	# Spot detection settings for channel 1
	detector_settings_C1 = {
			'DO_SUBPIXEL_LOCALIZATION' : True,
			'RADIUS' : 0.25,
			'TARGET_CHANNEL' : 1,
			'THRESHOLD' : 2.0,
			'DO_MEDIAN_FILTERING' : True,
		}

	detector_settings_C2 = detector_settings_C1.copy()
	# Specify any changes in settings for channel 2
	detector_settings_C2['TARGET_CHANNEL'] = 2
	detector_settings_C2['THRESHOLD'] = 2.0
	detector_settings_C2['RADIUS'] = 0.25

	# Tracker settings (for both channels)
	tracker_settings = {
		'LINKING_MAX_DISTANCE' : 0.5,
		'ALLOW_GAP_CLOSING' : False,
		'ALLOW_TRACK_SPLITTING' : False,
		'ALLOW_TRACK_MERGING' : False,
		}

	min_track_duration = 3



### End settings initialization ################################################


def get_images_in_dir(path, file_type=None, name_filter=None, recursive=False):
    '''Get paths to all image files in a given folder.

    Args:
    	path: The path from where to open the images. String and java.io.File are allowed.
    	file_type: Only accept files with the given extension (default: None).
    	name_filter: Only accept files that contain the given string (default: None).
    	recursive: Process directories recursively (default: False).
    '''
    # Converting a File object to a string.
    if isinstance(path, File):
        path = path.getAbsolutePath()

    def check_type(string):
        '''This function is used to check the file type.

        It is possible to use a single string or a list/tuple of strings as filter.
        This function can access the variables of the surrounding function.
        Args:
        	string: The filename to perform the check on.
        '''
        if file_type:
            # The first branch is used if file_type is a list or a tuple.
            if isinstance(file_type, (list, tuple)):
                for file_type_ in file_type:
                    if string.endswith(file_type_):
                        # Exit the function with True.
                        return True
                    else:
                        # Next iteration of the for loop.
                        continue
            # The second branch is used if file_type is a string.
            elif isinstance(file_type, string):
                if string.endswith(file_type):
                    return True
                else:
                    return False
            return False
        # Accept all files if file_type is None.
        else:
            return True

    def check_filter(string):
        '''This function is used to check for a given filter.

        It is possible to use a single string or a list/tuple of strings as filter.
        This function can access the variables of the surrounding function.
        Args:
        	string: The filename to perform the filtering on.
        '''
        if name_filter:
            # The first branch is used if name_filter is a list or a tuple.
            if isinstance(name_filter, (list, tuple)):
                for name_filter_ in name_filter:
                    if name_filter_ in string:
                        # Exit the function with True.
                        return True
                    else:
                        # Next iteration of the for loop.
                        continue
            # The second branch is used if name_filter is a string.
            elif isinstance(name_filter, string):
                if name_filter in string:
                    return True
                else:
                    return False
            return False
        else:
        # Accept all files if name_filter is None.
            return True

    # We collect all files to open in a list.
    path_to_images = []
    # Replacing some abbreviations (e.g. $HOME on Linux).
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    # If we don't want a recursive search, we can use os.listdir().
    if not recursive:
        for file_name in os.listdir(path):
            full_path = os.path.join(path, file_name)
            if os.path.isfile(full_path):
                if check_type(file_name):
                    if check_filter(file_name):
                        path_to_images.append(full_path)
    # For a recursive search os.walk() is used.
    else:
        # os.walk() is iterable.
        # Each iteration of the for loop processes a different directory.
        # the first return value represents the current directory.
        # The second return value is a list of included directories.
        # The third return value is a list of included files.
        for directory, dir_names, file_names in os.walk(path):
            # We are only interested in files.
            for file_name in file_names:
                # The list contains only the file names.
                # The full path needs to be reconstructed.
                full_path = os.path.join(directory, file_name)
                # Both checks are performed to filter the files.
                if check_type(file_name):
                    if check_filter(file_name):
                        # Add the file to the list of images to open.
                        path_to_images.append(full_path)

    """
    # Create the list that will be returned by this function.
    images = []
    for img_path in path_to_images:
        # IJ.openImage() returns an ImagePlus object or None.
        imp = IJ.openImage(img_path)
        # An object equals True and None equals False.
        if imp:
            images.append(imp)
    return images
    """
    return path_to_images

def split_string(input_string):
    '''Split a string to a list and strip it
    :param input_string: A string that contains semicolons as separators.
    '''
    string_splitted = input_string.split(';')
    # Remove whitespace at the beginning and end of each string
    strings_striped = [string.strip() for string in string_splitted]
    return strings_striped

def save_settings( 	settings_out,
					out_dir,
					tracker_settings,
				  	detector_settings_C1,
				  	detector_settings_C2,
				  	bkgnd_corr_settings,
				  	min_track_duration,
				  	all_settings_C1,
				  	all_settings_C2):
	""" Save tracker settings for the current run to a JSON file.

	"""

	# Prepare directories and file name

	if isinstance(out_dir, File):
		out_dir = out_dir.getAbsolutePath()
	out_settings_dir = os.path.join(out_dir, 'Tracking_Settings')
	if not os.path.exists(out_settings_dir):
		os.makedirs(out_settings_dir)

	fname_out = settings_out + '.json'
	full_path_out = os.path.join(out_settings_dir, fname_out)

	# Package miscellaneous metadata into dictionary
	misc_meta = {'settings_out' : settings_out}

	# combine all dictionaries into master list

	user_defined = {'detector_settings_C1' : detector_settings_C1,
				  	'detector_settings_C2' : detector_settings_C2,
				  	'tracker_settings': tracker_settings,
				  	'bkgnd_corr_settings': bkgnd_corr_settings,
				  	'min_track_duration' : min_track_duration,
				  	'misc_meta' : misc_meta}
	full_TrackMate_settings_dump = {
				  	'all_settings_C1_string' : all_settings_C1.toString(),
				  	'all_settings_C2_string' : all_settings_C2.toString()}
	data_to_save = [user_defined, full_TrackMate_settings_dump]

	with open(full_path_out, 'w') as fout:
		json.dump(data_to_save , fout, indent=2)

# Add a new spot feature to the model object;
def add_spot_feature (model, feature, name, short_name, dimension, is_int):
	""" Add a new spot feature to the model object.

	This must be done for custom features added using spot.putFeature to
	get properly integrated into the model. The trackmate model object
	is modified in-place.

	"""
	feature_model = model.getFeatureModel()

	spot_features = feature_model.getSpotFeatures()
	spot_feature_names = feature_model.getSpotFeatureNames()
	spot_feature_short_names = feature_model.getSpotFeatureShortNames()
	spot_feature_dimensions = feature_model.getSpotFeatureDimensions()
	spot_feature_is_int = feature_model.getSpotFeatureIsInt()

	spot_features.add(feature)
	spot_feature_names[feature] = name
	spot_feature_short_names[feature] = short_name
	spot_feature_dimensions[feature] = dimension
	spot_feature_is_int[feature] = is_int

	return

def calc_custom_intensity(model, settings, trackmate, feature_int,
							feature_radius, radius, imp):
	"""Recalculate spot intensities in model with a custom image and radius.

	Changes the model and settings objects to calculate mean spot
	intensities using the supplied radius (in physical units) and the
	supplied ImagePlus object. Can be used, e.g., to measure intensities
	of spots in a different channel or with a different radius. 'feature_int'
	and 'feature_radius' should be strings containing names of the new
	intensity and radius features in 'model'; make sure to add these features
	to the model beforehand. Note that the model and settings objects will be
	modified in place after calling this function.
	"""

	settings.imp = imp
	all_spots = model.getSpots()

	# Manually change spot radii
	model.beginUpdate()
	for spot in all_spots.iterator(False):
		spot.putFeature('RADIUS', radius)
		spot.putFeature(feature_radius, radius)
	model.endUpdate()

	# Recalculate spot features
	ok = trackmate.computeSpotFeatures(True)
	if not ok:
		sys.exit(str(trackmate.getErrorMessage()))

	# Write the new mean intensities into the new feature field
	model.beginUpdate()
	for spot in all_spots.iterator(False):
		mean = spot.getFeature('MEAN_INTENSITY')
		spot.putFeature(feature_int, mean)
	model.endUpdate()

	return

def calc_spot_intensity_global_bkgnd (model, settings, trackmate, imp,
										corr_int_name, custom_radius=None,
										custom_radius_field_name=None):
	"""Calculate mean spot intensities with global background correction.

	Computes global image background by taking the median of the bottom 50%
	percent of all pixels in the image (median is used rather than mean to
	negate possible effects from 'dead' pixels that would skew the mean down).
	50% is used with the assumption that at least this fraction of the image will
	be free of diffusing spots and background fluorescence. Then, subtracts the
	calculated global background from all spots in model. Note that, when
	processing movies, the first and last frames are used together for background
	subtraction and the results are averaged. If custom_radius is not set, uses
	the radius of spots stored inside model.
	"""

	settings.imp = imp
	all_spots = model.getSpots()

	# Calculate background
	num_frames = imp.getNFrames()

	stack = imp.getStack()

	# Calculate stats for first and last frames
	mean_bkgnd_ints = []
	for i in (1, num_frames):
		# Get the frame i
		pixels = list(stack.getPixels(i))
		pixels.sort(key=int)
		end_i = int(len(pixels) / 2) # get index of last pixel of the fist half
		pixels_1st_half = pixels[0:end_i]
		mean = sum(pixels_1st_half) / len(pixels_1st_half)
		mean_bkgnd_ints.append(mean)

	mean_bkgnd_int = sum(mean_bkgnd_ints) / len(mean_bkgnd_ints)

	# Change spot radii and recompute intensities if needed
	if custom_radius:
		model.beginUpdate()
		for spot in all_spots.iterator(False):
			spot.putFeature('RADIUS', custom_radius)
			if custom_radius_field_name:
				spot.putFeature(custom_radius_field_name, custom_radius)
		model.endUpdate()

		# Recalculate spot features
		ok = trackmate.computeSpotFeatures(True)
		if not ok:
			sys.exit(str(trackmate.getErrorMessage()))

	# Write the new background-corrected intensities into the new feature field
	model.beginUpdate()
	for spot in all_spots.iterator(False):
		mean = spot.getFeature('MEAN_INTENSITY')
		radius =  spot.getFeature('RADIUS')

		mean_corr = mean - mean_bkgnd_int
		spot.putFeature(corr_int_name, mean_corr)

	model.endUpdate()

	return

def calc_spot_intensity_local_bkgnd (imp_calibration, model, feat_r0, feat_r1,
										feat_mean_r0, feat_mean_r1, feature_out):
	"""Calculate mean spot intensities corrected for local background.

	The model object must contain fields storing inner and outer spot radii
	(with field names feat_r0 and feat_r1), as well as the pre-calculated
	mean intensities of pixels contained within the inner and outer radii
	(field names feat_mean_r0 and feat_mean_r1). The background correction
	draws a "donut" around each spot with inner radius r0 and outer radius
	r1, then takes mean pixel intensity within this donut region as the local
	background and subtracts its from the intensity of the spot contained
	within r0, returning the cumulative corrected intensity of the spot as
	the new feature "feature_out" that's written directly into the model
	object.

	CAUTION: Currently may not correctly calculate local background for spots
	near the edge of the image, where r1 extends outside the image borders.
	This should be corrected later on.
	"""

	if imp_calibration.pixelWidth != imp_calibration.pixelHeight:
		raise Exception("Pixels are not square! Cannot proceed with background correction.")

	all_spots = model.getSpots()
	model.beginUpdate()

	for spot in all_spots.iterator(False):
		r0 = spot.getFeature(feat_r0)
		r1 = spot.getFeature(feat_r1)
		mean_r0 = spot.getFeature(feat_mean_r0)
		mean_r1 = spot.getFeature(feat_mean_r1)

		# Get inner and outer spot areas in pixels
		a0 = (r0 / imp_calibration.pixelWidth) ** 2
	  	a1 = (r1 / imp_calibration.pixelWidth) ** 2

	  	# Calculate mean pixel intensity in the "donut" between r1 and r0
	  	donut_area = a1-a0

		# Calculate the donut's mean intensity (the local background)
		cum_int_r0 = a0 * mean_r0
		cum_int_r1 = a1 * mean_r1
		cum_int_donut = cum_int_r1 - cum_int_r0
		mean_int_donut = cum_int_donut / donut_area

		# subtract local background from the spot's intensity
		cum_int_corr = cum_int_r0 - (a0 * mean_int_donut)
		mean_int_corr = cum_int_corr / a0
		spot.putFeature(feature_out, mean_int_corr)

	model.endUpdate()

	return

def local_bkgnd_correction_2channel_2radii (model, settings, trackmate, imp, r0, r1):
	"""Calculate local spot background corrections in two channels and with two radii."""

	# Split the source image into channels
	imps = ChannelSplitter.split(imp)

	# Create features to store all the new intensity measurements
	add_spot_feature(model,'MEAN_INT_R0_C1','Mean_Intensity_R0_C1','Int_R0_C1',INTENSITY,False)
	add_spot_feature(model,'MEAN_INT_R1_C1','Mean_Intensity_R1_C1','Int_R1_C1',INTENSITY,False)
	add_spot_feature(model,'MEAN_INT_R0_C2','Mean_Intensity_R0_C2','Int_R0_C2',INTENSITY,False)
	add_spot_feature(model,'MEAN_INT_R1_C2','Mean_Intensity_R1_C2','Int_R1_C2',INTENSITY,False)
	add_spot_feature(model,'RADIUS_R0','Radius_R0','R0',LENGTH,False)
	add_spot_feature(model,'RADIUS_R1','Radius_R1','R1',LENGTH,False)

	# Calculate spot intensities at desired radii and in desired channels
	calc_custom_intensity(model, settings, trackmate, 'MEAN_INT_R0_C1', 'RADIUS_R0', r0, imps[0])
	calc_custom_intensity(model, settings, trackmate, 'MEAN_INT_R1_C1', 'RADIUS_R1', r1, imps[0])
	calc_custom_intensity(model, settings, trackmate, 'MEAN_INT_R0_C2', 'RADIUS_R0', r0, imps[1])
	calc_custom_intensity(model, settings, trackmate, 'MEAN_INT_R1_C2', 'RADIUS_R1', r1, imps[1])

  	# Calculate spot intensities with local background correction
  	add_spot_feature(model,'INT_C1_CORR_LOC','Total_Intensity_LocalBkgndCorr_C1',
  						'Int_LBC_C1',INTENSITY,False)
  	add_spot_feature(model,'INT_C2_CORR_LOC','Total_Intensity_LocalBkgndCorr_C2',
  						'Int_LBC_C2',INTENSITY,False)

	cal = imp.getCalibration()
	calc_spot_intensity_local_bkgnd (cal, model, 'RADIUS_R0', 'RADIUS_R1',
		'MEAN_INT_R0_C1', 'MEAN_INT_R1_C1', 'INT_C1_CORR_LOC')
	calc_spot_intensity_local_bkgnd (cal, model, 'RADIUS_R0', 'RADIUS_R1',
		'MEAN_INT_R0_C2', 'MEAN_INT_R1_C2', 'INT_C2_CORR_LOC')

	return

def global_bkgnd_correction_2channel (model, settings, trackmate, imp, r0):
	"""Globally correct for spot backgrounds in two channels."""

	# Split the source image into channels
	imps = ChannelSplitter.split(imp)

	# Create features to store all the new intensity measurements
	add_spot_feature(model,'INT_GLOBAL_C1','Global_Corrected_Intensity_C1','Global_C1',INTENSITY,False)
	add_spot_feature(model,'INT_GLOBAL_C2','Global_Corrected_Intensity_C2','Global_C2',INTENSITY,False)

	calc_spot_intensity_global_bkgnd (model, settings, trackmate, imps[0],
		'INT_GLOBAL_C1', custom_radius=r0)
	calc_spot_intensity_global_bkgnd (model, settings, trackmate, imps[1],
		'INT_GLOBAL_C2', custom_radius=r0)

	return

def run_trackmate(	img_path,
					out_dir,
					fname_prefix,
				  	detector_settings,
				  	tracker_settings,
				  	bkgnd_corr_settings=None,
				  	spot_qual=False,
				  	min_track_dur=3):

	"""Run TrackMate analysis on a given image file.

	Heavily borrowed from the Scripting TrackMate tutorial
	on the TrackMate wiki. Returns the final settings object
	"""

	imp = IJ.openImage(img_path)
	#imp.show()

	#------------------------
	# Prepare model object
	#------------------------

	model = Model()

	# Send all messages to ImageJ log window.
	model.setLogger(Logger.IJ_LOGGER)

	#------------------------
	# Prepare settings object
	#------------------------

	settings = Settings()
	settings.setFrom(imp)

	# Configure detector - We use the Strings for the keys
	settings.detectorFactory = LogDetectorFactory()

	settings.detectorSettings = detector_settings

	# Configure spot filters - Classical filter on quality
	if spot_qual:
		# Apply spot quality filter if spot_qual is a number.
		# 30 is a good ballpark default.
		filter1 = FeatureFilter('QUALITY', spot_qual, True)
		settings.addSpotFilter(filter1)
	else:
		filter1 = FeatureFilter('QUALITY', 0, True)
		settings.addSpotFilter(filter1)

	# Configure tracker - disable merges and fusions
	settings.trackerFactory = SparseLAPTrackerFactory()
	settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap()
	#print(LAPUtils.getDefaultLAPSettingsMap())

	for key, value in tracker_settings.items():
		settings.trackerSettings[key] = tracker_settings[key]

	# Configure track analyzers - Later on we want to filter out tracks
	# based on their displacement, so we need to state that we want
	# track displacement to be calculated. By default, out of the GUI,
	# not all features are calculated.

	# The displacement feature is provided by the TrackDurationAnalyzer.
	settings.addTrackAnalyzer(TrackDurationAnalyzer())
	settings.addSpotAnalyzerFactory(SpotIntensityAnalyzerFactory())

	# Configure track filters - We want to get rid of overly short tracks
	filter2 = FeatureFilter('TRACK_DURATION', min_track_dur, True)
	settings.addTrackFilter(filter2)

	#-------------------
	# Instantiate plugin
	#-------------------

	trackmate = TrackMate(model, settings)

	#--------
	# Process
	#--------

	ok = trackmate.checkInput()
	if not ok:
		sys.exit(str(trackmate.getErrorMessage()))

	ok = trackmate.process()
	if not ok:
		sys.exit(str(trackmate.getErrorMessage()))

	#-------------------------------
	# Perform background corrections
	#-------------------------------
	# Unpack background correction settings:
	if bkgnd_corr_settings:
		do_local_bkgnd_corr = bkgnd_corr_settings['do_local_bkgnd_corr']
		local_bkgnd_radii = bkgnd_corr_settings['local_bkgnd_radii']
		do_global_bkgnd_corr = bkgnd_corr_settings['do_global_bkgnd_corr']
	else: # use defaults
		do_local_bkgnd_corr=True,
		do_global_bkgnd_corr=True,
		local_bkgnd_radii=None

	if do_local_bkgnd_corr:
		# Perform local background correction
		if local_bkgnd_radii: # use custom radii, if specified
			r0 = local_bkgnd_radii[0]
			r1 = local_bkgnd_radii[1]
		else:
			# Get r0 (the tracked spot's radius)
			r0 = settings.detectorSettings['RADIUS']
			# By default, the background correctoin radius is 4x the inner spot radius
			r1=r0*4

		local_bkgnd_correction_2channel_2radii (model, settings, trackmate, imp, r0, r1)

	if do_global_bkgnd_corr:
		# Perform global background correction
		global_bkgnd_correction_2channel (model, settings, trackmate, imp, r0)

	#----------------
	# Display results
	#----------------
	display_results = False

	if display_results:
		selectionModel = SelectionModel(model)
		displayer = HyperStackDisplayer(model, selectionModel, imp)
		displayer.render()
		displayer.refresh()

	# Echo results with the logger we set at start:
	model.getLogger().log(str(model))

	#----------------
	# Save results
	#----------------
	if isinstance(out_dir, File):
		out_dir = out_dir.getAbsolutePath()
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	img_name = os.path.split(img_path)[1]
	img_name_noext = os.path.splitext(img_name)[0]

	fname_out = fname_prefix + img_name_noext + '.xml'
	full_path_out = os.path.join(out_dir, fname_out)

	outfile = TmXmlWriter(File(full_path_out))
	outfile.appendModel(model)
	outfile.appendSettings(settings)
	outfile.writeToFile()

	# close file
	imp.close()
	#outfile = None;
	return settings


##### Code to run when called as a script ###################################
if __name__ in ['__builtin__','__main__']:
    # Load images and run TrackMate on them using user-specified parameters
	image_paths = get_images_in_dir(import_dir,
	                               	split_string(file_types),
	                               	split_string(filters),
	                               	do_recursive
	                              	)
	for image_path in image_paths:
		print("Analyzing image", os.path.split(image_path)[1])
		if preserve_subdir_struct:
			curr_image_dir = os.path.split(image_path)[0]
			relpath = os.path.relpath(curr_image_dir, start=import_dir.getAbsolutePath())
			out_dir_curr = os.path.join(out_dir.getAbsolutePath(), relpath)
		else:
			out_dir_curr = out_dir


		settings_C1 = run_trackmate(image_path, out_dir_curr, 'C1_',
        				detector_settings_C1,
        				tracker_settings,
        				bkgnd_corr_settings,
        				spot_qual=False,
        				min_track_dur=min_track_duration)

		settings_C2 = run_trackmate(image_path, out_dir_curr, 'C2_',
        				detector_settings_C2,
        				tracker_settings,
        				bkgnd_corr_settings,
        				spot_qual=False,
        				min_track_dur=min_track_duration)


	# Save settings metadata
	save_settings( 	settings_out,
					out_dir,
					tracker_settings,
				  	detector_settings_C1,
				  	detector_settings_C2,
				  	bkgnd_corr_settings,
				  	min_track_duration,
				  	settings_C1,
				  	settings_C2)

	print('Done')
