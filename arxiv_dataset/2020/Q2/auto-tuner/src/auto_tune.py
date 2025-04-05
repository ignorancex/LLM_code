"""
Author: Renato Durrer
Created: 23.05.2019

"""
import os
import pickle
import time
import numpy as np

from Labber import LogFile, ScriptTools
from keras.models import model_from_json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.utils.measurement_funcs import measure
from src.utils.funcs import data_creator, preprocessor


class AutoTuner:
    """
    Class that performs the auto tuning of charge states for a double quantum dot system.

    Example
    -------
    # Instantiate
    tune = AutoTuner(f=12, p=8,
                         gate_names=gate_names, path_in=path_in, path_out=path_out,
                         transition_recognizer=path_to_model,
                         reference_recognizer=Path_to_model)

    # find reference point
    tune.find_reference(16, 4, x_0=x_0, y_0=y_0, res=7.5e-3, confidence=0.5, calibrate_nr=3, plot=False)

    tune.tune(1, 1, plot=False, calibrate_nr=5)  # perform the tuning
    tune.save_tuning_info(filepath)  # create dict containing tuning information
    tune.path_plotter(1e-3)  # plot the path

    """
    def __init__(self, f=None, p=None,
                 gate_names=None, path_in=None, path_out=None,
                 transition_recognizer='', reference_recognizer=''):
        """

        Parameters
        ----------
        f : int
            frame size
        p : int
            padding size

        gate_names : dict
                     A dict containing necessary information about certain gates.
                     It must contain the names of PG1, PG2, name of the log channel
                     for the current through the DQD, the name of the log channel
                     for the current through the QPC and the name of the QPC gate.

                     Example:
                     gate_names = {
                          'PG1': 'LPG',
                          'PG2': 'MPG',
                          'I_DQD': 'I TQD',
                          'I_QPC': 'I QPC',
                          'QPC_G': 'QPC_M'
                     }

        path_out: os.path
                  path where the measurement configuration file lies

        path_in: os.path
                 path where the measurement files are saved

        transition_recognizer : object
                                object possessing method predict that classifies charge
                                transitions for all three corners of the drawn frame
        reference_recognizer : object
                               object possessing method predict that classifies
                               whether or not QDs are empty for a given point
        """
        # frame definition
        self.f = f
        self.p = p

        # current charge ocupation number for left & right QD
        self.occupation_1 = None
        self.occupation_2 = None

        # desired charge occupation number for left & right QD
        self.occupation_1_f = None
        self.occupation_2_f = None

        # define gate names
        self.gate_names = gate_names

        # define plunger gate information
        self.PG1 = {
            'name': gate_names['PG1']
        }
        self.PG2 = {
            'name': gate_names['PG2']
        }

        # define variables for path that is taken
        self.trans_path_x = []  # PG1 values for the transition path
        self.trans_path_y = []  # PG2 values for the transition path
        self.ref_path_x = []  # PG1 values for the reference path
        self.ref_path_y = []  # PG2 values for the reference path
        self.occupation = []  # store charge occupation for each point in the path

        # store all classification outcomes for all frames
        self.trans_classif = []  # transition classifications
        self.ref_classif = []  # reference classifications

        # define lists for measurements
        self.ref_frames = []  # all measured frames for the reference point
        self.trans_frames = []  # all measured frames for the transition part
        self.trans_frame_start = []  # starting point for all transition frames, (PG1, PG2)

        # Variables for storing the final path
        self.meas_path = None  # one matrix that contains all measured transition frames
        self.x = None  # PG1 voltages for self.meas_path (x-axis)
        self.y = None  # PG2 voltages for self.meas_path (y-axis)

        # define paths where configuration file lies and path where measurements shall be stored
        self.path_in = path_in
        self.path_out = path_out

        # QPC operating point
        self.sweet_spot = None

        # machine learning model for recognizing charge transitions
        self.trans_rec = self.load_model_(classification_type='transition',
                                          model_name=transition_recognizer)
        self.occupation_ref_rec = self.load_model_(classification_type='reference',
                                                   model_name=reference_recognizer)

    def find_reference(self, f, b, x_0, y_0, res=8e-3, calibrate_nr=3, confidence=0.7, plot=False):
        """
        Finds a plunger gate voltage configuration for which the charge occupation is
        given as (0,0). That is, both dots are empty. This plunger gate voltage configuration
        can then be used as a reference point in the charge occupation tuning process.

        Needs a boundary for the plunger gate voltages from which a random starting point is chosen.

        Parameters
        ----------
        f : int
            coarse frame size

        b : int
            coarse frame border

        x_0 : float
              starting point for PG1 from which a reference point is searched.

        y_0 : float
              starting point for PG2 from which a reference point is searched.

        res : float, optional
              default: 8e-3
              coarse resolution in V

        calibrate_nr : int, optional
                       default: 3
                       How often the QPC is calibrated. If calibrate_nr == 3,
                       before measuring every third frame, the QPC is calibrated.

        confidence : float, optional
                     default: 0.8
                     Determines the confidence threshold with which we recognize the empty region.

        plot: bool, optional
              default: False
              If True, a plot of every frame plus its convolution filters is made.

        Returns
        -------
        float, float

        Voltage for PG1 and PG2 for which both quantum dots are unoccupied.
        """
        found = False
        counter = 0
        self.PG1['n_pts'] = f + b + 1
        self.PG2['n_pts'] = f + b + 1

        self.ref_path_x.append(x_0)
        self.ref_path_y.append(y_0)

        while not found:
            print('\n#####################'
                  '\n Frame nr. {}'.format(counter+1))
            # update measurement information
            self.PG1['start'] = self.ref_path_x[-1] + (b + 0.5) * res
            self.PG1['stop'] = self.ref_path_x[-1] - (f + 0.5) * res
            self.PG1['mean'] = self.PG1['start'] + (self.PG1['stop'] - self.PG1['start']) / 2.

            self.PG2['start'] = self.ref_path_y[-1] + (b + 0.5) * res
            self.PG2['stop'] = self.ref_path_y[-1] - (f + 0.5) * res
            self.PG2['mean'] = self.PG2['start'] + (self.PG2['stop'] - self.PG2['start']) / 2.

            # define measurement object
            output_path = os.path.join(self.path_out, 'reference_' + str(counter))
            measurement = ScriptTools.MeasurementObject(
                self.path_in,
                output_path
            )

            # recalibrate QPC every 3 frames
            if counter % calibrate_nr == 0 or self.sweet_spot is None:
                calibrate = True
                gate_config = None

            else:
                calibrate = False
                gate_config = {
                    self.gate_names['QPC_G']: self.sweet_spot
                }
            
            # perform measurement / get data
            measurement_signal = self.get_data_(measurement, output_path,
                                                DQD_log_channel=self.gate_names['I_DQD'],
                                                calibrate=calibrate, rescale=False,
                                                config=gate_config)
            self.ref_frames.append(measurement_signal['I_QPC'])
            I_DQD = measurement_signal['I_DQD']

            # reshape data in order to make it suitable for classifier
            reshaped_signal = self.ref_frames[-1].reshape((1, f + b, f + b, 1))

            # predict occupation state
            occupation = self.occupation_ref_rec.predict(reshaped_signal)[0]
            self.ref_classif.append(occupation)
            print('Classification confidences:\n{}'.format(occupation))
            print('PG1: {}V\n'
                  'PG2: {}V'.format(self.ref_path_x[-1], self.ref_path_y[-1]))
            counter += 1

            # plot measurement and visualize filters
            if plot:
                grid = plt.GridSpec(20, 20)
                fig = plt.figure()
                ax = plt.subplot(grid[:20, :20])
                axins1 = inset_axes(ax,
                                    width="3%",
                                    height="100%",
                                    loc='lower left',
                                    bbox_to_anchor=(1.01, 0., 1, 1),
                                    bbox_transform=ax.transAxes,
                                    borderpad=0,
                                    )

                im1 = ax.pcolormesh(self.ref_frames[-1][:, :], linewidth=0, rasterized=True)
                cbar = fig.colorbar(im1, cax=axins1)
                ax.axhline(y=16, color='black', linewidth=2)
                ax.axvline(x=16, color='black', linewidth=2)
                ax.plot(self.ref_path_x[-1], self.ref_path_y[-1])
                ax.set_ylim(0, 20)
                ax.set_xlim(0, 20)
                plt.title('Reference {}'.format(counter))
                plt.show()

            # If confidence that DQD empty is larger than a certain threshold -> terminate
            # Classification outcome: [1, 0] -> dots occupied, [0, 1] -> dots empty
            if(occupation[1] > confidence and not self.is_current_(I_DQD, threshold=7e-12)):
                found = True
                print(self.ref_path_x[-1])
                print(self.ref_path_y[-1])
                self.occupation_1 = 0
                self.occupation_2 = 0
                self.trans_path_x.append(self.ref_path_x[-1])
                self.trans_path_y.append(self.ref_path_y[-1])
                print('Found a reference point at\n'
                      'PG1: {}V\n'
                      'PG2: {}V'.format(self.ref_path_x[-1], self.ref_path_y[-1]))

            else:
                self.ref_path_x.append(self.ref_path_x[-1] - f*res/3)
                self.ref_path_y.append(self.ref_path_y[-1] - f*res/3)

        return self.ref_path_x[-1], self.ref_path_y[-1]

    def tune(self, occupation_1, occupation_2, res=1e-3, calibrate_nr=3, plot=False):
        """
        Finds the desired charge regime, given a reference point.

        Parameters
        ----------
        occupation_1 : int
                       desired charge occupation number for left dot

        occupation_2 : int
                       desired charge occupation number for right dot

        res: float, optional
             default: 1e-3
             Resolution with which frames are measured.

        calibrate_nr: int, optional
                      default: 3
                      How often the QPC is calibrated. If calibrate_nr == 3,
                      before measuring every third frame, the QPC is calibrated.

        plot: bool, optional
              default: False
              If True, all measurements taken are plotted.

        Returns
        -------
        tuple
            PG1 & PG2 voltage
        """
        # initialize variables
        self.occupation_1_f = occupation_1
        self.occupation_2_f = occupation_2

        # direction variable defines where the next frame is drawn
        # direction == 0: lower right corner (i.e. right)
        # direction == 1: upper right corner (i.e. top-right)
        # direction == 2: upper left corner (i.e. up)
        direction = 1
        found = False
        reverse = False  # if charge occupation too high, go back
        counter = 0  # length of path in frames

        print('Start tuning Charge States.')
        self.PG1['n_pts'] = self.f + 2 * self.p + 1
        self.PG2['n_pts'] = self.f + 2 * self.p + 1

        while not found:
            # define measurement variables
            self.PG1['stop'] = self.trans_path_x[-1] - (self.p + 0.5) * res
            self.PG1['start'] = self.trans_path_x[-1] + (self.f + self.p + 0.5) * res
            self.PG2['stop'] = self.trans_path_y[-1] - (self.p + 0.5) * res
            self.PG2['start'] = self.trans_path_y[-1] + (self.f + self.p + 0.5) * res

            # overwrite when necessary
            if reverse:
                if direction == 0:
                    self.PG1['stop'] = self.trans_path_x[-1] - (self.f + self.p + 0.5) * res
                    self.PG1['start'] = self.trans_path_x[-1] + (self.p + 0.5) * res
                if direction == 2:
                    self.PG2['stop'] = self.trans_path_y[-1] - (self.f + self.p + 0.5) * res
                    self.PG2['start'] = self.trans_path_y[-1] + (self.p + 0.5) * res

            # add starting point for plotting reasons
            self.trans_frame_start.append((self.PG1['start'], self.PG2['start']))

            # define mean PG values for QPC calibration
            self.PG2['mean'] = self.PG2['start'] + (self.PG2['stop'] - self.PG2['start']) / 2.
            self.PG1['mean'] = self.PG1['start'] + (self.PG1['stop'] - self.PG1['start']) / 2.

            # update charge occupation into list
            self.occupation.append((self.occupation_1, self.occupation_2))

            # update path to take
            if direction == 0:
                if not reverse:
                    self.trans_path_x.append(self.trans_path_x[-1] + self.f * res)
                    self.trans_path_y.append(self.trans_path_y[-1])
                else:
                    self.trans_path_x.append(self.trans_path_x[-1] - self.f * res)
                    self.trans_path_y.append(self.trans_path_y[-1])

            elif direction == 1:
                if not reverse:
                    self.trans_path_x.append(self.trans_path_x[-1] + self.f * res)
                    self.trans_path_y.append(self.trans_path_y[-1] + self.f * res)
                else:  # this is never True - just for completeness
                    self.trans_path_x.append(self.trans_path_x[-1] - self.f * res)
                    self.trans_path_y.append(self.trans_path_y[-1] - self.f * res)

            elif direction == 2:
                if not reverse:
                    self.trans_path_x.append(self.trans_path_x[-1])
                    self.trans_path_y.append(self.trans_path_y[-1] + self.f * res)
                else:
                    self.trans_path_x.append(self.trans_path_x[-1])
                    self.trans_path_y.append(self.trans_path_y[-1] - self.f * res)

            # Print path number and what points the path links
            print('\n###################\n'
                  'Path: {}\n'
                  'PG1: {} -> {}\n'
                  'PG2: {} -> {}'.format(counter,
                                         round(self.trans_path_x[-2], 3), round(self.trans_path_x[-1], 3),
                                         round(self.trans_path_y[-2], 3), round(self.trans_path_y[-1], 3)))

            # define measurement object
            output_path = os.path.join(self.path_out, 'tune_' + str(counter))
            measurement = ScriptTools.MeasurementObject(
                self.path_in,
                output_path
            )

            # recalibrate QPC every calibrate_nr frames
            if counter % calibrate_nr == calibrate_nr - 1 or self.sweet_spot is None:
                calibrate = True
                gate_config = None

            else:
                calibrate = False
                gate_config = {
                    self.gate_names['QPC_G']: self.sweet_spot
                }

            # get the measured QPC signal and reshape it
            measurement_signal = self.get_data_(measurement, output_path, DQD_log_channel=self.gate_names['I_DQD'],
                                                calibrate=calibrate, rescale=True, config=gate_config)
            self.trans_frames.append(measurement_signal['I_QPC'])
            I_DQD = measurement_signal['I_DQD']
            reshaped_signal = self.trans_frames[-1].reshape((1, self.f + 2*self.p, self.f + 2*self.p, 1))

            # check whether there is a current through the dot, if so - abort
            # TODO introduce threshold as a variable
            if self.is_current_(I_DQD, 7e-12):
                print('There is too much current through the dot. Tuning stopped.\n'
                      'Last PG values:\n'
                      'PG1: {}\n'
                      'PG2: {}'.format(self.trans_path_x[-1], self.trans_path_y[-1]))
                break

            # Make the predictions
            # (QD1 transition, QD2 transition)
            # transition[i] == [1, 0, 0, 0]: (False, False)
            # transition[i] == [0, 1, 0, 0]: (False, True)
            # transition[i] == [0, 0, 1, 0]: (True, False)
            # transition[i] == [0, 0, 0, 1]: (True, True)
            # for i in {0, 1, 2}
            transition = self.trans_rec.predict(reshaped_signal)
            self.trans_classif.append(transition)

            # Check whether there is a contradiction. If there is one,
            # measure the same frame again and redo the classifications.
            is_contra = self.is_contradiction_(transition)
            if is_contra:
                print('There is a contradiction.\n'
                      '{}'.format(transition))
                # Remeasure the frame.
                self.trans_frames.append(
                    self.get_data_(measurement, output_path, calibrate=calibrate, config=gate_config)['I_QPC'])
                # Redo the classifications
                reshaped_signal = self.trans_frames[-1].reshape((1, self.f + 2 * self.p, self.f + 2 * self.p, 1))
                transition = self.trans_rec.predict(reshaped_signal)

                # check again whether there is still a contradiction
                if self.is_contradiction_(transition):
                    print('There is still a contradiction. Proceed anyways.')
                else:
                    print('There is no contradiction anymore. Proceed as normal.')

            # plot the data
            if plot:
                pass
                # grid = plt.GridSpec(20, 20)
                # fig = plt.figure()
                # ax = plt.subplot(grid[:20, :20])
                # axins1 = inset_axes(ax,
                #                     width="3%",
                #                     height="100%",
                #                     loc='lower left',
                #                     bbox_to_anchor=(1.01, 0., 1, 1),
                #                     bbox_transform=ax.transAxes,
                #                     borderpad=0,
                #                     )
                #
                # im1 = ax.pcolormesh(self.trans_frames[-1][:, :], linewidth=0, rasterized=True)
                # cbar = fig.colorbar(im1, cax=axins1)
                # ax.axhline(y=self.p, color='black', linewidth=2)
                # ax.axhline(y=self.f+self.p, color='black', linewidth=2)
                # ax.axvline(x=self.p, color='black', linewidth=2)
                # ax.axvline(x=self.f+self.p, color='black', linewidth=2)
                # ax.plot(self.trans_path_x[-1], self.trans_path_y[-1])
                # ax.set_ylim(0, self.f+2*self.p)
                # ax.set_xlim(0, self.f+2*self.p)
                # plt.title('Path {}'.format(counter))
                # plt.show()

            # Trigger charge transitions
            if np.argmax(transition[direction]) == 0:  # no transition at all
                print('No transition.\n'
                      'Confidence: {}\n'
                      'Current charge occupation({},{})'.format(np.max(transition[direction]),
                                                                self.occupation_1, self.occupation_2))
            elif np.argmax(transition[direction]) == 1:  # transition in QD2
                if not reverse:
                    self.occupation_2 += 1
                else:
                    self.occupation_2 -= 1
                print('Transition in QD2.\n'
                      'Confidence: {}\n'
                      'Current charge occupation({},{})'.format(np.max(transition[direction]),
                                                                self.occupation_1, self.occupation_2))
            elif np.argmax(transition[direction]) == 2:  # transision in QD1
                if not reverse:
                    self.occupation_1 += 1
                else:
                    self.occupation_1 -= 1
                print('Transition in QD1.\n'
                      'Confidence: {}\n'
                      'Current charge occupation({},{})'.format(np.max(transition[direction]),
                                                                self.occupation_1, self.occupation_2))
            elif np.argmax(transition[direction]) == 3:  # transition in both QDs
                if not reverse:
                    self.occupation_2 += 1
                    self.occupation_1 += 1
                else:
                    self.occupation_2 -= 1
                    self.occupation_1 -= 1
                print('Transition in both QDs.\n'
                      'Confidence: {}\n'
                      'Current charge occupation({},{})'.format(np.max(transition[direction]),
                                                                self.occupation_1, self.occupation_2))

            # define where to go based on current charge occupation
            if self.occupation_1 < self.occupation_1_f and self.occupation_2 < self.occupation_2_f:
                direction = 1
                reverse = False
                print('-> Going up right.')

            elif self.occupation_1 == self.occupation_1_f and self.occupation_2 < self.occupation_2_f:
                direction = 2
                reverse = False
                print('-> Going up.')

            elif self.occupation_1 < self.occupation_1_f and self.occupation_2 == self.occupation_2_f:
                direction = 0
                reverse = False
                print('-> Going right.')

            # go back, if one of the dots has too many electrons
            elif self.occupation_1 > self.occupation_1_f:
                direction = 0
                reverse = True
                print('-> Going left.')

            elif self.occupation_2 > self.occupation_2_f:
                direction = 2
                reverse = True
                print('-> Going down.')

            # Terminate if desired charge configuration is reached
            elif self.occupation_1 == self.occupation_1_f and self.occupation_2 == self.occupation_2_f:
                print('\nTuning successful\n'
                      'PG1: {}V\n'
                      'PG2: {}V'.format(self.trans_path_x[-1], self.trans_path_y[-1]))
                found = True

            if counter >= 15:
                print('Could not find desired charge configuration.\n'
                      'Program stopped.')
                found = True

            counter += 1

    def path_plotter(self, res):
        """
        Method that plots the current path.

        Parameters
        ----------
        res : float,
              Resolution of the measurement,
              i.e. difference of the PG voltage between two measured points. (in V)

        Returns
        -------
        Void
        """
        # define edgepoint of the plot
        x_start = np.min(self.trans_path_x) - self.p * res
        x_end = np.max(self.trans_path_x) + (self.f + self.p) * res
        y_start = np.min(self.trans_path_y) - self.p * res
        y_end = np.max(self.trans_path_y) + (self.f + self.p) * res

        # define length of arrays
        x_len = int((x_end - x_start) / res)
        y_len = int((y_end - y_start) / res)

        # define x- and y-axis
        self.x = np.arange(x_start, x_end, res)
        self.y = np.arange(y_start, y_end, res)

        # define matrix that will be plotted
        self.meas_path = np.ones((y_len, x_len))

        # fill the matrix with the measured frames
        for k, frame in enumerate(self.trans_frames):
            start = ((self.trans_frame_start[k][0] - x_start) / res,
                     (self.trans_frame_start[k][1] - y_start) / res)
            end = ((self.trans_frame_start[k][0] + (2 * self.p + self.f) * res - x_start) / res,
                   (self.trans_frame_start[k][1] + (2 * self.p + self.f) * res - y_start) / res)
            # start = (int((self.trans_path_x[k] - self.p * res - x_start) / res),
            #          int((self.trans_path_y[k] - self.p * res - y_start) / res))
            # end = (int((self.trans_path_x[k] + (self.p + self.f) * res - x_start) / res),
            #        int((self.trans_path_y[k] + (self.p + self.f) * res - y_start) / res))
            self.meas_path[start[1]:end[1], start[0]:end[0]] = frame

        # Plot the path
        fig, ax = plt.subplots(1)
        ax.pcolormesh(self.x, self.y, self.meas_path)
        ax.plot(self.trans_path_x, self.trans_path_y, color='red')
        for k in range(len(self.trans_frames)):
            width = self.f * res
            rect = patches.Rectangle((self.trans_path_x[k], self.trans_path_y[k]), width, width,
                                     linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
        plt.show()

    def save_tuning_info(self, path):
        """
        Saves information of a tuning run as a dicte as a .pkl file.
        Parameters
        ----------
        path : os.path object
               Path to where the file is saved.

        Returns
        -------
        void

        File Structure
        --------------
        tuning_info = {
            'ref_path':             tuple of lists,  # (PG1, PG2) values for the reference frames
            'trans_path':           tuple of lists,  # (PG1, PG2) values for the transition frames
            'ref_frames':           list,  # measured reference frames
            'trans_frames':         list,  # measured transition frames
            'trans_frame_start':    list of tuples,  # starting point for each frame in voltage
            'trans_classifications':list,  # classification outcomes for each transition frame
            'ref_classifications':  list,  # classification outcomes for each reference frame
            'occupation':           list,  # charge occupation for each frame
            'path':                 [x, y, meas_path],  # transition frames sticked together
                                     x: voltages for PG1 (x-axis), y: voltages for PG2 (y-axis)
                                     meas_path: matrix with measurements of the transition path
            'config':               os.path,  # path to config file
            'output_path':          os.path,  # path to output_folder
        }
        """
        # define dictionary with all data
        tuning_info = {
            'ref_path': (self.ref_path_x, self.ref_path_y),  # (PG1, PG2) values for the reference frames
            'trans_path': (self.trans_path_x, self.trans_path_y),  # (PG1, PG2) values for the transition frames
            'ref_frames': self.ref_frames,  # measured reference frames
            'trans_frames': self.trans_frames,  # measured transition frames
            'trans_frame_start': self.trans_frame_start,  # starting point for each frame in voltage
            'trans_classifications': self.trans_classif,  # classification outcomes for each transition frame
            'ref_classifications': self.ref_classif,  # classification outcomes for each reference frame
            'occupation': self.occupation,  # charge occupation for each frame
            'path': [self.x, self.y, self.meas_path],  # transition frames sticked together
            'config': self.path_in,  # path to config file
            'output_path': self.path_out  # path to output_folder
        }

        # save it
        with open(path, 'wb') as f:
            pickle.dump(tuning_info, f)

    def get_data_(self, measurement, output, DQD_log_channel='',
                  return_step_channels=False, calibrate=True, rescale=True, config=None):
        """
        Performs measurement, saves and reads the data, processes it and returns the processed data.

        Parameters
        ----------
        measurement : Measurement Object
                      Measurement object with which measurement is performed

        output : os.path object
                 Output folder

        DQD_log_channel : string, optional
                          default = ''
                          Name of the DQD log channel. If equal to empty string '',
                          then only the QPC values are returned.

        return_step_channels : bool, optional
                               default = False
                               If True, the step channels for the plunger gates
                               are returned

        calibrate : Bool, optional
                    default: True
                    If True, QPC calibration is performed

        rescale : Bool, optionsl
                  default: True
                  If True, the standard deviation of the measured frame is again set to one
                  after derivative is taken.

        config : dict, {'Gate_Name': operation_voltage (float),...}
                 dict containing the set of gate voltages for the measurement

        Returns
        -------
        dict
        Key: 'I_QPC', value: pd.DataFrame -> QPC currents
        Key: 'I_DQD', value: pd.DataFrame -> DQD currents
        Key: 'step1', value: pd.DataFrame -> PG1 step channel
        Key: 'step2', value: pd.DataFrame -> PG2 step channel

        'I_DQD' is only present, if a DQD log channel is passed.
        'step1' and 'step2' are only present, if return_step_channels = True

        QPC values are processed by the function processor.
        """
        # perform measurement
        sweet_spot = measure(self.PG1, self.PG2, measurement, output,
                             gate_names=self.gate_names, calibrate=calibrate,
                             config=config, return_qpc_voltage=True, wait=True)

        if sweet_spot is not None:
            self.sweet_spot = sweet_spot

        # read the result
        file = LogFile(output + '.hdf5')

        # data is a dict with all step and log channels
        data = data_creator(file)
        QPC_signal = data[self.gate_names['I_QPC']]

        # perform preprocessing
        processed_signal = {}
        processed_signal['I_QPC'] = preprocessor(QPC_signal[::-1, ::-1], rescale=rescale)

        # return the DQD log channel
        if DQD_log_channel:
            DQD_signal = data[DQD_log_channel]
            processed_signal['I_DQD'] = DQD_signal

        # return the plunger gate step channels
        if return_step_channels:
            processed_signal['step1'] = data[self.PG1['name']]
            processed_signal['step2'] = data[self.PG2['name']]

        return processed_signal

    @staticmethod
    def load_model_(model_name='Architecture_2/Arch_2_2', classification_type='transition'):
        """
        loads a machine learning model that makes classifications

        Parameters
        ----------
        model_name: string
                    Name of the model to be load

        classification_type: string
                             type of the model, either 'transition' or 'reference'

        Returns
        -------
        scikit-learn model
        """
        # load json and create model
        json_file = open('../results/models/{}/{}.json'.format(classification_type, model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("../results/models/{}/{}.h5".format(classification_type, model_name))
        print("Loaded model from disk")
        return model

    @staticmethod
    def is_current_(I_DQD, threshold=None):
        """
        Checks whether or not a significiant current is flowing through
        the Double Quantum Dot. If so (output = True), the related DQD charge occupation
        will most likely not be (0,0).

        The threshold can be found by looking at the measured DQD current values.

        Parameters
        ----------
        I_DQD : array like
                Contains the measured current through the DQD

        threshold : float
                    Threshold for which it is decided whether or not current
                    is flowing through the double quantum dot.

        Returns
        -------
        bool,
        If True, there is significiant current flowing through the DQD.
        """
        # calculate mean
        is_curr = False
        I_mean = np.abs(np.mean(I_DQD))

        # check whether mean is higher than a set threshold
        if(I_mean > threshold):
            is_curr = True

        print('Checking Current\n'
              'I mean: {}\n'
              'Threshold: {}\n'.format(I_mean, threshold))
        return is_curr

    @staticmethod
    def is_contradiction_(transition):
        """
        Given the classification result of the charge transition classifier,
        check whether there are contradictions in the classifications

        Parameters
        ----------
        transition : array like,
                     classification result of
                     the charge transition classifier.

        Returns
        -------
        bool

        If True, there is at least one contradiction.
        If False, no contradiction could be detected.
        """
        is_contr = False

        # check implications of lower left corner
        if np.argmax(transition[0]) == 0:
            if np.argmax(transition[2]) == 2 or np.argmax(transition[2]) == 3:
                is_contr = True
        elif np.argmax(transition[0]) == 1:
            if np.argmax(transition[1]) == 0 or np.argmax(transition[1]) == 2:
                is_contr = True
            if np.argmax(transition[2]) != 1:
                is_contr = True
        elif np.argmax(transition[0]) == 2:
            if np.argmax(transition[1]) == 0 or np.argmax(transition[1]) == 1:
                is_contr = True
        elif np.argmax(transition[0]) == 3:
            if np.argmax(transition[1]) != 3:
                is_contr = True
            if np.argmax(transition[2]) == 0 or np.argmax(transition[2]) == 2:
                is_contr = True

        # check implicatiosn of upper right corner
        if np.argmax(transition[2]) == 0:
            if np.argmax(transition[0]) == 1 or np.argmax(transition[0]) == 3:
                is_contr = True
        elif np.argmax(transition[2]) == 1:
            if np.argmax(transition[1]) == 0 or np.argmax(transition[1]) == 2:
                is_contr = True
        elif np.argmax(transition[2]) == 2:
            if np.argmax(transition[0]) != 2:
                is_contr = True
            if np.argmax(transition[1]) == 0 or np.argmax(transition[1]) == 1:
                is_contr = True
        elif np.argmax(transition[2]) == 3:
            if np.argmax(transition[1]) != 3:
                is_contr = True
            if np.argmax(transition[0]) == 0 or np.argmax(transition[0]) == 1:
                is_contr = True

        return is_contr


if __name__ == "__main__":

    gate_names = {
        'PG1': 'MPG',
        'PG2': 'RPG',
        'I_DQD': 'I TQD',
        'I_QPC': 'I QPC',
        'QPC_G': 'QPC_R'
    }

    # perform repeated tuning for evaluation
    for run in range(0, 10, 1):
        path_in = os.path.abspath(r'C:\\Users\\Measure2\\Desktop\\measurement_series'
                                  r'\\2019\\08\\Data_0805\\465_Arch_2_MPG_RPG.hdf5')
        path_out = os.path.abspath(r'C:\\Users\\Measure2\\Desktop\\auto_tuner\\results\\outputs\\config1_reversed_(1,1)_{}'.format(run))
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        then = time.time()
        # Define Auto Tuner Instance
        tune = AutoTuner(f=12, p=8,
                         gate_names=gate_names, path_in=path_in, path_out=path_out,
                         transition_recognizer='s_grid_6/s_grid_6_8',
                         reference_recognizer='Binary_Grid_1/Binary_Grid_1_5')

        # define boundaries within which the starting point is drawn
        pg2_b = (45e-3, -145e-3)
        pg1_b = (80e-3, -110e-3)
        x_0 = (pg1_b[1] - pg1_b[0]) * np.random.random() + pg1_b[0]
        y_0 = (pg2_b[1] - pg2_b[0]) * np.random.random() + pg2_b[0]
        # find reference point
        tune.find_reference(16, 4, x_0=x_0, y_0=y_0, res=7.5e-3, confidence=0.5, calibrate_nr=3, plot=False)
        # perform tuning
        tune.tune(2, 1, plot=False, calibrate_nr=5)
        filepath = os.path.join(path_out, 'tuning_info.pkl')
        tune.save_tuning_info(filepath)
        # tune.path_plotter(1e-3)
        print('Tuning took {}s'.format(time.time() - then))
