# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch as th
from numpy import inf,max

class _Registration():
    def __init__(self, verbose=True):
        # transformation of the image
        self._transformation = None

        # image similarity measure
        self._image_loss = None

        # optimizer
        self._optimizer = None
        self._number_of_iterations = 100

        self._displacement = None

        self._verbose=verbose
        self.loss=inf

    def set_optimizer(self, optimizer, lr):
        self._optimizer = optimizer
        self._lr = lr

    def set_number_of_iterations(self, number_of_iterations):
        self._number_of_iterations = number_of_iterations

    def set_transformation(self, transformation):
        self._transformation = transformation

    def set_image_loss(self, loss):
        self._image_loss = loss


class _PairwiseRegistration(_Registration):
    def __init__(self, verbose=True):
        super(_PairwiseRegistration, self).__init__(verbose=verbose)

        # regulariser on the displacement
        self._regulariser_displacement = []

        # regulariser on the parameters
        self._regulariser_parameter = []

    def set_regulariser_displacement(self, regulariser_displacement):
        self._regulariser_displacement = regulariser_displacement

    def set_regulariser_parameter(self, regulariser_parameter):
        self._regulariser_parameter = regulariser_parameter


class _GroupwiseRegistration(_Registration):
    def __init__(self, verbose=True):
        super(_GroupwiseRegistration, self).__init__(verbose=verbose)

        self._images = None

    def SetImages(self, images):
        self._images = images


class _ImageSeriesRegistration(_Registration):
    def __init__(self, verbose=True):
        super(_GroupwiseRegistration, self).__init__(verbose=verbose)

        self._image_series = None
        self._fixed_image = None

    def SetImageSeries(self, images):
        self._images = images

    def SetFixedImage(self, image):
        self._fixed_image = image


class PairwiseRegistration(_PairwiseRegistration):
    def __init__(self, verbose=True):
        super(PairwiseRegistration, self).__init__(verbose=verbose)

    def _closure(self):
        self._optimizer.zero_grad()

        displacement = self._transformation()

        # compute the image loss
        lossList = []
        loss_names = []
        for image_loss in self._image_loss:
             lossList.append(image_loss(displacement))
             loss_names.append(image_loss.name)

        # compute the regularisation loss on the displacement
        for reg_disp in self._regulariser_displacement:
            lossList.append(reg_disp(displacement))
            loss_names.append(reg_disp.name)

        # compute the regularisation loss on the parameter
        for reg_param in self._regulariser_parameter:
            lossList.append(reg_param(self._transformation.named_parameters()))
            loss_names.append(reg_param.name)

        if self._verbose:
            for loss_value, loss_name in zip(lossList, loss_names):
                print(str(loss_name) + ": " + str(loss_value.data.item()) + " ", end='', flush=True)
            print("")

        # sum up all loss terms
        loss = sum(lossList)

        loss.backward()

        return loss

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def adjust_learning_rate(self, optimizer, i_iter, lr, num_stemps, power):
        """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
        lr = self.lr_poly(lr, i_iter, num_stemps, power)
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def start(self, EarlyStopping=False, StopPatience=10):

        if EarlyStopping:
            from copy import deepcopy
            n = 0
            try:
                self.loss
            except:
                self.loss=inf

        for iter_index in range(self._number_of_iterations):
            if self._verbose:
                print(str(iter_index) + " ", end='', flush=True)

            if iter_index == self._number_of_iterations / 3:
                self._optimizer.param_groups[0]['lr'] = self._lr / 10.

            elif iter_index == 2 * self._number_of_iterations / 3:
                self._optimizer.param_groups[0]['lr'] = self._lr / 10.
                # self.adjust_learning_rate(self._optimizer, iter_index - self._number_of_iterations / 2, self._lr,
                #                           self._number_of_iterations, 0.9)
            # elif iter_index == self._number_of_iterations / 2:
            #     self._optimizer.param_groups[0]['lr'] = self._lr / 100.
            # # elif iter_index == self._number_of_iterations / 2:
            #     self.adjust_learning_rate(self._optimizer, iter_index - self._number_of_iterations / 2, self._lr / 10., self._number_of_iterations, 0.9)

            #print(self._optimizer.param_groups[0]['lr'])

            loss = self._optimizer.step(self._closure)
            if EarlyStopping:
                if loss < self.loss:
                    n = 0
                    self.loss=loss
                    best=deepcopy(self._transformation)
                else:
                    n += 1
                if n > StopPatience:
                    self._transformation = best
                    return
        self.loss = loss


class DemonsRegistraion(_Registration):
    def __init__(self, verbose=True):
        super(DemonsRegistraion, self).__init__(verbose=verbose)

        # regulariser on the displacement
        self._regulariser = []

    def set_regulariser(self, regulariser):
            self._regulariser = regulariser

    def _closure(self):
        self._optimizer.zero_grad()

        displacement = self._transformation()

        # compute the image loss
        lossList = []
        loss_names = []
        for image_loss in self._image_loss:
            lossList.append(image_loss(displacement))
            loss_names.append(image_loss.name)

        if self._verbose:
            for loss_value, loss_name in zip(lossList, loss_names):
                print(str(loss_name) + ": " + str(loss_value.data.item()) + " ", end='', flush=True)

            print("")

        # sum up all loss terms
        loss = sum(lossList)

        loss.backward()

        return loss

    def start(self):

        for iter_index in range(self._number_of_iterations):
            if self._verbose:
                print(str(iter_index) + " ", end='', flush=True)

            loss = self._optimizer.step(self._closure)

            for regulariser in self._regulariser:
                regulariser.regularise(self._transformation.parameters())


