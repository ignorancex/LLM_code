# tf_robot_learning, a all-around tensorflow library for robotics.
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Emmanuel Pignat <emmanuel.pignat@idiap.ch>,
#
# This file is part of tf_robot_learning.
#
# tf_robot_learning is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# tf_robot_learning is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_robot_learning. If not, see <http://www.gnu.org/licenses/>.

import tensorflow as tf
from .robot import Robot
from .three_joint import ThreeJointRobot
from tensorflow_probability import distributions as ds

pi = 3.14159

class BimanualThreeJointRobot(Robot):
	def __init__(self, q=None, dq=None, ddq=None, ls=None, session=None):
		Robot.__init__(self)

		self._ls = tf.constant([0.25, 0.25, 0.25, 0.25, 0.25]) if ls is None else ls


		margin = 0.02
		self._joint_limits = tf.constant([[0. + margin, pi - margin],
										  [-pi + margin, pi - margin],
										  [-pi + margin, pi - margin],
										  [-pi + margin, pi - margin],
										  [-pi + margin, pi - margin]],
										 dtype=tf.float32)
		self._arms = [
			ThreeJointRobot(ls=tf.gather(self._ls, [0, 1, 2])),
			ThreeJointRobot(ls=tf.gather(self._ls, [0, 1, 2])),
		]

		self._dof = 5

	def joint_limit_cost(self, q, std=0.1):
		qs = [q[:, 0:3], tf.concat([q[:, 0][:, None], q[:, 3:5]], axis=1)]
		return self._arms[0].joint_limit_cost(qs[0], std=std) + self._arms[0].joint_limit_cost(qs[1], std=std)

	def xs(self, q, concat=True):
		if q.shape.ndims == 1:
			qs = [q[0:3], tf.concat([q[0][None], q[3:5]], axis=0)]
			return tf.concat([self._arms[0].xs(qs[0]), self._arms[1].xs(qs[0])], axis=0)

		else:
			qs = [q[:, 0:3], tf.concat([q[:, 0][:, None], q[:, 3:5]], axis=1)]

			fks = [
				self._arms[0].xs(qs[0]),
				self._arms[1].xs(qs[1])
			]
			if concat:
				return tf.concat(fks, axis=1)
			else:
				return fks

