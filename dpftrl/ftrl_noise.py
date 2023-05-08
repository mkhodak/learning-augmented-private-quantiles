# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The tree aggregation protocol for noise addition in DP-FTRL."""

import torch
from collections import namedtuple

from absl import app


class CummuNoiseTorch:
    @torch.no_grad()
    def __init__(self, std, shapes, device, test_mode=False):
        """
        :param std: standard deviation of the noise
        :param shapes: shapes of the noise, which is basically shape of the gradients
        :param device: device for pytorch tensor
        :param test_mode: if in test mode, noise will be 1 in each node of the tree
        """
        assert std >= 0
        self.std = std
        self.shapes = shapes
        self.device = device
        self.step = 0
        self.binary = [0]
        self.noise_sum = [torch.zeros(shape).to(self.device) for shape in shapes]
        self.recorded = [[torch.zeros(shape).to(self.device) for shape in shapes]]
        self.test_mode = test_mode

    @torch.no_grad()
    def __call__(self):
        """
        :return: the noise to be added by DP-FTRL
        """
        if self.std <= 0 and not self.test_mode:
            return self.noise_sum

        self.step += 1

        idx = 0
        while idx < len(self.binary) and self.binary[idx] == 1:
            self.binary[idx] = 0
            for ns, re in zip(self.noise_sum, self.recorded[idx]):
                ns -= re
            idx += 1
        if idx >= len(self.binary):
            self.binary.append(0)
            self.recorded.append([torch.zeros(shape).to(self.device) for shape in self.shapes])

        for shape, ns, re in zip(self.shapes, self.noise_sum, self.recorded[idx]):
            if not self.test_mode:
                n = torch.normal(0, self.std, shape).to(self.device)
            else:
                n = torch.ones(shape).to(self.device)
            ns += n
            re.copy_(n)

        self.binary[idx] = 1
        return self.noise_sum

    @torch.no_grad()
    def proceed_until(self, step_target):
        """
        Proceed until the step_target-th step. This is for the binary tree completion trick.

        :return: the noise to be added by DP-FTRL
        """
        if self.step >= step_target:
            raise ValueError(f'Already reached {step_target}.')
        while self.step < step_target:
            noise_sum = self.__call__()
        return noise_sum


Element = namedtuple('Element', 'height value')


class CummuNoiseEffTorch:
    """
    The tree aggregation protocol with the trick in Honaker, "Efficient Use of Differentially Private Binary Trees", 2015
    """
    @torch.no_grad()
    def __init__(self, std, shapes, device):
        """
        :param std: standard deviation of the noise
        :param shapes: shapes of the noise, which is basically shape of the gradients
        :param device: device for pytorch tensor
        """
        self.std = std
        self.shapes = shapes
        self.device = device

        self.step = 0
        self.noise_sum = [torch.zeros(shape).to(self.device) for shape in shapes]
        self.stack = []

    @torch.no_grad()
    def get_noise(self):
        return [torch.normal(0, self.std, shape).to(self.device) for shape in self.shapes]

    @torch.no_grad()
    def push(self, elem):
        for i in range(len(self.shapes)):
            self.noise_sum[i] += elem.value[i] / (2.0 - 1 / 2 ** elem.height)
        self.stack.append(elem)

    @torch.no_grad()
    def pop(self):
        elem = self.stack.pop()
        for i in range(len(self.shapes)):
            self.noise_sum[i] -= elem.value[i] / (2.0 - 1 / 2 ** elem.height)

    @torch.no_grad()
    def __call__(self):
        """
        :return: the noise to be added by DP-FTRL
        """
        self.step += 1

        # add new element to the stack
        self.push(Element(0, self.get_noise()))

        # pop the stack
        while len(self.stack) >= 2 and self.stack[-1].height == self.stack[-2].height:
            # create new element
            left_value, right_value = self.stack[-2].value, self.stack[-1].value
            new_noise = self.get_noise()
            new_elem = Element(
                self.stack[-1].height + 1,
                [x + (y + z) / 2 for x, y, z in zip(new_noise, left_value, right_value)])

            # pop the stack, update sum
            self.pop()
            self.pop()

            # append to the stack, update sum
            self.push(new_elem)
        return self.noise_sum

    @torch.no_grad()
    def proceed_until(self, step_target):
        """
        Proceed until the step_target-th step. This is for the binary tree completion trick.

        :return: the noise to be added by DP-FTRL
        """
        if self.step >= step_target:
            raise ValueError(f'Already reached {step_target}.')
        while self.step < step_target:
            noise_sum = self.__call__()
        return noise_sum


def main(argv):
    # This is a small test. If we set the noise in each node as 1 (by setting
    # test_mode=True), we should be seeing the returned noise as the number of
    # 1s in the binary representations of i when cummu_noises is called i times.

    def countSetBits(n):
        count = 0
        while (n):
            n &= (n - 1)
            count += 1
        return count
    cummu_noises = CummuNoiseTorch(1.0, [(1,)], 'cuda', test_mode=True)
    for epoch in range(31):
        random_noise = cummu_noises()
        assert random_noise[0].cpu().numpy()[0] == countSetBits(epoch + 1)


if __name__ == '__main__':
    app.run(main)