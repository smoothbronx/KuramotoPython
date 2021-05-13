from numpy.random import random, normal
from numpy import sin, meshgrid, linspace, zeros_like
from numpy import pi as PI, e as E, sum as numpy_sum
from scipy.integrate import odeint


class Kuramoto:
    def __init__(self, coupling: float = 1, dt: float = 0.01, total_time: float = 10, nodes_count=None,
                 vibration_array=None):
        if nodes_count is None and vibration_array is None:
            raise ValueError("nodes_count or vibration_array must be specified")

        self.dt = dt
        self.total_time = total_time
        self.coupling = coupling

        if vibration_array is not None:
            self.natfreqs = vibration_array
            self.nodes_count = len(vibration_array)
        else:
            self.nodes_count = nodes_count
            self.natfreqs = normal(size=self.nodes_count)

    def derivative(self, angles_vector, time, connectivity_matrix):
        assert len(angles_vector) == len(self.natfreqs) == len(
            connectivity_matrix), 'Input dimensions do not match, check lengths'
        angles_i, angles_j = meshgrid(angles_vector, angles_vector)
        dxdt = self.natfreqs + self.coupling / connectivity_matrix.sum(axis=0) * (
                    connectivity_matrix * sin(angles_j - angles_i)).sum(axis=0)
        return dxdt

    def integrate(self, angles_vector, connectivity_matrix):
        time = linspace(0, self.total_time, int(self.total_time / self.dt))
        time_series = odeint(self.derivative, angles_vector, time, args=(connectivity_matrix,))
        return time_series.T

    def run(self, connectivity_matrix=None, angles_vector=None):
        assert (connectivity_matrix == connectivity_matrix.T).all(), 'connectivity_matrix must be symmetric'
        if angles_vector is None:
            angles_vector = self.init_angles()
        return self.integrate(angles_vector, connectivity_matrix)

    def mean_frequency(self, activity_matrix, connectivity_matrix):
        assert len(connectivity_matrix) == activity_matrix.shape[0], 'connectivity_matrix does not match act_mat'
        _, steps_count = activity_matrix.shape
        dxdt = zeros_like(activity_matrix)
        for time in range(steps_count):
            dxdt[:, time] = self.derivative(activity_matrix[:, time], time, connectivity_matrix)
        return numpy_sum(dxdt * self.dt, axis=1) / self.total_time

    @staticmethod
    def phase_coherence(angles_vector):
        total = sum([(E ** (1j * i)) for i in angles_vector])
        return abs(total / len(angles_vector))

    def init_angles(self):
        return 2 * PI * random(size=self.nodes_count)
