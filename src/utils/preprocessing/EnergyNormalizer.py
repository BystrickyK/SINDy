import numpy as np
from src.utils.containers import DynaFrame

class EnergyNormalizer:
    def __init__(self, data):
        self.energy = None       # energy of each signal

        self.data = data
        self.normalized = False  # data state, goes to True on normalization, or to False on "de-normalization"
        self.params = None       # normalization parameters (scales)

    def calculate_energies_(self):
        ""
        Calculates the energy of each signal in self.data
        Number of elements is equal to number of columns in self.data
        """

        def signal_energy(signal):
            signal = np.array(signal)
            energy = np.sum(np.square(signal))
            return energy

        energy = np.apply_along_axis(func1d=signal_energy, axis=0, arr=self.data)
        return energy

    def set_energies(self):
        """
        Sets the energy attribute self.energy that's used for normalization
        """
        self.energy = self.calculate_energies_()

    def normalize(self):
        """
        Assuming rows are samples and columns are signals
        Rescales the signals so that each signal has unit energy
        """
        if self.energy is None:
            self.set_energies()

        if not self.normalized:
            self.params = 1 / np.sqrt(self.energy)
            self.data = self.data * self.params
            self.normalized = True
        else:
            raise Warning('The data has already been normalized')

    def denormalize(self):
        if self.normalized:
            self.data = self.data / self.params
            self.normalized = False
        else:
            raise Warning('The data hasn\'t been normalized')
