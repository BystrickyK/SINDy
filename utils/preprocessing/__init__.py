import numpy as np


class EnergyNormalizer:
    def __init__(self, data):
        self.energy = None       # energy of each signal
        self.mean_energy = None  # mean energy across all signals

        self.data = data
        self.normalized = False  # data state, goes to True on normalization, or to False on "de-normalization"
        self.params = None       # normalization parameters (scales)

    def calculate_energies(self):

        def signal_energy(signal):
            signal = np.array(signal)
            energy = np.sum(np.square(signal))
            return energy

        if self.energy is None:
            self.energy = np.apply_along_axis(func1d=signal_energy, axis=0, arr=self.data)
            self.mean_energy = np.mean(self.energy)
            return self.energy
        else:
            print("Energies have already been calculated. They'll be re-calculated on return, but not saved as an attribute.")
            energy = np.apply_along_axis(func1d=signal_energy, axis=0, arr=self.data)
            return energy

    def normalize(self):
        """
        Assuming rows are samples and columns are signals. Rescales the signals so their energy is 1.
        """
        if not self.normalized:
            self.params = 1 / self.energy
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




