class Units:
    """
    class that contains the units and physical constant for the SQcircuit
    """

    def __init__(self):
        # Henry units
        self.henryList = {'H': 1.0, 'mH': 1.0e-3, 'uH': 1.0e-6, 'nH': 1.0e-9, 'pH': 1.0e-12, 'fH': 1.0e-15}

        # Farad units
        self.faradList = {'F': 1, 'mF': 1.0e-3, 'uF': 1.0e-6, 'nF': 1.0e-9, 'pF': 1.0e-12, 'fF': 1.0e-15}

        # frequency unit list
        self.freqList = {'Hz': 1.0, 'kHz': 1.0e3, 'MHz': 1.0e6, 'GHz': 1.0e9, 'THz': 1.0e12}

        # reduced Planck constant
        self.hbar = 1.0545718e-34

        # magnetic flux quantum
        self.Phi0 = 2.067833e-15

        # electron charge
        self.e = 1.6021766e-19

        # main frequency unit of the SQcircuit
        self.freq = self.freqList["GHz"]

    def setFreq(self, fUnit):
        """
        changes the main frequency unit of the SQcircuit
        inputs:
            -- fUnit: the desired frequency unit
        """
        assert fUnit in self.freqList, "The input format is not correct."

        self.freq = self.freqList[fUnit]


# The global units of the SQcircuit.
unit = Units()
