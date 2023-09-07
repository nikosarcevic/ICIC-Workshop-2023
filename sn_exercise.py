import numpy as np
import yaml


class SNExercise:
    """A class to compute various cosmological quantities."""

    def __init__(self, redshift_range):
        """
        Initialize the SNExercise instance.

        Parameters:
        - redshift_range (float): The redshift range to be used for calculations.
        """
        # Load parameters from the YAML file
        with open("data_input/sn_parameters.yaml", 'r') as file:
            parameters = yaml.safe_load(file)

        # Initialize instance attributes
        self.redshift_range = redshift_range
        self.h = parameters["h"]
        self.omega_m = parameters["omega_m"]

    def calculate_parameter_s(self, omega_m=None):
        """
        Calculate the 's' parameter.

        Parameters:
        - omega_m (float, optional): Matter density parameter. Defaults to instance's omega_m.

        Returns:
        - float: Computed 's' value.
        """
        if not omega_m:
            omega_m = self.omega_m

        if omega_m == 0:
            raise ValueError("omega_m cannot be zero.")

        return ((1 - omega_m) / omega_m) ** (1 / 3)

    def calculate_scale_factor(self, redshift_range=None):
        """
        Calculate the scale factor.

        Parameters:
        - redshift_range (float, optional): Redshift value. Defaults to instance's redshift_range.

        Returns:
        - float: Computed scale factor.
        """
        if redshift_range is None:
            redshift_range = self.redshift_range

        scale_factor = 1 / (1 + redshift_range)

        return scale_factor

    def conformal_time(self, redshift_range=None, omega_m=None):
        """
        Calculate the eta parameter.

        Parameters:
        - redshift_range (float, optional): Redshift value. Defaults to instance's redshift_range.
        - omega_m (float, optional): Matter density parameter. Defaults to instance's omega_m.

        Returns:
        - float: Computed eta value.
        """
        if redshift_range is None:
            redshift_range = self.redshift_range
        if omega_m is None:
            omega_m = self.omega_m

        s = self.calculate_parameter_s(omega_m)
        a = self.calculate_scale_factor(redshift_range)

        # Intermediate calculations for eta
        multiplier = 2 * np.sqrt(s ** 3 + 1)
        multiplicand = (
                1 / a ** 4 - 0.1540 * s / a ** 3 + 0.4304 * s ** 2 / a ** 2 +
                0.19097 * s ** 3 / a + 0.066941 * s ** 4
        )

        return multiplier * (multiplicand ** (-1 / 8))

    def luminosity_distance(self, redshift_range=None, omega_m=None):
        """
        Calculate the luminosity distance.

        Parameters:
        - redshift_range (float, optional): Redshift value. Defaults to instance's redshift_range.
        - omega_m (float, optional): Matter density parameter. Defaults to instance's omega_m.

        Returns:
        - float: Computed luminosity distance.
        """
        if redshift_range is None:
            redshift_range = self.redshift_range
        if omega_m is None:
            omega_m = self.omega_m

        eta_a1 = self.conformal_time(redshift_range=0, omega_m=omega_m)
        eta_z = self.conformal_time(redshift_range, omega_m=omega_m)

        # Luminosity distance calculation
        d_l = 3_000 * (redshift_range + 1) * (eta_a1 - eta_z)
        return d_l

    def distance_modulus(self, redshift_range=None, omega_m=None, h=None):
        """
        Calculate the distance modulus.

        Parameters:
        - redshift_range (float, optional): Redshift value. Defaults to instance's redshift_range.
        - omega_m (float, optional): Matter density parameter. Defaults to instance's omega_m.

        Returns:
        - float: Computed distance modulus.
        """
        if redshift_range is None:
            redshift_range = self.redshift_range
        if omega_m is None:
            omega_m = self.omega_m
        if h is None:
            h = self.h

        luminosity_distance = self.luminosity_distance(redshift_range, omega_m)

        if (luminosity_distance <= 0).any():
            raise ValueError("All luminosity distances must be positive for valid log computation.")

        mu = 25. - 5. * np.log10(h) + 5. * np.log10(luminosity_distance)

        return mu
