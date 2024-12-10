import h5py
import unyt as u

from .hdf5 import Hdf5Frontend
from .header import Header

_units = [u.Ampere, u.cm, u.g, u.K, u.s]


class SwiftFrontend(Hdf5Frontend):
    def _make_aliases(self):
        self.gas.alias("Density", "Densities")
        self.gas.alias("Mass", "Masses")
        self.gas.alias("SmoothingLength", "SmoothingLengths")
        self.gas.alias("StarFormationRate", SwiftFrontend.sanitize_sfr)

    def sanitize_sfr(self):
        if (data := self.gas.check_cache("StarFormationRate")) is not None:
            return data
        data = self.gas.StarFormationRates.copy()
        data[data < 0] = 0
        self.gas.add_cache(data, "StarFormationRate")
        return data

    def _get_unit(self, group, key):
        with h5py.File(self.fname) as f:
            attrs = f[group][key].attrs
            factor = attrs[
                "Conversion factor to CGS (not including cosmological corrections)"
            ][0]
            exponents = [attrs[f"U_{c} exponent"][0] for c in "ILMTt"]
            unit = 1.0
            for part, exp in zip(_units, exponents):
                unit = unit * part**exp
            if unit == 1 and factor == 1:
                return None
            return factor * unit

    def load_header(self):
        with h5py.File(self.fname) as f:
            header = f["Header"].attrs
            cosmo = f["Cosmology"].attrs

            redshift = header["Redshift"][0]
            scale = header["Scale-factor"][0]
            h = cosmo["H0 [internal units]"][0] / 100
            H = cosmo["H0 [internal units]"][0] * u.km / u.s / u.Mpc
            box_size = header["BoxSize"] * u.Mpc
            num_part = header["NumPart_Total"]
            Omega_cdm = cosmo["Omega_cdm"]
            Omega_b = cosmo["Omega_b"]
            Omega_m = cosmo["Omega_m"]
            Omega_Lambda = cosmo["Omega_lambda"]

            return Header(
                redshift=redshift,
                scale=scale,
                h=h,
                H=H,
                box_size=box_size,
                num_part=num_part,
                Omega_cdm=Omega_cdm,
                Omega_b=Omega_b,
                Omega_m=Omega_m,
                Omega_Lambda=Omega_Lambda,
            )

    def __str__(self) -> str:
        return "SWIFT " + super().__str__()
