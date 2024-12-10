import h5py
import numpy as np
import unyt as u

from .hdf5 import Hdf5Frontend
from .header import Header


class GadgetFrontend(Hdf5Frontend):
    # NOTE: This does not check to make sure h is what we need it to be
    h = 0.6711

    units = dict(
        mass=1e10 * u.Msun / h,
        length=u.kpc,
        velocity=1e3 * u.km / u.s,
    )
    units["time"] = units["length"] / units["velocity"]

    gas_units = dict(
        Coordinates=units["length"],
        StarFormationRate=u.Msun / u.yr,
        Masses=units["mass"],
        InternalEnergy=units["velocity"] ** 2,
        Density=units["mass"] / units["length"] ** 3,
        Velocities=units["velocity"],
        SmoothingLength=units["length"],
    )
    dm_units = dict(
        Coordinates=units["length"],
        Masses=units["mass"],
        Velocities=units["velocity"],
    )
    star_units = dict(
        Coordinates=units["length"],
        Masses=units["mass"],
        Velocities=units["velocity"],
        SmoothingLength=units["length"],
        InitialMass=units["mass"],
    )
    bh_units = dict(
        Coordinates=units["length"],
        Masses=units["mass"],
        Velocities=units["velocity"],
        SmoothingLength=units["length"],
        Mdot=units["mass"] / units["time"],
    )
    field_units = dict(
        PartType0=gas_units,
        PartType1=dm_units,
        PartType4=star_units,
        PartType5=bh_units,
    )

    def _get_unit(self, group, key):
        if group in self.field_units:
            if key in self.field_units[group]:
                return self.field_units[group][key]
        return None

    def load_header(self):
        with h5py.File(self.fname) as f:
            header = f["Header"].attrs

            redshift = header["Redshift"]
            scale = header["Time"]
            h = header["HubbleParam"]
            H = h * 100 * u.km / u.s / u.Mpc
            box_size = np.ones(3) * header["BoxSize"] * u.Mpc / h
            num_part = header["NumPart_Total"]
            Omega_b = 0.049  # TODO: Don't hard code this
            Omega_m = header["Omega0"]
            Omega_cdm = Omega_m - Omega_b
            Omega_Lambda = header["OmegaLambda"]

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
        return "GADGET " + super().__str__()
