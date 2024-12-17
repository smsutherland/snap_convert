from pathlib import Path
import h5py
import numpy as np
import unyt as u
import subprocess

from .hdf5 import Hdf5Frontend
from .header import Header
from snap_conv.util import git_version


class GadgetFrontend(Hdf5Frontend):
    # NOTE: This does not check to make sure h is what we need it to be
    h = 0.6711

    units = dict(
        mass=1e10 * u.Msun / h,
        length=u.kpc / h,
        velocity=u.km / u.s,
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

    @classmethod
    def _get_output_unit(cls, group, key):
        if group in cls.field_units:
            if key in cls.field_units[group]:
                return cls.field_units[group][key]
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

    @classmethod
    def write(cls, source, fname):
        with h5py.File(fname, "w") as f:
            header = f.create_group("Header").attrs
            header["BoxSize"] = source.header.box_size[0].to(u.kpc)
            header["HubbleParam"] = source.header.h
            header["NumFilesPerSnapshot"] = 1
            header["NumPart_ThisFile"] = source.header.num_part
            header["NumPart_Total"] = source.header.num_part
            header["NumPart_Total_HighWord"] = np.zeros_like(source.header.num_part)
            header["Omega0"] = source.header.Omega_m
            header["OmegaLambda"] = source.header.Omega_Lambda
            header["Redshift"] = source.header.redshift
            header["Time"] = source.header.scale

            header["snap_conv_version"] = git_version

            fields = [
                "ParticleIDs",
                "Coordinates",
                "StarFormationRate",
                "Masses",
                "InternalEnergy",
                "Density",
                "Velocities",
                "SmoothingLength",
            ]
            gas = f.create_group("PartType0")
            for name in fields:
                data = getattr(source.gas, name)
                unit = cls._get_output_unit("PartType0", name)
                if unit is not None:
                    data = data.to(unit)
                gas[name] = data

            fields = [
                "ParticleIDs",
                "Coordinates",
                "Masses",
                "Velocities",
            ]
            dm = f.create_group("PartType1")
            for name in fields:
                data = getattr(source.dark_matter, name)
                unit = cls._get_output_unit("PartType1", name)
                if unit is not None:
                    data = data.to(unit)
                dm[name] = data

            fields = [
                "ParticleIDs",
                "Coordinates",
                "Masses",
                "Velocities",
                "SmoothingLength",
                "InitialMass",
                "StellarFormationTime",
            ]
            stars = f.create_group("PartType4")
            for name in fields:
                data = getattr(source.stars, name)
                unit = cls._get_output_unit("PartType4", name)
                if unit is not None:
                    data = data.to(unit)
                stars[name] = data

            fields = [
                "ParticleIDs",
                "Coordinates",
                "Masses",
                "Velocities",
                "SmoothingLength",
                "Mdot",
            ]
            bhs = f.create_group("PartType5")
            for name in fields:
                data = getattr(source.black_holes, name)
                unit = cls._get_output_unit("PartType5", name)
                if unit is not None:
                    data = data.to(unit)
                bhs[name] = data

    def __str__(self) -> str:
        return "GADGET " + super().__str__()
