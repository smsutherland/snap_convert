import h5py
import numpy as np
import unyt as u

from snap_conv.util import git_version

from .hdf5 import Hdf5Frontend
from .header import Header

_units = [u.Ampere, u.cm, u.g, u.K, u.s]


class SwiftFrontend(Hdf5Frontend):
    def _make_aliases(self):
        self.gas.alias("Density", "Densities")
        self.gas.alias("SmoothingLength", "SmoothingLengths")
        self.gas.alias("StarFormationRate", SwiftFrontend.sanitize_sfr)
        self.gas.alias("InternalEnergy", "InternalEnergies")
        self.stars.alias("SmoothingLength", "SmoothingLengths")
        self.stars.alias("InitialMass", "InitialMasses")
        self.stars.alias("StellarFormationTime", "BirthScaleFactors")
        self.black_holes.alias("Masses", "SubgridMasses")
        self.black_holes.alias("SmoothingLength", "SmoothingLengths")
        self.black_holes.alias("Mdot", "AccretionRates")

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

    @classmethod
    def _get_output_unit(cls, group, key):
        units = dict(
            mass=1e10 * u.Msun,
            length=u.Mpc,
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

        if group in field_units:
            if key in field_units[group]:
                return field_units[group][key]
        return None

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

    @classmethod
    def write(cls, source, fname):
        with h5py.File(fname, "w") as f:
            header = f.create_group("Header").attrs
            header["BoxSize"] = source.header.box_size.to(u.Mpc)
            header["NumFilesPerSnapshot"] = 1
            header["NumPart_ThisFile"] = source.header.num_part
            header["NumPart_Total"] = source.header.num_part
            header["NumPart_Total_HighWord"] = np.zeros_like(source.header.num_part)
            header["Time"] = source.header.scale
            header["Redshift"] = [source.header.redshift]
            header["Scale-factor"] = [source.header.scale]
            header["snap_conv_version"] = git_version

            cosmo = f.create_group("Cosmology").attrs
            cosmo["H0 [internal units]"] = [source.header.H.to(u.km / u.s / u.Mpc)]
            cosmo["Omega_b"] = [source.header.Omega_b]
            cosmo["Omega_cdm"] = [source.header.Omega_cdm]
            cosmo["Omega_lambda"] = [source.header.Omega_Lambda]
            cosmo["Omega_m"] = [source.header.Omega_m]
            cosmo["Redshift"] = [source.header.redshift]
            cosmo["Scale-factor"] = [source.header.scale]
            cosmo["h"] = [source.header.h]

            fields = [
                ("ParticleIDs",),
                ("Coordinates",),
                ("StarFormationRates", "StarFormationRate"),
                ("Masses",),
                ("InternalEnergies", "InternalEnergy"),
                ("Densities", "Density"),
                ("Velocities",),
                ("SmoothingLengths", "SmoothingLength"),
            ]
            gas = f.create_group("PartType0")
            for names in fields:
                name = names[0]
                for n in names:
                    if hasattr(source.gas, n):
                        data = getattr(source.gas, name)
                        unit = cls._get_output_unit("PartType5", name)
                        if unit is not None:
                            data = data.to(unit)
                        gas[name] = data
                        if unit is not None:
                            gas[name].attrs[
                                "Conversion factor to CGS (not including cosmological corrections)"
                            ] = [
                                unit.get_conversion_factor(unit.get_cgs_equivalent())[0]
                            ]
                        else:
                            gas[name].attrs[
                                "Conversion factor to CGS (not including cosmological corrections)"
                            ] = [1.0]
                        break
            fields = [
                ("ParticleIDs",),
                ("Coordinates",),
                ("Masses",),
                ("Velocities",),
                ("SmoothingLengths", "SmoothingLength"),
                ("InitialMasses", "InitialMass"),
                ("BirthScaleFactors", "StellarFormationTime"),
            ]
            stars = f.create_group("PartType4")
            for names in fields:
                name = names[0]
                for n in names:
                    if hasattr(source.gas, n):
                        data = getattr(source.gas, name)
                        unit = cls._get_output_unit("PartType5", name)
                        if unit is not None:
                            data = data.to(unit)
                        stars[name] = data
                        if unit is not None:
                            stars[name].attrs[
                                "Conversion factor to CGS (not including cosmological corrections)"
                            ] = [
                                unit.get_conversion_factor(unit.get_cgs_equivalent())[0]
                            ]
                        else:
                            stars[name].attrs[
                                "Conversion factor to CGS (not including cosmological corrections)"
                            ] = [1.0]
                        break
            fields = [
                ("ParticleIDs",),
                ("Coordinates",),
                ("Masses",),
                ("Velocities",),
            ]
            dm = f.create_group("PartType1")
            for names in fields:
                name = names[0]
                for n in names:
                    if hasattr(source.gas, n):
                        data = getattr(source.gas, name)
                        unit = cls._get_output_unit("PartType5", name)
                        if unit is not None:
                            data = data.to(unit)
                        dm[name] = data
                        if unit is not None:
                            dm[name].attrs[
                                "Conversion factor to CGS (not including cosmological corrections)"
                            ] = [
                                unit.get_conversion_factor(unit.get_cgs_equivalent())[0]
                            ]
                        else:
                            dm[name].attrs[
                                "Conversion factor to CGS (not including cosmological corrections)"
                            ] = [1.0]
                        break
            fields = [
                ("ParticleIDs",),
                ("Coordinates",),
                ("Masses",),
                ("Velocities",),
                ("SmoothingLengths", "SmoothingLength"),
                ("InitialMasses", "InitialMass"),
                ("BirthScaleFactors", "StellarFormationTime"),
            ]
            stars = f.create_group("PartType4")
            for names in fields:
                name = names[0]
                for n in names:
                    if hasattr(source.gas, n):
                        data = getattr(source.gas, name)
                        unit = cls._get_output_unit("PartType5", name)
                        if unit is not None:
                            data = data.to(unit)
                        stars[name] = data
                        if unit is not None:
                            stars[name].attrs[
                                "Conversion factor to CGS (not including cosmological corrections)"
                            ] = [
                                unit.get_conversion_factor(unit.get_cgs_equivalent())[0]
                            ]
                        else:
                            stars[name].attrs[
                                "Conversion factor to CGS (not including cosmological corrections)"
                            ] = [1.0]
                        break

            fields = [
                ("ParticleIDs",),
                ("Coordinates",),
                ("Masses",),
                ("Velocities",),
                ("SmoothingLengths", "SmoothingLength"),
                ("AccretionRates", "Mdot"),
            ]
            bhs = f.create_group("PartType5")
            for names in fields:
                name = names[0]
                for n in names:
                    if hasattr(source.gas, n):
                        data = getattr(source.gas, name)
                        unit = cls._get_output_unit("PartType5", name)
                        if unit is not None:
                            data = data.to(unit)
                        bhs[name] = data
                        if unit is not None:
                            bhs[name].attrs[
                                "Conversion factor to CGS (not including cosmological corrections)"
                            ] = [
                                unit.get_conversion_factor(unit.get_cgs_equivalent())[0]
                            ]
                        else:
                            bhs[name].attrs[
                                "Conversion factor to CGS (not including cosmological corrections)"
                            ] = [1.0]
                        break
