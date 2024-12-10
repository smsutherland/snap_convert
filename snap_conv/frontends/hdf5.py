from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import h5py
import unyt as u

_particle_names = ["gas", "dark_matter", None, None, "stars", "black_holes"]
_particle_class_names = ["Gas", "DarkMatter", None, None, "Stars", "BlackHoles"]


class Hdf5Frontend(ABC):
    def __init__(self, fname, cache_size: Optional[int] = 1024**3):
        self.fname = fname
        self.cache_size = cache_size

        self._get_metadata()
        self.header = self.load_header()
        self.load_num = 0
        self._make_aliases()

    def _get_metadata(self):
        with h5py.File(self.fname) as f:
            keys = f.keys()
            for i in range(6):
                group = f"PartType{i}"
                if group in keys:
                    name = _particle_names[i]
                    value = self._load_particles(
                        self.fname, f[group], group, _particle_class_names[i]
                    )
                    setattr(self, name, value)

    @abstractmethod
    def load_header(self): ...

    def _load_particles(
        self,
        fname: str,
        dataset,
        group: str,
        ptype_name: str,
    ):
        type_dict: Dict[str, Any] = {"_parent": self}
        for k in dataset.keys():
            type_dict[k] = property(_make_getter(fname, group, k))

        def alias(self, destination: str, target):
            self.aliases[destination] = target

        def __getattr__(self, name: str, /) -> Any:
            if name in self.aliases:
                target = self.aliases[name]
                if isinstance(target, str):
                    return getattr(self, target)
                else:
                    return target(self._parent)
            raise AttributeError(name)

        def check_cache(self, key: str):
            if hasattr(self, f"_key_{key}"):
                attr = getattr(self, f"_key_{key}")
                if attr is not None:
                    return attr[0]
                else:
                    return None

        def add_cache(self, data, key):
            setattr(self, f"_key_{key}", (data, self._parent.load_num))
            self._parent.load_num += 1

        type_dict["aliases"] = {}
        type_dict["alias"] = alias
        type_dict["__getattr__"] = __getattr__
        type_dict["check_cache"] = check_cache
        type_dict["add_cache"] = add_cache

        return type(ptype_name + "Dataset", (), type_dict)()

    def make_room(self, incoming):
        if self.cache_size is None:
            return
        loaded = self._get_loaded_fields()
        loaded.sort(key=lambda x: x[2])
        total_size = sum(e[3] for e in loaded)
        bytes_left = total_size - self.cache_size + incoming
        while bytes_left > 0:
            ptype, key, _, nbytes = loaded[0]
            setattr(getattr(self, ptype), key, None)
            bytes_left -= nbytes

    def get_loaded_size(self):
        loaded = self._get_loaded_fields()
        total_size = sum(e[3] for e in loaded)
        return total_size

    def _get_loaded_fields(self):
        loaded = []
        for ptype in _particle_names:
            if ptype is None:
                continue

            if not hasattr(self, ptype):
                continue
            particle_dataset = getattr(self, ptype)
            for key in dir(particle_dataset):
                if not key.startswith("_key_"):
                    continue
                field = getattr(particle_dataset, key)
                if field is not None:
                    loaded.append((ptype, key, field[1], field[0].nbytes))
        return loaded

    def _make_aliases(self):
        pass

    @abstractmethod
    def _get_unit(self, group, key): ...

    def __str__(self) -> str:
        return f"Dataset at {self.fname}"

    def __repr__(self) -> str:
        return str(self)


def _make_getter(fname: str, group: str, key: str):
    def getter(self):
        if (data := self.check_cache(key)) is not None:
            return data

        with h5py.File(fname) as f:
            loaded_group = f[group]
            assert isinstance(loaded_group, h5py.Group)
            loaded_data = loaded_group[key]
            assert isinstance(loaded_data, h5py.Dataset)
            nbytes = loaded_data.nbytes
            self._parent.make_room(nbytes)
            unit = self._parent._get_unit(group, key)
            if unit is not None:
                loaded_data = u.unyt_array(loaded_data[:], unit)
            else:
                loaded_data = loaded_data[:]
            self.add_cache(loaded_data, key)
            return loaded_data

    return getter
