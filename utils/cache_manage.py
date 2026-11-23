"""HDF5 caches for the training and analysis data, so the expensive preprocessing runs once.

There are three caches, and they are independent.

* :class:`MuramDataCache` (``.muram_cache*``): one file per MURaM step with the degraded
  Stokes profiles (continuum units, Hinode grid), the atmosphere on the log tau nodes
  (K, km/s, G) and the physics approximations (``blos``, ``vlos``, ``temp``). The arrays
  are not normalized; :meth:`MuramDataCache.load` normalizes them when it builds the
  Dataset. A file is valid for a configuration hash (computed by the caller from every
  setting that changes its content) and a log tau grid.
* :class:`BalancedTrainDataCache` (``.muram_balanced_cache*``): the final balanced,
  normalized training arrays per step, valid for one signature of the whole training
  configuration.
* :class:`ModestDataCache` (``.modest_cache``): the processed MODEST scene, uncropped and
  unmasked, keyed by a hash of its settings.

The two MURaM caches are two layers: the balanced one is built from the per-step one.
Deleting only the balanced cache leaves stale approximations in the per-step cache, so
after changing an approximation both must be cleared (or their hashes must change).

Examples:
    The training loop's use of the per-step cache, in short::

        cache = MuramDataCache(".muram_cache_nicole_tau500")
        if cache.exists(step, config_hash, logtau_values=logtau):
            stokes, mhd, approx = cache.load_raw(step)
        else:
            ...  # process the step, then
            cache.save(step, stokes, mhd, approx, config_hash=config_hash, logtau_values=logtau)
"""

import json
import hashlib
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

from utils.muram_data import MuramStepDataset
from utils.normalizer import MhdNormalizer, StokesNormalizer


class BalancedTrainDataCache:
    """Cache of the balanced, normalized training arrays, one HDF5 file per step.

    All files belong to one signature (a hash of the training configuration). The manifest
    ``balanced_manifest.json`` records the signature and, per step, the file, the number of
    samples and its size. A different signature means the cache must be reset before use.
    """

    def __init__(self, cache_dir: str = ".muram_balanced_cache", compression: str = "lzf"):
        """Open (or create) the cache folder and read its manifest.

        Args:
            cache_dir: Folder of the cache.
            compression: HDF5 compression filter for the arrays (``"lzf"`` is fast).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.manifest_path = self.cache_dir / "balanced_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, object]:
        """Read the manifest, or return an empty one."""
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {
            "version": 1,
            "signature_hash": None,
            "signature": {},
            "steps": {},
            "total_steps": 0,
            "total_selected": 0,
            "total_bytes": 0,
        }

    def _save_manifest(self) -> None:
        """Write the manifest as indented JSON."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    @staticmethod
    def make_signature_hash(signature: dict) -> str:
        """First 12 hex digits of the MD5 of the signature as sorted-key JSON.

        Key order does not matter, but every value does: any change gives a new hash.

        Args:
            signature: JSON-serializable dict describing the training configuration.

        Returns:
            A 12-character hex string.

        Examples:
            >>> h1 = BalancedTrainDataCache.make_signature_hash({"a": 1, "b": [2, 3]})
            >>> h1 == BalancedTrainDataCache.make_signature_hash({"b": [2, 3], "a": 1}), len(h1)
            (True, 12)
            >>> h1 == BalancedTrainDataCache.make_signature_hash({"a": 1, "b": [2, 4]})
            False
        """
        payload = json.dumps(signature, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    def _step_path(self, step: int) -> Path:
        """Path of the file for ``step``, e.g. ``balanced_step_000110.h5``."""
        return self.cache_dir / f"balanced_step_{step:06d}.h5"

    def reset(self, signature: dict, signature_hash: str) -> None:
        """Delete every step file and start a new manifest for ``signature``.

        Args:
            signature: The new configuration signature (stored in the manifest).
            signature_hash: Its hash from :meth:`make_signature_hash`.

        Examples:
            >>> import tempfile
            >>> c = BalancedTrainDataCache(tempfile.mkdtemp())
            >>> c.reset({"lr": 1e-3}, "abc123")
            >>> c.manifest["signature_hash"], c.manifest["total_steps"]
            ('abc123', 0)
        """
        for fp in self.cache_dir.glob("balanced_step_*.h5"):
            fp.unlink()
        self.manifest = {
            "version": 1,
            "signature_hash": signature_hash,
            "signature": signature,
            "steps": {},
            "total_steps": 0,
            "total_selected": 0,
            "total_bytes": 0,
        }
        self._save_manifest()

    def clear(self) -> None:
        """Delete every step file and reset the manifest to an empty one with no signature.

        Examples:
            >>> import tempfile
            >>> c = BalancedTrainDataCache(tempfile.mkdtemp())
            >>> c.clear()
            >>> c.manifest["signature_hash"] is None
            True
        """
        for fp in self.cache_dir.glob("balanced_step_*.h5"):
            fp.unlink()
        self.manifest = {
            "version": 1,
            "signature_hash": None,
            "signature": {},
            "steps": {},
            "total_steps": 0,
            "total_selected": 0,
            "total_bytes": 0,
        }
        self._save_manifest()

    def ensure_signature(self, signature: dict, signature_hash: str) -> bool:
        """Return True if the cache belongs to ``signature_hash``; an empty cache adopts it.

        Args:
            signature: The configuration signature.
            signature_hash: Its hash.

        Returns:
            True if the manifest has no signature yet (it is then set) or the same one;
            False if it holds a different one.

        Examples:
            >>> import tempfile
            >>> c = BalancedTrainDataCache(tempfile.mkdtemp())
            >>> c.ensure_signature({"x": 1}, "h1"), c.ensure_signature({"x": 2}, "h2")
            (True, False)
        """
        current_hash = self.manifest.get("signature_hash")
        if current_hash is None:
            self.manifest["signature_hash"] = signature_hash
            self.manifest["signature"] = signature
            self._save_manifest()
            return True
        return current_hash == signature_hash

    def has_step(self, step: int, signature_hash: str) -> bool:
        """True if ``step`` is cached and valid for ``signature_hash`` and its file exists.

        Args:
            step: Simulation step.
            signature_hash: Expected signature hash.

        Returns:
            Whether the step can be loaded.

        Examples:
            >>> import tempfile
            >>> c = BalancedTrainDataCache(tempfile.mkdtemp())
            >>> c.reset({}, "h"); c.has_step(110, "h")
            False
        """
        if self.manifest.get("signature_hash") != signature_hash:
            return False
        entry = self.manifest.get("steps", {}).get(str(step))
        if not isinstance(entry, dict):
            return False
        path = self._step_path(step)
        if not path.exists():
            return False
        return bool(entry.get("valid", False))

    def save_step(
        self,
        step: int,
        signature_hash: str,
        stokes_input: np.ndarray,
        mhd_targets: np.ndarray,
        spatial_indices: np.ndarray,
        approx_data: dict[str, np.ndarray],
        extra_metadata: dict[str, object] | None = None,
    ) -> Path:
        """Write one step's balanced arrays and update the manifest.

        Args:
            step: Simulation step.
            signature_hash: Must equal the manifest's signature.
            stokes_input: Normalized model input, ``(n, n_channels, n_wl)``.
            mhd_targets: Normalized targets, ``(n, 3 n_tau)``.
            spatial_indices: Pixel ``(x, y)`` per sample, ``(n, 2)``.
            approx_data: The step's ``(nx, ny)`` approximation maps, indexed by the
                samples' ``spatial_indices``; must hold ``"blos"`` (G), ``"vlos"`` (km/s) and
                ``"temp"`` (K). Other keys are ignored.
            extra_metadata: JSON-serializable info stored in the manifest entry.

        Returns:
            Path of the written file.

        Raises:
            ValueError: If the signature does not match (reset first).
            KeyError: If an approximation is missing.

        Examples:
            >>> import tempfile
            >>> c = BalancedTrainDataCache(tempfile.mkdtemp()); c.reset({}, "h")
            >>> approx = {k: np.zeros((2, 2)) for k in ("blos", "vlos", "temp")}
            >>> _ = c.save_step(110, "h", np.zeros((3, 2, 5)), np.zeros((3, 6)), np.zeros((3, 2)), approx)
            >>> c.has_step(110, "h"), c.manifest["total_selected"]
            (True, 3)
        """
        if self.manifest.get("signature_hash") != signature_hash:
            raise ValueError("Balanced cache signature mismatch. Reset cache before saving.")

        path = self._step_path(step)
        with h5py.File(path, "w") as f:
            ds = f.create_group("dataset")
            ds.create_dataset(
                "stokes_input",
                data=np.asarray(stokes_input, dtype=np.float32),
                compression=self.compression,
            )
            ds.create_dataset(
                "mhd_targets",
                data=np.asarray(mhd_targets, dtype=np.float32),
                compression=self.compression,
            )
            ds.create_dataset(
                "spatial_indices",
                data=np.asarray(spatial_indices, dtype=np.int64),
                compression=self.compression,
            )

            ap = f.create_group("approximations")
            for key in ("blos", "vlos", "temp"):
                if key not in approx_data:
                    raise KeyError(f"Missing approximation key '{key}' for balanced cache step {step}")
                ap.create_dataset(
                    key,
                    data=np.asarray(approx_data[key], dtype=np.float32),
                    compression=self.compression,
                )

            f.attrs["step"] = int(step)
            f.attrs["signature_hash"] = str(signature_hash)

        entry = {
            "file": path.name,
            "selected": int(stokes_input.shape[0]),
            "bytes": int(path.stat().st_size),
            "valid": True,
        }
        if extra_metadata:
            entry["metadata"] = extra_metadata

        steps = self.manifest.setdefault("steps", {})
        steps[str(step)] = entry

        self.manifest["total_steps"] = int(len(steps))
        self.manifest["total_selected"] = int(sum(int(v.get("selected", 0)) for v in steps.values()))
        self.manifest["total_bytes"] = int(sum(int(v.get("bytes", 0)) for v in steps.values()))
        self._save_manifest()
        return path

    def load_step(
        self,
        step: int,
        signature_hash: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Read one step's arrays: ``(stokes_input, mhd_targets, spatial_indices, approx_data)``.

        Args:
            step: Simulation step.
            signature_hash: Expected signature hash.

        Returns:
            float32 input and targets, int64 indices, and a dict with the ``blos``, ``vlos``
            and ``temp`` maps (float32).

        Raises:
            FileNotFoundError: If :meth:`has_step` is False.

        Examples:
            >>> x, y, xy, approx = c.load_step(110, "h")  # doctest: +SKIP
        """
        if not self.has_step(step, signature_hash):
            raise FileNotFoundError(f"Balanced cache miss for step {step}")

        path = self._step_path(step)
        with h5py.File(path, "r") as f:
            stokes_input = np.asarray(f["dataset"]["stokes_input"][:], dtype=np.float32)
            mhd_targets = np.asarray(f["dataset"]["mhd_targets"][:], dtype=np.float32)
            spatial_indices = np.asarray(f["dataset"]["spatial_indices"][:], dtype=np.int64)
            approx_data = {
                "blos": np.asarray(f["approximations"]["blos"][:], dtype=np.float32),
                "vlos": np.asarray(f["approximations"]["vlos"][:], dtype=np.float32),
                "temp": np.asarray(f["approximations"]["temp"][:], dtype=np.float32),
            }
        return stokes_input, mhd_targets, spatial_indices, approx_data

    def get_stats(self) -> dict[str, object]:
        """Summary of the manifest: signature, number of steps and samples, size in bytes and MB.

        Returns:
            Dict with ``cache_dir``, ``signature_hash``, ``total_steps``, ``total_selected``,
            ``total_bytes``, ``total_size_mb`` and the per-step ``steps`` entries.

        Examples:
            >>> import tempfile
            >>> BalancedTrainDataCache(tempfile.mkdtemp()).get_stats()["total_steps"]
            0
        """
        return {
            "cache_dir": str(self.cache_dir),
            "signature_hash": self.manifest.get("signature_hash"),
            "total_steps": int(self.manifest.get("total_steps", 0)),
            "total_selected": int(self.manifest.get("total_selected", 0)),
            "total_bytes": int(self.manifest.get("total_bytes", 0)),
            "total_size_mb": float(self.manifest.get("total_bytes", 0)) / (1024**2),
            "steps": self.manifest.get("steps", {}),
        }

    def print_cache_info(self) -> None:
        """Print :meth:`get_stats` as a table.

        Examples:
            >>> BalancedTrainDataCache(".muram_balanced_cache").print_cache_info()  # doctest: +SKIP
        """
        stats = self.get_stats()
        print("\n" + "=" * 80)
        print("BALANCED TRAIN CACHE STATISTICS".center(80))
        print("=" * 80)
        print(f"Cache directory:      {stats['cache_dir']}")
        print(f"Signature hash:       {stats['signature_hash']}")
        print(f"Steps cached:         {stats['total_steps']}")
        print(f"Selected samples:     {stats['total_selected']}")
        print(f"Total size:           {stats['total_size_mb']:.1f} MB")
        print("=" * 80 + "\n")


class MuramDataCache:
    """Per-step cache of the processed MURaM data in HDF5.

    Each file ``step_NNNNNN.h5`` has three groups: ``stokes`` (degraded profiles on the
    Hinode grid in continuum units, plus the ``mean_continuum``, ``circular_polarization``
    and ``hinode_wl`` entries), ``mhd`` (``T`` in K, ``Vz`` in km/s, ``Bz`` in G on the log
    tau nodes) and ``approximations`` (``blos`` in G, ``vlos`` in km/s, ``temp`` in K, one
    value per pixel). Nothing is normalized. ``metadata.json`` records the configuration
    hash, shapes and log tau grid of each step, and the cache-wide log tau grid.
    """
    
    def __init__(self, cache_dir: str = ".muram_cache", compression: str = "gzip"):
        """Open (or create) the cache folder and read its metadata.

        Args:
            cache_dir: Folder of the cache.
            compression: HDF5 compression filter (``"gzip"``, level 4).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        self._reserved_keys = {"__cache_info__"}

    
    def _load_metadata(self) -> dict[str, dict]:
        """Read ``metadata.json``, or return an empty dict."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Write ``metadata.json``."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    @staticmethod
    def _config_hash(config_dict: dict) -> str:
        """First 8 hex digits of the MD5 of ``config_dict`` as sorted-key JSON.

        Examples:
            >>> MuramDataCache._config_hash({"b": 1, "a": 2}) == MuramDataCache._config_hash({"a": 2, "b": 1})
            True
        """
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    @staticmethod
    def make_config_hash(config_dict: dict) -> str:
        """Configuration hash shared by the training and statistics scripts (8 hex digits).

        Args:
            config_dict: JSON-serializable dict of every setting that changes a cached file.

        Returns:
            The hash string.

        Examples:
            >>> len(MuramDataCache.make_config_hash({"wl_range": [15, 60]}))
            8
        """
        return MuramDataCache._config_hash(config_dict)
    
    def _get_cache_path(self, step: int) -> Path:
        """Path of the file for ``step``, e.g. ``step_000110.h5``."""
        return self.cache_dir / f"step_{step:06d}.h5"
    
    def exists(self, step: int, config_hash: str | None = None, logtau_values: np.ndarray | list[float] | None = None) -> bool:
        """True if ``step`` is cached and, when ``config_hash`` is given, cached with that hash.

        Without ``config_hash`` only the file's existence is checked; the training script uses
        that relaxed check for the legacy data source only.

        Args:
            step: Simulation step.
            config_hash: Expected configuration hash, or ``None`` to skip the check.
            logtau_values: Expected log tau grid. If given, the cache's grid must match it.

        Returns:
            Whether the step can be loaded.

        Raises:
            ValueError: If the cache was built on a different log tau grid.

        Examples:
            >>> import tempfile
            >>> c = MuramDataCache(tempfile.mkdtemp())
            >>> c.exists(110, "abcd1234", logtau_values=[-1.0, 0.0])
            False
        """
        self._ensure_logtau_compatible(logtau_values)

        cache_path = self._get_cache_path(step)
        if not cache_path.exists():
            return False
        
        # If config_hash provided, validate
        if config_hash is not None:
            step_key = str(step)
            if step_key not in self.metadata:
                return False
            cached_hash = self.metadata[step_key].get('config_hash')
            if cached_hash != config_hash:
                print(f"  ⚠ Cache config mismatch for step {step}")
                print(f"    Expected: {config_hash}, Found: {cached_hash}")
                return False
        
        return True

    def _ensure_logtau_compatible(self, logtau_values: np.ndarray | list[float] | None) -> None:
        """Raise if the cache (globally or for any step) was built on a different log tau grid.

        The first call with a grid records it as the cache-wide grid.
        """
        expected = self._normalize_logtau_values(logtau_values)
        if expected is None:
            return

        cache_info = self.metadata.get("__cache_info__", {})
        cached_global = cache_info.get("logtau_values")
        if cached_global is not None and cached_global != expected:
            raise ValueError(
                "Cache logtau grid mismatch.\n"
                f"Expected: {expected}\n"
                f"Found (cache metadata): {cached_global}\n"
                f"Delete cache at '{self.cache_dir}' and rerun."
            )

        # Also validate per-step metadata if present
        for step_key in self._step_keys():
            cached_step = self.metadata.get(step_key, {}).get("logtau_values")
            if cached_step is not None and cached_step != expected:
                raise ValueError(
                    "Cache logtau grid mismatch at step "
                    f"{step_key}.\nExpected: {expected}\nFound: {cached_step}\n"
                    f"Delete cache at '{self.cache_dir}' and rerun."
                )

        # If missing global metadata, initialize it
        if cached_global is None:
            self.metadata["__cache_info__"] = {"logtau_values": expected}
            self._save_metadata()

    @staticmethod
    def _normalize_logtau_values(logtau_values: np.ndarray | list[float] | None) -> list[float] | None:
        """Grid as a list of Python floats for comparison and JSON, via float32.

        The values are rounded to 6 decimals but stay float32, so -0.8 becomes
        -0.800000011920929. Comparisons are consistent because both sides go through
        this function.

        Examples:
            >>> MuramDataCache._normalize_logtau_values(np.array([-0.8, 0.0]))
            [-0.800000011920929, 0.0]
        """
        if logtau_values is None:
            return None
        arr = np.asarray(logtau_values, dtype=np.float32)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("logtau_values must be a 1D array with at least 2 values.")
        return [float(x) for x in np.round(arr, 6).tolist()]

    def _step_keys(self) -> list[str]:
        """Metadata keys that are step numbers."""
        return [k for k in self.metadata.keys() if k.isdigit()]

    def save(
        self,
        step: int,
        stokes_data: dict[str, np.ndarray],
        mhd_data: dict[str, np.ndarray],
        approx_data: dict[str, np.ndarray] | None = None,
        config_hash: str | None = None,
        logtau_values: np.ndarray | list[float] | None = None,
        verbose: bool = True,
    ) -> Path:
        """Write one step's Stokes, MHD and approximation arrays and update the metadata.

        A partial file is deleted if writing fails, and the error is raised again.

        Args:
            step: Simulation step.
            stokes_data: Dict of arrays; ``"I"`` must be ``(nx, ny, n_wl)``. Stored as given.
            mhd_data: Dict of arrays; ``"T"`` must be ``(nx, ny, n_tau)``. Stored as given.
            approx_data: Dict of ``(nx, ny)`` approximation maps, or ``None``.
            config_hash: Configuration hash stored with the step (``"unknown"`` if ``None``).
            logtau_values: The log tau grid, stored in the file and the metadata.
            verbose: Print the file size.

        Returns:
            Path of the written file.

        Examples:
            >>> import tempfile, contextlib, io
            >>> c = MuramDataCache(tempfile.mkdtemp())
            >>> stokes = {"I": np.ones((2, 2, 5))}; mhd = {"T": np.full((2, 2, 3), 5000.0)}
            >>> with contextlib.redirect_stdout(io.StringIO()):
            ...     _ = c.save(110, stokes, mhd, {"temp": np.full((2, 2), 6000.0)}, "abcd1234", [-1.0, 0.0, 1.0])
            ...     s, m, a = c.load_raw(110)
            >>> c.exists(110, "abcd1234", [-1.0, 0.0, 1.0]), float(a["temp"][0, 0])
            (True, 6000.0)
        """
        cache_path = self._get_cache_path(step)
        
        try:
            with h5py.File(cache_path, 'w') as f:
                # Store Stokes data
                stokes_group = f.create_group('stokes')
                for key, data in stokes_data.items():
                    stokes_group.create_dataset(
                        key, data=data, compression=self.compression, compression_opts=4
                    )
                
                # Store MHD data
                mhd_group = f.create_group('mhd')
                for key, data in mhd_data.items():
                    mhd_group.create_dataset(
                        key, data=data, compression=self.compression, compression_opts=4
                    )
                
                # Store physics approximations
                approx_group = f.create_group('approximations')
                if approx_data is not None:
                    for key, data in approx_data.items():
                        approx_group.create_dataset(
                            key, data=data, compression=self.compression, compression_opts=4
                        )
                
                # Store metadata
                f.attrs['step'] = step
                if config_hash is not None:
                    f.attrs['config_hash'] = config_hash
                if logtau_values is not None:
                    f.attrs['logtau_values'] = np.asarray(logtau_values, dtype=np.float32)

            # Normalize/serialize logtau for json metadata
            logtau_list = self._normalize_logtau_values(logtau_values)

            # Update metadata
            self.metadata[str(step)] = {
                'config_hash': config_hash or 'unknown',
                'nx': stokes_data['I'].shape[0],
                'ny': stokes_data['I'].shape[1],
                'nwl': stokes_data['I'].shape[2],
                'ntau': mhd_data['T'].shape[2],
                'logtau_values': logtau_list,
                'file_size_mb': cache_path.stat().st_size / (1024**2),
            }
            self._save_metadata()
            
            if verbose:
                file_size_mb = cache_path.stat().st_size / (1024**2)
                print(f"  ✓ Cached step {step} ({file_size_mb:.1f} MB)")
            
            return cache_path
        
        except Exception as e:
            print(f"  ✗ Failed to cache step {step}: {e}")
            # Clean up partial file
            if cache_path.exists():
                cache_path.unlink()
            raise
    
    def load(
        self,
        step: int,
        stokes_normalizer: StokesNormalizer,
        mhd_normalizer: MhdNormalizer,
        verbose: bool = True,
    ) -> tuple[MuramStepDataset, dict[str, np.ndarray]]:
        """Read one step and wrap it in a :class:`MuramStepDataset` (normalizing it).

        Args:
            step: Simulation step.
            stokes_normalizer: Fitted Stokes normalizer.
            mhd_normalizer: Fitted MHD normalizer.
            verbose: Print a message.

        Returns:
            ``(dataset, approx_data)``.

        Raises:
            FileNotFoundError: If the step is not cached.

        Examples:
            >>> dataset, approx = cache.load(110, stokes_norm, mhd_norm)  # doctest: +SKIP
        """
        cache_path = self._get_cache_path(step)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        
        try:
            with h5py.File(cache_path, 'r') as f:
                # Load Stokes data
                stokes_data = {
                    key: f['stokes'][key][:] for key in f['stokes'].keys()
                }
                
                # Load MHD data
                mhd_data = {
                    key: f['mhd'][key][:] for key in f['mhd'].keys()
                }
                
                # Load physics approximations
                approx_data = {
                    key: f['approximations'][key][:] for key in f['approximations'].keys()
                }
            
            # Create dataset
            dataset = MuramStepDataset(
                stokes_data=stokes_data,
                mhd_data=mhd_data,
                stokes_normalizer=stokes_normalizer,
                mhd_normalizer=mhd_normalizer,
            )
            
            if verbose:
                print(f"  ✓ Loaded cached step {step}")
            
            return dataset, approx_data
        
        except Exception as e:
            print(f"  ✗ Failed to load cache for step {step}: {e}")
            raise
    
    def load_raw(
        self,
        step: int,
        verbose: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Read one step's arrays without building a Dataset: ``(stokes, mhd, approx)``.

        Args:
            step: Simulation step.
            verbose: Print a message.

        Returns:
            Three dicts of NumPy arrays (``approx`` is empty if the file has none).

        Raises:
            FileNotFoundError: If the step is not cached.

        Examples:
            >>> stokes, mhd, approx = cache.load_raw(110)  # doctest: +SKIP
        """
        cache_path = self._get_cache_path(step)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        with h5py.File(cache_path, 'r') as f:
            stokes_data = {k: f['stokes'][k][:] for k in f['stokes'].keys()}
            mhd_data = {k: f['mhd'][k][:] for k in f['mhd'].keys()}
            approx_data = {}
            if 'approximations' in f:
                approx_data = {k: f['approximations'][k][:] for k in f['approximations'].keys()}

        if verbose:
            print(f"  ✓ Loaded raw cached step {step}")
        return stokes_data, mhd_data, approx_data
    
    def clear(self, step: int | None = None, confirm: bool = True) -> None:
        """Delete one step's file, or all files, and their metadata.

        Args:
            step: Step to delete, or ``None`` for all.
            confirm: Ask on the terminal before deleting.

        Examples:
            >>> cache.clear(confirm=False)  # doctest: +SKIP
        """
        if step is not None:
            cache_path = self._get_cache_path(step)
            if not cache_path.exists():
                print(f"Cache file not found: {cache_path}")
                return
            
            if confirm:
                response = input(f"Delete cache for step {step}? (y/n): ")
                if response.lower() != 'y':
                    return
            
            cache_path.unlink()
            if str(step) in self.metadata:
                del self.metadata[str(step)]
            self._save_metadata()
            print(f"  ✓ Deleted cache for step {step}")
        
        else:
            if confirm:
                response = input(f"Delete all cache files in {self.cache_dir}? (y/n): ")
                if response.lower() != 'y':
                    return
            
            # Delete all .h5 files
            for cache_file in self.cache_dir.glob("step_*.h5"):
                cache_file.unlink()
            
            # Clear metadata
            self.metadata = {}
            self._save_metadata()
            print(f"  ✓ Cleared all cache files")
    
    def get_cache_stats(self) -> dict[str, any]:
        """Number of cached steps, total size in MB, step list and cache-wide log tau grid.

        Returns:
            Dict with ``total_steps_cached``, ``total_size_mb``, ``cache_dir``, ``steps`` and
            ``cache_logtau_values``.

        Examples:
            >>> import tempfile
            >>> MuramDataCache(tempfile.mkdtemp()).get_cache_stats()["total_steps_cached"]
            0
        """
        cache_files = list(self.cache_dir.glob("step_*.h5"))
        total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024**2)
        
        step_keys = self._step_keys()
        return {
            'total_steps_cached': len(cache_files),
            'total_size_mb': total_size_mb,
            'cache_dir': str(self.cache_dir),
            'steps': step_keys if step_keys else [],
            'cache_logtau_values': self.metadata.get("__cache_info__", {}).get("logtau_values"),
        }
    
    def print_cache_info(self):
        """Print :meth:`get_cache_stats` and a per-step table.

        Examples:
            >>> MuramDataCache(".muram_cache_nicole_tau500").print_cache_info()  # doctest: +SKIP
        """
        stats = self.get_cache_stats()
        
        print("\n" + "=" * 80)
        print("CACHE STATISTICS".center(80))
        print("=" * 80)
        print(f"Cache directory:      {stats['cache_dir']}")
        print(f"Steps cached:         {stats['total_steps_cached']}")
        print(f"Total size:           {stats['total_size_mb']:.1f} MB")
        if stats.get('cache_logtau_values') is not None:
            print(f"Cache logtau nodes:   {stats['cache_logtau_values']}")
        
        if stats['steps']:
            steps_sorted = sorted([int(s) for s in stats['steps']])
            print(f"Step range:           {steps_sorted[0]} - {steps_sorted[-1]}")
            print("\nDetailed cache info:")
            print(f"{'Step':<8} {'nx':>6} {'ny':>6} {'nwl':>6} {'ntau':>6} {'Size (MB)':>12} {'Config':<10}")
            print("-" * 80)
            for step_key in sorted(stats['steps']):
                meta = self.metadata[step_key]
                print(f"{step_key:<8} {meta['nx']:>6} {meta['ny']:>6} {meta['nwl']:>6} "
                      f"{meta['ntau']:>6} {meta['file_size_mb']:>12.1f} {meta['config_hash']:<10}")
        
        print("=" * 80 + "\n")


class ModestDataCache:
    """Cache of the processed MODEST scene (uncropped, unmasked), one HDF5 file per settings hash.

    Note:
        The SPINOR log tau keys are stored as float32, so after a cache hit -0.8 comes back
        as -0.800000011920929. Code that takes the keys from ``tau_values`` works; code that
        indexes with the literal -0.8 (e.g. the default of
        ``ModestData.plot_spinor_atmosphere``) raises KeyError. Wavelength axes also come
        back as plain arrays (Angstrom), without units.
    """

    def __init__(self, cache_dir: str = ".modest_cache"):
        """Open (or create) the cache folder and read its metadata.

        Args:
            cache_dir: Folder of the cache.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict]:
        """Read ``metadata.json``, or return an empty dict."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Write ``metadata.json``."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    @staticmethod
    def _config_hash(config_dict: dict) -> str:
        """First 12 hex digits of the MD5 of ``config_dict`` as sorted-key JSON."""
        payload = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    @staticmethod
    def make_config_hash(config_dict: dict) -> str:
        """Hash of the MODEST processing settings (12 hex digits).

        Args:
            config_dict: JSON-serializable settings (paths, upsampling factor, V multiplier...).

        Returns:
            The hash string.

        Examples:
            >>> len(ModestDataCache.make_config_hash({"upsampling_factor": 2}))
            12
        """
        return ModestDataCache._config_hash(config_dict)

    def _cache_path(self, config_hash: str) -> Path:
        """Path of the file for a hash, e.g. ``modest_<hash>.h5``."""
        return self.cache_dir / f"modest_{config_hash}.h5"

    def exists(self, config_hash: str) -> bool:
        """True if a file for ``config_hash`` exists.

        Args:
            config_hash: Hash from :meth:`make_config_hash`.
        """
        return self._cache_path(config_hash).exists()

    def save(self, config_hash: str, payload: dict, config: dict | None = None, verbose: bool = True) -> Path:
        """Write a processed MODEST payload and record its settings in the metadata.

        Args:
            config_hash: Hash from :meth:`make_config_hash`.
            payload: Dict as built by ``ModestData._snapshot_payload``.
            config: The settings behind the hash, stored for reference.
            verbose: Print the file size.

        Returns:
            Path of the written file.

        Examples:
            >>> import tempfile, contextlib, io
            >>> c = ModestDataCache(tempfile.mkdtemp())
            >>> with contextlib.redirect_stdout(io.StringIO()):
            ...     _ = c.save("h", {"continuum": np.ones((2, 3))})
            ...     back = c.load("h")
            >>> back["continuum"].shape, back["obs_stokes"]
            ((2, 3), None)
        """
        cache_path = self._cache_path(config_hash)
        self._save_hdf5(cache_path, payload)

        self.metadata[config_hash] = {
            'file': cache_path.name,
            'file_size_mb': cache_path.stat().st_size / (1024**2),
            'config': config or {},
            'backend': 'hdf5',
        }
        self._save_metadata()

        if verbose:
            print(f"  ✓ Cached MODEST snapshot ({cache_path.name}, {self.metadata[config_hash]['file_size_mb']:.1f} MB)")

        return cache_path

    def load(self, config_hash: str, verbose: bool = True) -> dict:
        """Read a processed MODEST payload.

        Args:
            config_hash: Hash of the wanted settings.
            verbose: Print a message.

        Returns:
            The payload dict (missing products are ``None``).

        Raises:
            FileNotFoundError: If no file exists for the hash.

        Examples:
            >>> payload = ModestDataCache().load(h)  # doctest: +SKIP
        """
        cache_path = self._cache_path(config_hash)
        if not cache_path.exists():
            raise FileNotFoundError(f"MODEST cache not found: {cache_path}")

        payload = self._load_hdf5(cache_path)

        if verbose:
            print(f"  ✓ Loaded MODEST cache {cache_path.name}")

        return payload

    def clear(self, config_hash: str | None = None, confirm: bool = True) -> None:
        """Delete one cached scene, or all, and their metadata.

        Args:
            config_hash: Hash to delete, or ``None`` for all.
            confirm: Ask on the terminal before deleting.

        Examples:
            >>> ModestDataCache().clear(confirm=False)  # doctest: +SKIP
        """
        if config_hash is not None:
            cache_path = self._cache_path(config_hash)
            if not cache_path.exists():
                print(f"MODEST cache file not found: {cache_path}")
                return

            if confirm:
                response = input(f"Delete MODEST cache {cache_path.name}? (y/n): ")
                if response.lower() != 'y':
                    return

            cache_path.unlink()
            self.metadata.pop(config_hash, None)
            self._save_metadata()
            print(f"  ✓ Deleted MODEST cache {cache_path.name}")
            return

        if confirm:
            response = input(f"Delete all MODEST cache files in {self.cache_dir}? (y/n): ")
            if response.lower() != 'y':
                return

        for cache_file in self.cache_dir.glob("modest_*.h5"):
            cache_file.unlink()
        self.metadata = {}
        self._save_metadata()
        print("  ✓ Cleared all MODEST cache files")

    def get_cache_stats(self) -> dict[str, any]:
        """Number of cached scenes, total size in MB and the stored hashes.

        Returns:
            Dict with ``cache_dir``, ``entries``, ``total_size_mb``, ``hashes`` and ``backend``.

        Examples:
            >>> import tempfile
            >>> ModestDataCache(tempfile.mkdtemp()).get_cache_stats()["entries"]
            0
        """
        cache_files = list(self.cache_dir.glob("modest_*.h5"))
        total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024**2)
        return {
            'cache_dir': str(self.cache_dir),
            'entries': len(cache_files),
            'total_size_mb': total_size_mb,
            'hashes': sorted(self.metadata.keys()),
            'backend': 'hdf5',
        }

    def print_cache_info(self):
        """Print :meth:`get_cache_stats`.

        Examples:
            >>> ModestDataCache().print_cache_info()  # doctest: +SKIP
        """
        stats = self.get_cache_stats()
        print("\n" + "=" * 80)
        print("MODEST CACHE STATISTICS".center(80))
        print("=" * 80)
        print(f"Cache directory:      {stats['cache_dir']}")
        print(f"Backend:              {stats['backend']}")
        print(f"Entries cached:       {stats['entries']}")
        print(f"Total size:           {stats['total_size_mb']:.1f} MB")
        print("=" * 80 + "\n")

    @staticmethod
    def _array_like(value):
        """Plain array from an array or Quantity (units dropped); ``None`` passes through."""
        if value is None:
            return None
        if hasattr(value, 'value'):
            return np.asarray(value.value)
        return np.asarray(value)

    def _save_hdf5(self, cache_path: Path, payload: dict) -> None:
        """Write the payload: Stokes dicts as groups, SPINOR as one group per quantity, maps as datasets."""
        with h5py.File(cache_path, 'w') as f:
            f.attrs['backend'] = 'hdf5'

            def _save_stokes_dict(name: str, data: dict | None):
                """Write one Stokes dict as a group of datasets (skipped if ``None``)."""
                if data is None:
                    return
                grp = f.create_group(name)
                for key, arr in data.items():
                    grp.create_dataset(key, data=np.asarray(arr), compression='lzf')

            arr = self._array_like(payload.get('continuum'))
            if arr is not None:
                f.create_dataset('continuum', data=arr, compression='lzf')

            _save_stokes_dict('obs_stokes', payload.get('obs_stokes'))
            _save_stokes_dict('upsampled_stokes', payload.get('upsampled_stokes'))
            _save_stokes_dict('deconvolved_stokes', payload.get('deconvolved_stokes'))
            _save_stokes_dict('smoothed_stokes', payload.get('smoothed_stokes'))
            _save_stokes_dict('inverted_profs', payload.get('inverted_profs'))

            arr = self._array_like(payload.get('inverted_atmos'))
            if arr is not None:
                f.create_dataset('inverted_atmos', data=arr, compression='lzf')

            spinor = payload.get('spinor_atm')
            if spinor is not None:
                spin_grp = f.create_group('spinor_atm')
                for param, tau_dict in spinor.items():
                    param_grp = spin_grp.create_group(param)
                    tau_values = [float(tau) for tau in sorted(tau_dict.keys())]
                    param_grp.attrs['tau_values'] = np.asarray(tau_values, dtype=np.float32)
                    for idx, tau in enumerate(tau_values):
                        param_grp.create_dataset(f"{idx:03d}", data=np.asarray(tau_dict[tau]), compression='lzf')

            arr = self._array_like(payload.get('wl'))
            if arr is not None:
                f.create_dataset('wl', data=arr)
            arr = self._array_like(payload.get('wl_inv'))
            if arr is not None:
                f.create_dataset('wl_inv', data=arr)

            arr = self._array_like(payload.get('circ_pol'))
            if arr is not None:
                f.create_dataset('circ_pol', data=arr, compression='lzf')

            arr = self._array_like(payload.get('mask'))
            if arr is not None:
                f.create_dataset('mask', data=np.asarray(arr, dtype=np.uint8), compression='lzf')

            sample_pixels = payload.get('sample_pixels')
            if sample_pixels is not None:
                f.attrs['sample_pixels_json'] = json.dumps(sample_pixels)

    def _load_hdf5(self, cache_path: Path) -> dict:
        """Read a payload written by :meth:`_save_hdf5`."""
        payload: dict = {}
        with h5py.File(cache_path, 'r') as f:
            def _load_stokes_dict(name: str):
                """Read one Stokes group, or ``None`` if absent."""
                if name not in f:
                    return None
                grp = f[name]
                return {key: grp[key][:] for key in grp.keys()}

            payload['continuum'] = f['continuum'][:] if 'continuum' in f else None
            payload['obs_stokes'] = _load_stokes_dict('obs_stokes')
            payload['upsampled_stokes'] = _load_stokes_dict('upsampled_stokes')
            payload['deconvolved_stokes'] = _load_stokes_dict('deconvolved_stokes')
            payload['smoothed_stokes'] = _load_stokes_dict('smoothed_stokes')
            payload['inverted_profs'] = _load_stokes_dict('inverted_profs')
            payload['inverted_atmos'] = f['inverted_atmos'][:] if 'inverted_atmos' in f else None

            spinor = None
            if 'spinor_atm' in f:
                spinor = {}
                spin_grp = f['spinor_atm']
                for param in spin_grp.keys():
                    param_grp = spin_grp[param]
                    tau_values = [float(t) for t in param_grp.attrs.get('tau_values', [])]
                    tau_dict = {}
                    for idx, tau in enumerate(tau_values):
                        tau_dict[tau] = param_grp[f"{idx:03d}"][:]
                    spinor[param] = tau_dict
            payload['spinor_atm'] = spinor

            payload['wl'] = f['wl'][:] if 'wl' in f else None
            payload['wl_inv'] = f['wl_inv'][:] if 'wl_inv' in f else None
            payload['circ_pol'] = f['circ_pol'][:] if 'circ_pol' in f else None
            payload['mask'] = (f['mask'][:] > 0) if 'mask' in f else None

            sample_pixels_json = f.attrs.get('sample_pixels_json', None)
            payload['sample_pixels'] = json.loads(sample_pixels_json) if sample_pixels_json is not None else None

        return payload


# Backward-compatible alias; prefer MuramDataCache in new code
DataCache = MuramDataCache
