# tests/test_imports.py

def test_import_cosmology():
    try:
        import cctoolkit.cosmology
    except ImportError as e:
        assert False, f"Importing 'cctoolkit.cosmology' failed: {e}"

def test_import_hmf():
    try:
        import cctoolkit.hmf
    except ImportError as e:
        assert False, f"Importing 'cctoolkit.hmf' failed: {e}"

def test_import_bias():
    try:
        import cctoolkit.bias
    except ImportError as e:
        assert False, f"Importing 'cctoolkit.bias' failed: {e}"

def test_import_utils():
    try:
        import cctoolkit.utils
    except ImportError as e:
        assert False, f"Importing 'cctoolkit.utils' failed: {e}"

def test_import_baryons():
    try:
        import cctoolkit.baryons
    except ImportError as e:
        assert False, f"Importing 'cctoolkit.baryons' failed: {e}"


