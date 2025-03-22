
from dhnamlib.pylib.object import ObjectCache


# Call functions to set objects before the `configuration` module is loaded.

NO_DOMAIN_NAME = object()

_default_domain_name_cache = ObjectCache()

set_default_domain_name = _default_domain_name_cache.set_object

def get_default_domain_name():
    if not _default_domain_name_cache.is_cached():
        set_default_domain_name(NO_DOMAIN_NAME)

    return _default_domain_name_cache.get_object()
