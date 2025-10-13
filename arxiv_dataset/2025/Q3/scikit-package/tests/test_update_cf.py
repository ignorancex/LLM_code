import importlib
from pathlib import Path

from scikit_package.cli.update.cf import _update_meta_yaml


def test_update_meta_yaml_realistic(tmpdir):
    original_meta = """{%- set version = "1.0.5" -%}
package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://pypi.org/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz  # noqa: E501
  sha256: 1b71d398d73800db32b09785af0e7
"""
    new_version = "1.0.6"
    new_sha256 = "123456789abcdef0123456789"
    expected_updated_meta = """{%- set version = "1.0.6" -%}
package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  url: https://pypi.org/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz  # noqa: E501
  sha256: 123456789abcdef0123456789
"""
    meta_file = Path(tmpdir) / "meta.yaml"
    meta_file.write_text(original_meta)
    _update_meta_yaml(str(meta_file), new_version, new_sha256)
    updated_meta = meta_file.read_text()
    assert updated_meta == expected_updated_meta


# C1: a src file in a namespace package. Expect the it's correct relative path
# in the submodule is returned.
def test_get_src_file_location_in_submodule():
    spec = importlib.util.spec_from_file_location(
        "post_gen_project",
        Path(__file__).parents[1] / "hooks" / "post_gen_project.py",
    )
    post_gen_project = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(post_gen_project)

    filename = "src/diffpy.my_project/utils/helper.py"
    expected_filename = "src/diffpy/my_project/utils/helper.py"
    actual_filename = post_gen_project.get_src_file_location_in_submodule(
        filename, "diffpy.my_project"
    )
    assert actual_filename == expected_filename
