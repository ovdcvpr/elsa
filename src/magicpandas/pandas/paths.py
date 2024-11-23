from magicpandas.pandas.column import Column
from pathlib import Path

# todo: create warning if anything overrides Series methods
# todo: allow seamless interaction with pathlib.Path e.g. Path / Paths -> Paths

"""
This also showcases why magicpandas' use of dunder, with the non-dunder
working as a proxy, is necessary. pathlib.Path uses the names:
    owner
    root
    name
so we cannot mimic pathlib.Path without breaking magicpandas. However 
something like __owner__ or __root__ would be inconvenient to the user
and intimidating to a beginner, so we use both. 
"""

class Paths(Column):
    """
    mimic pathlib.Path with a magic Column
    using a Series with dtype=pyarrow
    """
    __dtype__ = 'pyarrow[string]'

    def __rdiv__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __add__(self, other):
        raise NotImplementedError

    def __fspath__(self):
        raise NotImplementedError

    def absolute(self):
        raise NotImplementedError

    def as_posix(self):
        raise NotImplementedError

    def as_uri(self):
        raise NotImplementedError

    def chmod(self, mode):
        raise NotImplementedError

    def cwd(self):
        raise NotImplementedError

    def exists(self):
        raise NotImplementedError

    def expanduser(self):
        raise NotImplementedError

    def glob(self, pattern):
        raise NotImplementedError

    def group(self):
        raise NotImplementedError

    def home(self):
        raise NotImplementedError

    def is_absolute(self):
        raise NotImplementedError

    def is_dir(self):
        raise NotImplementedError

    def is_file(self):
        raise NotImplementedError

    def is_mount(self):
        raise NotImplementedError

    def is_symlink(self):
        raise NotImplementedError

    def is_fifo(self):
        raise NotImplementedError

    def is_block_device(self):
        raise NotImplementedError

    def is_char_device(self):
        raise NotImplementedError

    def is_socket(self):
        raise NotImplementedError

    def iterdir(self):
        raise NotImplementedError

    def joinpath(self, *other):
        raise NotImplementedError

    def lchmod(self, mode):
        raise NotImplementedError

    def lstat(self):
        raise NotImplementedError

    def match(self, pattern):
        raise NotImplementedError

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        raise NotImplementedError

    def open(self, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
        raise NotImplementedError

    def owner(self):
        raise NotImplementedError

    # todo: these conflict with pandas: do we prioritize pathlib or pandas?
    # def rename(self, target):
    #     raise NotImplementedError
    #
    # def replace(self, target):
    #     raise NotImplementedError

    def resolve(self, strict=False):
        raise NotImplementedError

    def rglob(self, pattern):
        raise NotImplementedError

    def relative_to(self, *other):
        raise NotImplementedError

    def rmdir(self):
        raise NotImplementedError

    def samefile(self, other_path):
        raise NotImplementedError

    def stat(self):
        raise NotImplementedError

    def symlink_to(self, target, target_is_directory=False):
        raise NotImplementedError

    def touch(self, mode=0o666, exist_ok=True):
        raise NotImplementedError

    def unlink(self, missing_ok=False):
        raise NotImplementedError

    def with_name(self, name):
        raise NotImplementedError

    def with_suffix(self, suffix):
        raise NotImplementedError

    def read_bytes(self):
        raise NotImplementedError

    def read_text(self, encoding=None, errors=None):
        raise NotImplementedError

    def write_bytes(self, data):
        raise NotImplementedError

    def write_text(self, data, encoding=None, errors=None):
        raise NotImplementedError

    @property
    def anchor(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def parent(self):
        raise NotImplementedError

    @property
    def parents(self):
        raise NotImplementedError

    @property
    def stem(self):
        raise NotImplementedError

    @property
    def suffix(self):
        raise NotImplementedError

    @property
    def suffixes(self):
        raise NotImplementedError

    @property
    def drive(self):
        raise NotImplementedError

    @property
    def root(self):
        raise NotImplementedError

    @property
    def parts(self):
        raise NotImplementedError

    @property
    def is_reserved(self):
        raise NotImplementedError


class paths(Paths):
    ...

locals()['paths'] = Paths


