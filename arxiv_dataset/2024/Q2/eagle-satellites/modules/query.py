from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import subprocess



class DatabaseQuery(object):
    """Class to interpret `query_database.sh`"""

    def __init__(self, filename='query_database.sh'):
        self.filename = filename
        with open(filename) as f:
            self._content = f.read().splitlines()


    @property
    def aliases(self):
        variables = {}
        for i, line in enumerate(self._content):
            if line.startswith('## Aliases'):
                break
        for j, line in enumerate(self._content[i:]):
            if line.startswith('## End aliases'):
                break
            if line.startswith('#'):
                continue
            var = line.split('=')
            variables[var[0]] = var[1]
        return variables


    @property
    def user(self):
        for line in self._content:
            if line.startswith('user='):
                return line.split('=')[1]


    @property
    def passwd(self):
        for line in self._content:
            if line.startswith('passwd='):
                return line.split('=')[1]


    def wget(self, query, output=None):
        cmd = ['wget', '--http-user={0}'.format(self.user),
               '--http-passwd={0}'.format(self.passwd), query]
        if output is not None:
            cmd.append('-O')
            cmd.append(output)
        subprocess.run(cmd)


