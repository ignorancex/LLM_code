import os

pwd = os.path.abspath(__file__)[:-9]
for f in os.listdir(pwd):
    if f[0] == '.':
        continue
    path = os.path.join(pwd, f)
    if os.path.isdir(path):
        path = os.path.join(path, 'setup.py')
        if os.path.isfile(path):
            r = os.system('python %s' % path)
            if r != 0:
                raise RuntimeError('Some subpackages fail to be built!')
print('\033[42mAll subpackages built successfully!\033[0m')
