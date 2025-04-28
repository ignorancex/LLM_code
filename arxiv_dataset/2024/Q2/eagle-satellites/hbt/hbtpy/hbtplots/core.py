class PlotterHelper:

    def __init__(self):
        return

    def axlabel(self, col, statistic='mean', with_units=True):
        """Cannot handle ratios of different columns yet"""
        if '/' in col:
            raise NotImplementedError(
                f'axlabel cannot yet handle ratios of columns ({col})')
        name = None
        if statistic == 'mean':
            name = self.label(col, with_units=with_units)
        elif statistic == 'std':
            name = self.label(col, with_units=False).replace('$', '')
            name = fr'\sigma({name})'
            if with_units:
                name = fr'{name}\,{self.units(col)}'
        elif statistic == 'std/mean':
            num = self.axlabel(col, statistic='std', with_units=False)
            den = self.axlabel(col, statistic='mean', with_units=False)
            name = fr'{num} / {den}'.replace('$', '')
        if name:
            return fr'${name}$'
        return f'{statistic}({col})'

    def bins(self, col, n=8):
        bins = {
            'ComovingMostBoundDistance': np.logspace(-2, 0.5, 9),
            'M200Mean': np.logspace(13, 14.7, 9),
            'mu': np.logspace(-5, 0, 9),
            'Mstar': np.logspace(7.7, 12, 11),
            }
        if col in bins:
            return bins[col]
        return n

    def label(self, col, with_units=True):
        label = {
            'ComovingMostBoundDistance': 'R',
            'M200Mean': r'M_\mathrm{200m}',
            'mu': r'\mu',
            'Mbound': r'm_\mathrm{sub}',
            'Mstar': r'm_{\!\star}',
            'Mdm': r'm_\mathrm{DM}',
            'Mgas': r'm_\mathrm{gas}',
            'LastMaxMass': r'm_\mathrm{sub,max}'
            }
        name = None
        if col in label:
            name = label[col]
        elif col.startswith('ComovingMostBoundDistance'):
            name = f'{label[col[:-1]]}_{col[-1]}'
        if '/' in col:
            scol = col.split('/')
            if len(scol) != 2:
                raise ValueError(f'ambiguous column name {col}')
            num = self.label(scol[0], with_units=False).replace('$', '')
            den = self.label(scol[1], with_units=False).replace('$', '')
            name = fr'{num} / {den}'
            if with_units:
                uns = [self.units(scol[0]), self.units(scol[1])]
                if uns[0] != uns[1]:
                    name = fr'{name}\,({uns[0]}/{uns[1]})'
        else:
            if col[:7] == 'history':
                histlabel = self.label_historical(col)
                q = col.split(':')[2]
                name = fr'{label[q]}^\mathrm{{{histlabel}}}'
            if with_units:
                un = self.units(col)
                if un:
                    name = fr'{name}\,({un})'
        if name:
            return fr'${name}$'
        return col

    def label_historical(self, event):
        if event[:7] != 'history':
            raise ValueError(f'column {event} not a historical column')
        labels = {'first_infall': 'infall', 'last_infall': 'acc',
                  'cent': 'cent', 'sat': 'sat'}
        event = event.split(':')
        if event[1] in labels:
            return labels[event[1]]
        return event[1]

    def units(self, col):
        units = {'mass': r'\mathrm{M}_\odot', 'time': 'Gyr',
                 'distance': r'h^{-1}\mathrm{Mpc}'}
        if col in self.sim.mass_columns:
            return units['mass']
        elif 'distance' in col.lower():
            return units['distance']
        elif 'time' in col.lower():
            return units['time']
        return ''


class BasePlotter(PlotterHelper):

    def __init__(self, sim, subs):
        self.sim = sim
        self.subs = subs

