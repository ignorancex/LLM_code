"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""


from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt, cm
import matplotlib as mpl
import datetime as dt
import locale

from case_parameters import CaseParameters
from config import Config
from result_exporter import ResultExporter
from utils import *
from variant_parameters import VariantParameters

#latex setup for labels
locale.setlocale(locale.LC_ALL, 'en_US')
locale._override_localeconv = {'thousands_sep': ' '}
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#specify colors and color factors
RECOVEREDCOLOR = [30/255,94/255,172/255]
VACCINATEDCOLOR = [30/255,140/255,47/255]
ACTIVEVARIANTSCOLORS = {
	'WILDTYPE':[204/255,102/255,0],
	'ALPHA':[0,85/255,170/255],
	'DELTA':[210/255,0,150/255],
	'OMICRON_BA1':[17/255,170/255,9/255],
	'OMICRON_BA2': [250 / 255, 200 / 255, 0 / 255],
	'OMICRON_BA4': [100 / 255, 200 / 255, 200 / 255],
	'OMICRON_BA5': [50 / 255, 200 / 255, 215 / 255]
}

VACCINATEDFACTOR = 0.8
VACCINATEDRECOVEREDFACTOR1 = 0.5
UNDETECTEDFACTOR=0.75
DETECTEDFACTOR = 1.0

class ResultPlotter:
	def __init__(self) -> None:
		"""
		Class to plot the simulation results
		"""
		self.Results = None
		self.Variants = None
		self.peaks = {}
		self.plotPDF = False
		self.darkBG = False
		self.dpi = 500

	def load_result(self,csvfilename:str):
		re = ResultExporter()
		self.Results, self.Variants = re.load_from_csv(csvfilename)
		try:
			self.Variants.remove('WILDTYPE')
			self.Variants.insert(0,'WILDTYPE')
		except:
			None
		self.peaks = {}

	def set_plotPDF(self,plotPDF):
		self.plotPDF = plotPDF

	def set_darkBG(self,darkBG):
		self.darkBG = darkBG

	def set_dpi(self,dpi):
		self.dpi = dpi

	def set_result(self,result:dict,variants:list):
		self.Results = result
		self.Variants = variants
		try:
			self.Variants.remove('WILDTYPE')
			self.Variants.insert(0,'WILDTYPE')
		except:
			None
		self.peaks = {}

	def fill_between(self, time:list, lower:list, upper:list, **kwargs) -> plt.fill:
		"""
		Convenience function to make filling between lines easier
		:param time: vector of times
		:param lower: vector of lower values
		:param upper: vector of upper values
		:param kwargs: optional kwargs for the 'fill' function
		:return:
		"""
		return plt.fill(time + time[::-1] + [time[0]], lower + upper[::-1] + [lower[0]], **kwargs, edgecolor=None)

	def _make_all_texts_white(self) -> None:
		"""
		Called if the darkBG variable is set to True. Sets all texts and axis features to bright gray to allow display on dark slides
		Only useful for png output
		:return:
		"""
		col = [204/255, 204/255, 204/255]
		ax = plt.gca()
		ax.spines['bottom'].set_color(col)
		ax.spines['top'].set_color(col)
		ax.spines['left'].set_color(col)
		ax.spines['right'].set_color(col)
		ax.xaxis.label.set_color(col)
		ax.yaxis.label.set_color(col)
		ax.tick_params(axis='x',colors=col)
		ax.tick_params(axis='y',colors=col)
		legends = [c for c in ax.get_children() if isinstance(c, mpl.legend.Legend)]
		for l in legends:
			for t in l.texts:
				t.set_color(col)
			l.legendPatch.set_facecolor([55/255*0.6,59/255*0.6,60/255*0.6])
			l.legendPatch.set_edgecolor(col)

	def xaxis_settings(self,Results=None) -> None:
		timeticks = [dt.datetime(2020, i, 1) for i in range(1, 13)]
		timeticks.extend([dt.datetime(2021, i, 1) for i in range(1, 13)])
		timeticks.extend([dt.datetime(2022, i, 1) for i in range(1, 13)])
		plt.xticks(timeticks, [x.strftime('%Y-%m-%d') for x in timeticks], rotation=40, ha='right')
		if Results!=None:
			plt.xlim([Results['time'][0], Results['time'][-1]])
		plt.grid(True, which='major', axis='both', linestyle='dotted', linewidth=0.5, zorder=-100)

	def plot_cases_by_variant(self,plotName:str, newconfirmed=False) -> None:
		"""
		Plots active or new confirmed cases by variant
		:param plotName: path for saving the first plot
		:param newconfirmed: if false, active detected and undetected cases are plotted by variant; otherwise daily new confirmed cases
		:return:
		"""
		if newconfirmed and not len([x for x in self.Results.keys() if x.startswith('new confirmed ')]) > 0:
			return

		plt.figure(figsize=[8, 4])
		lg = list()
		lower = [0 for x in self.Results['time']]
		for v in self.Variants:
			vname = vname_function(v)
			if newconfirmed:
				upper = [x + y for x, y in zip(lower, self.Results['new confirmed ' + v])]
				lg.append('new confirmed cases ' + vname)
			else:
				upper = [x + y for x, y in zip(lower, self.Results['active detected ' + v])]
				lg.append('active detected cases ' + vname)
			pp1, = self.fill_between(self.Results['time'], lower, upper,
									 color=[x * DETECTEDFACTOR for x in ACTIVEVARIANTSCOLORS[v]], linewidth=0.0,
									 zorder=0)
			# pp1 = self.fill_between(Results['time'],lower,upper,color=[x*DETECTEDFACTOR for x in ACTIVEVARIANTSCOLORS[v]], linewidth=0.0, zorder=0,alpha=1.0)
			lower = upper

		if not newconfirmed:
			for v in self.Variants:
				vname = vname_function(v)
				upper = [x + y for x, y in zip(lower, self.Results['active undetected ' + v])]
				lg.append('active undetected cases ' + vname)
				pp1, = self.fill_between(self.Results['time'], lower, upper,
										 color=[x * UNDETECTEDFACTOR for x in ACTIVEVARIANTSCOLORS[v]], linewidth=0.0,
										 zorder=0)
				# pp1 = self.fill_between(Results['time'],lower,upper,color=[x*DETECTEDFACTOR for x in ACTIVEVARIANTSCOLORS[v]], linewidth=0.0, zorder=0,alpha=1.0)
				lower = upper

		if newconfirmed:
			plt.ylabel('new confirmed cases')
		else:
			plt.ylabel('active cases')
		plt.legend(lg, loc='upper left')
		# plt.legend(lg[:5],loc='upper left')
		# plt.legend([pp1,pp2],['active detected cases','active undetected cases'],loc='upper left')
		# plt.legend(['active detected cases'],loc='upper left')
		self.xaxis_settings(self.Results)
		plt.gca().set_ylim(bottom=0)
		lm = plt.ylim()
		pos = plt.yticks()[0]
		plt.yticks(pos, [locale.format_string("%d", x, grouping=True) for x in pos])
		plt.ylim(lm)
		if self.darkBG:
			self._make_all_texts_white()
		# duplicate axis for showing numbers in % of population
		ax2 = plt.gca().twinx()
		ax2.yaxis.set_ticks_position('right')
		ax2.yaxis.set_visible(True)
		percents = [x * self.Results['population'][0] / 100 for x in range(0, 100)]
		plt.yticks(percents, [str(x) + '\%' for x in range(0, 100)])
		# plt.ylim([0,1.1*(max(Results['active detected'])+max(Results['active undetected']))])
		plt.ylim(lm)
		plt.tight_layout()
		# plt.savefig(plotName)
		if self.darkBG:
			self._make_all_texts_white()
		plt.tight_layout()
		if self.darkBG:
			plt.savefig(plotName, dpi=self.dpi, transparent=True)
		else:
			plt.savefig(plotName, dpi=self.dpi)
		if self.plotPDF:
			plt.savefig(plotName.replace('.png', '.pdf'))

	def plot_immunity(self, config:Config, plotName1:str,  plotName2:str, target:str) -> None:
		"""
		Plots immunisation level against certain target
		:param config: config instance of the simulation
		:param plotName1: path for saving the first plot
		:param plotName2: path for saving the second plot
		:param target: immunisation target as string
		:return:
		"""
		##### check peak ####
		if not target in self.peaks.keys():
			self.peaks[target] = list()
			if 'new confirmed ' + target in self.Results.keys():
				cases = self.Results['new confirmed ' + target]
				times = self.Results['time']
				macases = [
					sum([cases[i - 3], cases[i - 2], cases[i - 1], cases[i], cases[i + 1], cases[i + 2], cases[i + 3]]) / 7
					for i in range(3, len(cases) - 3)]
				timesma = times[3:len(cases) - 3]
				vp = VariantParameters(config)
				for i in range(10, len(macases) - 10):
					if macases[i] == max(macases[max(0, i - 50):min(i + 50, len(macases))]):
						ind = vp.get_variants().index(target)
						ratio = vp.get_variant_ratio(timesma[i].date())[ind]
						if ratio > 0.5:
							try:
								self.peaks[target].append(timesma[i])
							except:
								self.peaks[target] = [timesma[i]]

		############################ IMMUNITY LOSS PLOT #####################

		vname = vname_function(target)
		plt.figure(figsize=[8, 4])
		a1 = self.Results['past detected immune '+target]
		a2 = self.Results['past undetected immune '+target]
		a3 = [x + y for x, y in
			  zip(self.Results['past detected + vaccinated immune '+target], self.Results['past undetected + vaccinated immune '+target])]
		a4 = self.Results['vaccinated immune '+target]
		p1 = self.fill_between(self.Results['time'], [0 for x in self.Results['time']], a1,
							   color=[x * DETECTEDFACTOR for x in RECOVEREDCOLOR], zorder=0)
		p2 = self.fill_between(self.Results['time'], a1, [x + y for x, y in zip(a1, a2)],
							   color=[x * UNDETECTEDFACTOR for x in RECOVEREDCOLOR], zorder=0)
		p2 = self.fill_between(self.Results['time'], [x + y for x, y in zip(a1, a2)],
							   [x + y + z for x, y, z in zip(a1, a2, a3)],
							   color=[x * VACCINATEDRECOVEREDFACTOR1 for x in VACCINATEDCOLOR], zorder=0)
		p3 = self.fill_between(self.Results['time'], [x + y + z for x, y, z in zip(a1, a2, a3)],
							   [x + y + z + a for x, y, z, a in zip(a1, a2, a3, a4)],
							   color=[x * VACCINATEDFACTOR for x in VACCINATEDCOLOR], zorder=0)
		immunizationLevel = [x + y + z + a for x, y, z, a in zip(a1, a2, a3, a4)][-1] / self.Results['population'][0]
		print('------------')
		print(vname)
		print('immunization Level :' + str(immunizationLevel))
		self.xaxis_settings(self.Results)
		plt.ylabel('immunes')
		plt.legend(['immune against ' + vname + ' by detected infection',
					'immune against ' + vname + ' by undetected infection',
					'immune against ' + vname + ' by vaccination + infection',
					'immune against ' + vname + ' by vaccination'], loc='upper left')
		mx = self.Results['population'][0]
		# plot peaks
		if target in self.peaks.keys():
			dtss = self.peaks[target]
			for dts in dtss:
				i = self.Results['time'].index(dts)
				level = [x + y + z + a for x, y, z, a in zip(a1, a2, a3, a4)][i]
				xl = plt.xlim()
				plt.plot([xl[0], xl[1]], [level, level], color='r', linestyle='dashed', linewidth=1)
				# sine
				fun = lambda A, x: A + (1 - A) / 2 + (1 - A) / 2 * np.cos(x * 2 * np.pi)
				rat = (dts - dt.datetime(dts.year, 1, 1)).days / (
						dt.datetime(dts.year, 12, 31) - dt.datetime(dts.year, 1, 1)).days
				# y = fun(0.4,rat)
				# fac1= level/y
				y = fun(0.6, rat)
				fac2 = level / y

				# ys1 = list()
				ys2 = list()
				for d in self.Results['time']:
					rat = (d - dt.datetime(d.year, 1, 1)).days / (
								dt.datetime(d.year, 12, 31) - dt.datetime(d.year, 1, 1)).days
					# ys1.append(fac1*fun(0.4,rat))
					ys2.append(fac2 * fun(0.6, rat))
					d += dt.timedelta(1)
				# plt.plot(Results['time'], ys1, color='r', linestyle='dotted', linewidth=1)
				plt.plot(self.Results['time'], ys2, color='r', linestyle='dotted', linewidth=1)

				plt.scatter([self.Results['time'][i]], [level], color='r', marker='x')
				plt.text(self.Results['time'][i - 4], level + 100000, 'peak ' + vname, va='bottom', ha='right', color='r')
				plt.xlim(xl)
		plt.ylim([0, mx])
		pos = plt.yticks()[0]
		plt.yticks(pos, [locale.format_string("%d", x, grouping=True) for x in pos])
		plt.ylim([0, mx])
		if self.darkBG:
			self._make_all_texts_white()
		# duplicate axis for showing numbers in % of population
		ax2 = plt.gca().twinx()
		ax2.yaxis.set_ticks_position('right')
		ax2.yaxis.set_visible(True)
		percents = [x *  self.Results['population'][0] / 100.0 for x in range(0, 100, 5)]
		plt.yticks(percents, [str(x) + '\%' for x in range(0, 100, 5)])
		plt.ylim([0, mx])
		if self.darkBG:
			self._make_all_texts_white()
		plt.tight_layout()
		if self.darkBG:
			plt.savefig(plotName1, dpi=self.dpi, transparent=True)
		else:
			plt.savefig(plotName1, dpi=self.dpi)
		if self.plotPDF:
			plt.savefig(plotName1.replace('.png', '.pdf'))

		############################## COMPARISON WITH IMMUNITY EVENTS ###########################
		a1 = self.Results['past detected immune '+target]
		a2 = self.Results['past undetected immune '+target]
		a31 = self.Results['past detected + vaccinated immune '+target]
		a32 = self.Results['past undetected + vaccinated immune '+target]
		a4 = self.Results['vaccinated immune '+target]
		b1 = self.Results['past detected susceptible '+target]
		b2 = self.Results['past undetected susceptible '+target]
		b31 = self.Results['past detected + vaccinated susceptible '+target]
		b32 = self.Results['past undetected + vaccinated susceptible '+target]
		b4 = self.Results['vaccinated susceptible '+target]

		plt.figure(figsize=[10, 5])
		plt.subplot(1, 2, 1)
		lower = [0 for x in self.Results['time']]
		values = a1
		upper = [x + y for x, y in zip(lower, values)]
		pl1 = plt.fill_between(self.Results['time'], lower, upper, color=[x * DETECTEDFACTOR for x in RECOVEREDCOLOR])
		lower = upper
		values = a2
		upper = [x + y for x, y in zip(lower, values)]
		pl2 = plt.fill_between(self.Results['time'], lower, upper, color=[x * UNDETECTEDFACTOR for x in RECOVEREDCOLOR])
		lower = upper
		values = [x + y for x, y in zip(a31, a32)]
		upper = [x + y for x, y in zip(lower, values)]
		pl3 = plt.fill_between(self.Results['time'], lower, upper,
							   color=[x * VACCINATEDRECOVEREDFACTOR1 for x in VACCINATEDCOLOR])
		lower = upper
		values = a4
		upper = [x + y for x, y in zip(lower, values)]
		pl4 = plt.fill_between(self.Results['time'], lower, upper, color=VACCINATEDCOLOR)
		lower = upper
		plt.legend([pl4, pl3, pl2, pl1], ['immune against ' + vname + '\nby vaccination',
										  'immune against ' + vname + '\nby vaccination + infection',
										  'immune against ' + vname + '\nby undetected infection',
										  'immune against ' + vname + '\nby detected infection'], loc='upper left')
		mx =  self.Results['population'][0]
		plt.ylim([0, mx])
		pos = plt.yticks()[0]
		plt.yticks(pos, [locale.format_string("%d", x, grouping=True) for x in pos])
		plt.ylim([0, mx])
		self.xaxis_settings(self.Results)
		plt.xticks(rotation=90)
		if self.darkBG:
			self._make_all_texts_white()
		# duplicate axis for showing numbers in % of population
		ax2 = plt.gca().twinx()
		ax2.yaxis.set_ticks_position('right')
		ax2.yaxis.set_visible(True)
		percents = [x *  self.Results['population'][0] / 100.0 for x in range(0, 100, 5)]
		plt.yticks(percents, ["" for x in range(0, 100, 5)])
		plt.ylim([0, mx])
		if self.darkBG:
			self._make_all_texts_white()
		# display maximum reachable level of immunity (if everything worked and remained forever)
		plt.subplot(1, 2, 2)
		lower = [0 for x in self.Results['time']]
		values = [x + y for x, y in zip(a1, b1)]
		upper = [x + y for x, y in zip(lower, values)]
		pl1 = plt.fill_between(self.Results['time'], lower, upper, color=[x * DETECTEDFACTOR for x in RECOVEREDCOLOR],
							   alpha=0.5, linewidth=0, zorder=0)
		plt.plot(self.Results['time'], upper, color=[x * DETECTEDFACTOR for x in RECOVEREDCOLOR], zorder=1,
				 linestyle='dashed')
		lower = upper
		values = [x + y for x, y in zip(a2, b2)]
		upper = [x + y for x, y in zip(lower, values)]
		pl2 = plt.fill_between(self.Results['time'], lower, upper, color=[x * UNDETECTEDFACTOR for x in RECOVEREDCOLOR],
							   alpha=0.5, linewidth=0, zorder=0)
		plt.plot(self.Results['time'], upper, color=[x * UNDETECTEDFACTOR for x in RECOVEREDCOLOR], zorder=1,
				 linestyle='dashed')
		lower = upper
		values = [x + y + a + b for x, y, a, b in zip(a31, a32, b31, b32)]
		upper = [x + y for x, y in zip(lower, values)]
		pl3 = plt.fill_between(self.Results['time'], lower, upper,
							   color=[x * VACCINATEDRECOVEREDFACTOR1 for x in VACCINATEDCOLOR], alpha=0.5, linewidth=0,
							   zorder=0)
		plt.plot(self.Results['time'], upper, color=[x * VACCINATEDRECOVEREDFACTOR1 for x in VACCINATEDCOLOR], zorder=1,
				 linestyle='dashed')
		lower = upper
		values = [x + y for x, y in zip(a4, b4)]
		upper = [x + y for x, y in zip(lower, values)]
		pl4 = plt.fill_between(self.Results['time'], lower, upper, color=VACCINATEDCOLOR, alpha=0.5, linewidth=0, zorder=0)
		plt.plot(self.Results['time'], upper, color=VACCINATEDCOLOR, zorder=1, linestyle='dashed')
		lower = upper
		plt.legend([pl4, pl3, pl2, pl1],
				   ['vaccinated', 'vaccinated + infected', 'undetected infected', 'detected infected'],
				   loc='upper left')
		mx = self.Results['population'][0]
		plt.ylim([0, mx])
		pos = plt.yticks()[0]
		plt.yticks(pos, ["" for x in pos])
		plt.ylim([0, mx])
		self.xaxis_settings(self.Results)
		plt.xticks(rotation=90)
		if self.darkBG:
			self._make_all_texts_white()
		# duplicate axis for showing numbers in % of population
		ax2 = plt.gca().twinx()
		ax2.yaxis.set_ticks_position('right')
		ax2.yaxis.set_visible(True)
		percents = [x *  self.Results['population'][0] / 100.0 for x in range(0, 100, 5)]
		plt.yticks(percents, [str(x) + '\%' for x in range(0, 100, 5)])
		plt.ylim([0, mx])
		if self.darkBG:
			self._make_all_texts_white()
		plt.tight_layout()
		if self.darkBG:
			plt.savefig(plotName2, dpi=self.dpi, transparent=True)
		else:
			plt.savefig(plotName2, dpi=self.dpi)
		if self.plotPDF:
			plt.savefig(plotName2.replace('.png', '.pdf'))


	def plot_reinfections(self, plotName:str, Relative=bool) -> None:
		"""
		Plots reinfections in a bubble diagram
		:param plotName: path to save the plot
		:param Relative: if true, then the edges and bubbles are plotted normed to 10000 infections
		:return:
		"""
		#cmp = cm.get_cmap('plasma')
		#for i,v in enumerate(self.Variants):
		#	ACTIVEVARIANTSCOLORS[v]=cmp(i/len(self.Variants))

		incs = dict()
		for v1 in self.Variants:
			for v2 in self.Variants:
				incs[(v1, v2)] = sum(self.Results['detected reinfection ({},{})'.format(v1, v2)])
		for v2 in self.Variants:
			incs[(None, v2)] = sum(self.Results['detected reinfection (None,{})'.format(v2)])
		incidences = dict()
		for v1 in self.Variants:
			for v2 in self.Variants:
				incidences[(v1, v2)] = [0, sum([y for x, y in incs.items() if x[1] == v1]),
										sum([y for x, y in incs.items() if x[1] == v2]), incs[(v1, v2)]]
		if Relative:
			for k in incidences.keys():
				incidences[k][-1] = incidences[k][-1] / incidences[k][-2] / incidences[k][-3] * 10000 * 10000
				incidences[k][-2] = 10000
				incidences[k][-3] = 10000
		mx = max([x[-1] for x in incidences.values()])

		########### BUBBLE Diagram #######
		plt.figure(figsize=[16, 10])
		fac21 = 4
		fac22 = 250
		if Relative:
			fac2 = fac22
		else:
			fac2 = fac21
		R = 1

		cts = [incidences[(v, v)][-2] for v in self.Variants]
		sm = sum(cts) + sum(cts[1:-1])
		factor = np.pi / sm / 1.5
		radius = [factor * x for x in cts]

		angles = list()
		x = 0
		for i in range(len(cts) - 1):
			angles.append(x)
			x += cts[i]
			x += cts[i + 1]
		angles.append(x)
		angles = [np.pi - x / angles[-1] * np.pi for x in angles]

		# boxst = {'facecolor':[1,1,1],'alpha':0.7,'boxstyle':'round','edgecolor':[1,1,1]}

		mids = dict()
		for i, v in enumerate(self.Variants):
			angle = angles[i]
			mids[v] = [R * np.cos(angle), R * np.sin(angle)]
			xs = list()
			ys = list()
			for j in range(361):
				xs.append(mids[v][0] + radius[i] * np.cos(j * np.pi / 180))
				ys.append(mids[v][1] + radius[i] * np.sin(j * np.pi / 180))
			plt.fill(xs, ys, color=ACTIVEVARIANTSCOLORS[v], zorder=i, linewidth=0)
			if sum(ACTIVEVARIANTSCOLORS[v][:3]) < 1.5:
				fontc = [1, 1, 1]
			else:
				fontc = [0, 0, 0]

			if Relative:
				txt = '{:}\n{:.0f}'.format(v.replace('_','\_'), incidences[(v, v)][-2])
			else:
				txt = '{:}\n{:.0f}'.format(v.replace('_','\_'), incidences[(v, v)][-2] / 1000)
			if radius[i] < 0.1:
				plt.text((R + 0.15) * np.cos(angle), (R + 0.15) * np.sin(angle), txt, ha='center', va='center',
						 zorder=100)  # ,bbox=boxst)
			else:
				plt.text(mids[v][0], mids[v][1], txt, ha='center', va='center', color=fontc, zorder=100)  # ,bbox=boxst)

		for i, v in enumerate(self.Variants):
			for v2 in self.Variants[i + 1:]:
				i2 = self.Variants.index(v2)
				vec = [mids[v2][0] - mids[v][0], mids[v2][1] - mids[v][1]]
				norm = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
				vec = [x / norm for x in vec]
				x0 = mids[v][0] + radius[i] * vec[0]
				x1 = mids[v2][0] - radius[i2] * vec[0]
				y0 = mids[v][1] + radius[i] * vec[1]
				y1 = mids[v2][1] - radius[i2] * vec[1]
				if Relative:
					xm = x0 + 0.1 * vec[0]
					ym = y0 + 0.1 * vec[1]
				else:
					xm = x0 + 0.05 * vec[0]
					ym = y0 + 0.05 * vec[1]
				# plt.plot([x0,x1],[y0,y1],linewidth=incidences[(v,v2)][-1]*factor*fac2,color=cmp(i/len(variants)),zorder=i)
				wdth = incidences[(v, v2)][-1] * factor * fac2
				if wdth > 0.02:
					headwdth = wdth
				else:
					headwdth = 0.02
				plt.arrow(mids[v][0], mids[v][1], x1 - mids[v][0], y1 - mids[v][1], width=wdth,
						  color=ACTIVEVARIANTSCOLORS[v], zorder=i, length_includes_head=True,
						  head_width=headwdth)  # ,capstyle='butt')
				if sum(ACTIVEVARIANTSCOLORS[v][:3]) < 1.5:
					fontc = [1, 1, 1]
				else:
					fontc = [0, 0, 0]
				boxst = {'color': ACTIVEVARIANTSCOLORS[v], 'alpha': 1.0, 'boxstyle': 'round'}
				rot = np.arctan2(mids[v2][1] - mids[v][1], mids[v2][0] - mids[v][0]) * 180 / np.pi
				if Relative:
					txt = '{:.02f}'.format(incidences[(v, v2)][-1])
				else:
					txt = '{:.0f}'.format(incidences[(v, v2)][-1] / 1000)
				plt.text(xm, ym, txt, rotation=rot, ha='center', va='center', color=fontc, bbox=boxst, zorder=100)

		plt.axis('equal')
		plt.axis('off')
		if Relative:
			plt.title('relative reinfections')
			plt.savefig(plotName,dpi=self.dpi)
			if self.plotPDF:
				plt.savefig(plotName.replace('.png','.pdf'))
		else:
			plt.title('infections and reinfections (in 1000 persons)')
			plt.savefig(plotName, dpi=self.dpi)
			if self.plotPDF:
				plt.savefig(plotName.replace('.png','.pdf'))

	'''
	def reevalPlot(self,prefix,csvfilename:str,target:str) -> None:
		"""
		Routine plots the content of the csv file.
		Method creates three plots:
		plot of active cases per variant
		plot of overall level of immunity against the observed variant
		comparison plot of actual and potential level of immunity against the observed variant
		:param csvfilename: name of the csv file
		:param target: name of the observable
		:return:
		"""
		self.popaustria = sum(POPULATION.values())
		infectedFile = csvfilename
		filenamePrefix = prefix
		plotName0 = os.path.join('reeval',filenamePrefix+'_reeval_confirmed.png')

		#specify colors and color factors
		recoveredColor = [30/255,94/255,172/255]
		vaccinatedColor = [30/255,140/255,47/255]
		activeVariantsColors = {
			'WILDTYPE':[204/255,102/255,0],
			'ALPHA':[0,85/255,170/255],
			'DELTA':[210/255,0,150/255],
			'OMICRON_BA1':[17/255,170/255,9/255],
			'OMICRON_BA2': [250 / 255, 200 / 255, 0 / 255],
			'OMICRON_BA4': [100 / 255, 200 / 255, 200 / 255],
			'OMICRON_BA5': [80 / 255, 80 / 255, 80 / 255]
		}

		#activeVariantsColors = {x:activeVariantsColors['WILDTYPE'] for x in activeVariantsColors.keys()}

		vaccinatedFactor = 0.8
		vaccinatedRecoveredFactor1 = 0.5
		undetectedFactor=0.75
		detectedFactor = 1.0

		Results,Variants = self.load(infectedFile,target)
		Variants.remove('WILDTYPE')
		Variants.insert(0,'WILDTYPE') #make sure wildtype is first in list

		################### CONFIRMED CASES PLOT ############
		if len([x for x in Results.keys() if x.startswith('new_confirmed_')]) > 0:
			plt.figure(figsize=[8, 4])
			lg = list()
			lower = [0 for x in Results['time']]
			vs = [x for x in Variants]
			for v in vs:
				try:
					upper = [x + y for x, y in zip(lower, Results['new_confirmed_' + v])]
					if v=='OMICRON':
						v='OMICRON_BA1'
					elif v=='OMICRON2':
						v = 'OMICRON_BA2'
					pp1, = self.fill_between(Results['time'], lower, upper,
											 color=[x * detectedFactor for x in activeVariantsColors[v]],
											 linewidth=0.0, zorder=0)
					# pp1 = self.fill_between(Results['time'],lower,upper,color=[x*detectedFactor for x in activeVariantsColors[v]], linewidth=0.0, zorder=0,alpha=1.0)
					lower = upper
					vname = vname_function(v)
					vname = vname.replace('BA.4','Inf+')
					vname = vname.replace('Inf+/5','Esc+')
					lg.append('new confirmed cases ' + vname)
				except:
					raise()
			plt.ylabel('new confirmed cases')
			cp = CaseParameters(Config('config_base.json',''))
			dates = list(cp.cases.keys())
			dates.sort()
			values = [cp._get(x,True) for x in dates]
			valuesMa = [(a+b+c+d+e+f+g)/7 for a,b,c,d,e,f,g in zip(values[:-6],values[1:-5],values[2:-4],values[3:-3],values[4:-2],values[5:-1],values[6:])]
			plt.plot(dates,values,'r',linewidth=0.3,linestyle='dashed',zorder=1)
			plt.plot(dates[6:],valuesMa,'r',linewidth=1,zorder=1)
			lg.insert(0,'new confirmed cases (2022-09-28)')
			lg.insert(0,'new confirmed cases 7d-average (2022-09-28)')
			plt.plot([dt.datetime(2022,5,16),dt.datetime(2022,5,16)],[0,80000],'k',linewidth=0.5,linestyle='dotted')
			plt.text(dt.datetime(2022,5,16),40000,'date of forecast',rotation=90,ha='right',va='center')

			plt.legend(lg, loc='upper left')
			# plt.legend(lg[:5],loc='upper left')
			# plt.legend([pp1,pp2],['active detected cases','active undetected cases'],loc='upper left')
			# plt.legend(['active detected cases'],loc='upper left')
			self.xaxis_settings(Results)
			plt.gca().set_ylim(bottom=0)
			lm = plt.ylim()
			pos = plt.yticks()[0]
			plt.yticks(pos, [locale.format_string("%d", x, grouping=True) for x in pos])
			plt.ylim(lm)
			if DARKBG:
				self._make_all_texts_white()
			# duplicate axis for showing numbers in % of population
			ax2 = plt.gca().twinx()
			ax2.yaxis.set_ticks_position('right')
			ax2.yaxis.set_visible(True)
			percents = [x * self.popaustria / 100 for x in range(0, 100)]
			plt.yticks(percents, [str(x) + '\%' for x in range(0, 100)])
			# plt.ylim([0,1.1*(max(Results['active detected'])+max(Results['active undetected']))])
			plt.ylim(lm)
			if DARKBG:
				self._make_all_texts_white()
			plt.tight_layout()
			if DARKBG:
				plt.savefig(plotName0, dpi=DPI, transparent=True)
			else:
				plt.savefig(plotName0, dpi=DPI)
	'''
