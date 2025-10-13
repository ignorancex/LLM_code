from psychopy import visual, core, event, monitors, logging
from psychopy.hardware import keyboard
from psychopy_visionscience import NoiseStim
import os
import sys
import pandas as pd
import pyglet
import pickle as pkl


class Experiment():

	def __init__(self, subject_id):
		self.exp_path = 'human_experiment/data/exp_samples.xlsx' 
		self.practice_path = 'human_experiment/data/practice_samples.xlsx'
		
		self.subject_id = subject_id

		# set practice based on the experiment condition
		# condition 1 is questions 3 and 6
		# condition 2 is questions 4 and 5
		if self.subject_id < 50:
			self.practice_subject_id = 1
		else:
			self.practice_subject_id = 2
		
		self.cache_path = f'human_experiment/cache/subject_{subject_id}.pkl'
		self.save_path = f'human_experiment/data/subject_{subject_id}.xlsx'

		# load data
		practice_data_df = pd.read_excel(self.practice_path)
		self.practice_data_df = practice_data_df[practice_data_df['Subject_id'] == self.practice_subject_id]

		exp_data_df = pd.read_excel(self.exp_path)
		self.exp_data_df = exp_data_df[exp_data_df['Subject_id']==self.subject_id]

		self.unique_q_idx = sorted(list(set(self.exp_data_df['Q_id'].to_list())))

		#print(self.unique_q_idx)

		if self.exp_data_df.empty:
			raise ValueError(f'ERROR: invalid subject_id={subject_id}! Valid range is [1, 100]')

		#### MONITOR SETUP ####
		monitor = 'MacBookAir'

		if monitor == 'SAMSUNG':
			# 22 in SAMSUNG monitor
			# physical dimensions 508x285.75
			self.monitor = monitors.Monitor('SAMSUNG', width=50.8, distance=60)
			self.monitor.setSizePix([1920, 1080])
			refresh = 1.0/60.0 # 60Hz refresh rate

			# setup window
			self.win = visual.Window([1920, 1080], monitor='SAMSUNG', units='pix', allowGUI=True, fullscr=True, checkTiming=False)

		else:
			# setup monitor
			# MacBook Air monitor
			# physical dimensions 285.75 x 179.29 mm
			self.monitor = monitors.Monitor('MacBookAir', width=28.575, distance=60)
			self.monitor.setSizePix([1440, 900])
			refresh = 1.0/60.0 # 60Hz refresh rate

			# setup window
			self.win = visual.Window([1440, 900], monitor='MacBookAir', units='pix', allowGUI=True, fullscr=True, checkTiming=False)

		##########################

		self.win.recordFrameIntervals = True
		self.win.refreshThreshold = refresh + 0.004

		# setup mouse and logging
		self.mouse = event.Mouse(win=self.win, visible=True)
		#self.keyboard = keyboard.Keyboard()

		self.key=pyglet.window.key
		self.keyboard = self.key.KeyStateHandler()
		self.win.winHandle.push_handlers(self.keyboard)

		self.textbox = visual.TextBox2(self.win, text='',
										pos=(0,-0.05), 
										letterHeight=0.04, 
						                size=[0.6, 0.1],
						                anchor='center-top',
						                borderColor='lightgrey',
										units='height',
										editable = True)

		logging.console.setLevel(logging.WARNING)

		# constants for the experiment
		image_dur = 0.2
		fix_dur = 1.5

		self.image_frames = int(image_dur/refresh)
		self.fix_frames = int(fix_dur/refresh)
		self.image_dir_path = os.environ['SNAP_DATA_PATH']
		self.practice_dir_path = 'human_experiment/data/practice_samples/'

		#'/Users/yulia/Documents/sensor_bias/data_v3/'

		# stimuli for the experiment
		self.fix = visual.TextStim(self.win, text='+', units='norm', height=0.1)
		self.noise_stim = NoiseStim(self.win, noiseType='White', pos=(0,0), size=(1440, 900))
		self.next_txt = visual.TextStim(self.win,
										text='Press space to continue...',
										pos=(0, -0.5),
										units='norm',
										height = 0.06,
										wrapWidth=0.8)

		self.next_txt1 = visual.TextStim(self.win,
										text='Press enter to continue...',
										pos=(0, -0.5),
										units='norm',
										height = 0.06,
										wrapWidth=0.8)
		event.globalKeys.clear()
		event.globalKeys.add(key='escape', func=core.quit)

	def parse_question(self, q_str):
		q, instr = [x.strip() for x in q_str.split('?')]

		if 'A)' in q_str:
			instr, ans_opts = [x.strip() for x in instr.split(':')]
			ans_opts = [x.strip() for x in ans_opts.split('   ')]
		elif 'other' in q_str:
			instr, ans_opts = [x.strip() for x in instr.split(':')]
			ans_opts = [x.strip() for x in ans_opts.split(',')]
		else:
			ans_opts = []

		return q+'?', instr, ans_opts

	def get_subject_info(self):
		save_path = f'human_experiment/data/subject_{self.subject_id}_info.txt'
		if os.path.exists(save_path):
			return True

		info_txt = visual.TextStim(self.win,
										text='A couple of questions before we begin\n\n\nHow old are you?',
										pos=(0, 0.25),
										units='norm',
										height = 0.07,
										wrapWidth=0.8)

		#a_txt = self.display_question(info_txt, [])

		age = self.get_keyboard_feedback(info_txt)

		info_txt = visual.TextStim(self.win,
										text='What is your gender?',
										pos=(0, 0.25),
										units='norm',
										height = 0.07,
										wrapWidth=0.8)
		ans_opts = ['male', 'female', 'prefer not to say']
		a_txt = self.display_question(info_txt, ans_opts)

		gender = self.get_mouse_feedback(info_txt, a_txt)

		with open(save_path, 'w') as fid:
			fid.write(f'age={age}, gender={gender}')
		print(f'Saved subject {self.subject_id} info')

		return False


	def get_feedback(self, q_id, q_txt, a_txt):
		if q_id in [3, 4]:
			return self.get_keyboard_feedback(q_txt)
		elif q_id in [5, 6]:
			return self.get_mouse_feedback(q_txt, a_txt)
		else:
			raise ValueError(f'ERROR: Wrong q_id {q_id}, should be {3,4,5,6}!')

	def get_keyboard_feedback(self, q_txt):
		self.textbox.clear()

		while '\n' not in self.textbox.text:
			q_txt.draw()
			self.textbox.draw()
			if len(self.textbox.text):
				self.next_txt1.draw()
			self.win.flip()

		ans = self.textbox.text[:-1]

		return ans

	def get_mouse_feedback(self, q_txt, a_txt):
		
		selected_idx = -1
		while True:
			for a_idx, a in enumerate(a_txt):
				if self.mouse.isPressedIn(a):
					# Set this variable to point to the latest pressed shape
					selected_idx = a_idx
					break

			for a_idx, a in enumerate(a_txt):
				if a_idx == selected_idx:
					a.color=(0, 1, 1)
					a.bold=True
				else:
					a.color=(1, 1, 1)
					a.bold=False
			
			q_txt.draw()
			for a in a_txt:
				a.draw()
			if selected_idx >= 0:
				self.next_txt.draw()
			
			self.win.flip()

			if selected_idx >= 0:
				if self.keyboard[self.key.SPACE]:
					break
				#keys_pressed = self.keyboard.getKeys['space'], clear=False, waitRelease=False)
				#if len(keys_pressed):
				#	break

		return a_txt[selected_idx].text

	def display_exp_intro(self):
		intro_str = '''In this experiment you will answer questions about images shown on the screen.

There will be 200 images split into 4 blocks of 50 trials.

Before every block you will see a question that will apply to *all* images in the block.

During each trial:

	1. Look at the cross in the center.
	2. An image will be shown *very* briefly.
	3. Answer the question: type or select with mouse.
	4. Press space to continue.

You can take a break after every block. 

This is *not* a speed test, answer at your own pace.

Before the experiment you will get a chance to practice.

Press space to start practice...'''

		intro_txt = visual.TextStim(
					self.win,
					text=intro_str,
					pos=(0, 0),
					units='norm',
					height = 0.06,
					wrapWidth=1.5)	

		intro_txt.draw()
		self.win.flip()
		event.waitKeys()


	def display_block_intro(self, block_id, q_txt, ans_opts, practice):
		
		if practice:
			block_str = f'Practice block {block_id}\n\nFor the next 10 images you will answer the following question:'
		else:
			block_str = f'Block {block_id}\n\nFor the next 50 images you will answer the following question:'

		if len(ans_opts):
			ans_str = 'After each image, click on the most suitable option.'
		else:
			ans_str = 'After each image, type the answer.'

		ans_str += '\n\nPress space to start...'

		block_txt = visual.TextStim(
					self.win,
					text=block_str,
					pos=(0, 0.25),
					units='norm',
					height = 0.07,
					wrapWidth=0.8)

		ans_txt = visual.TextStim(
					self.win,
					text=ans_str,
					pos=(0, -0.3),
					units='norm',
					height = 0.07,
					wrapWidth=0.8)

		block_txt.draw()
		q_txt.height=0.1
		q_txt.draw()
		q_txt.height = 0.07
		ans_txt.draw()

		self.win.flip()

	def display_block_outro(self, block_id, practice=False):

		if practice:
			if block_id == 2:
				block_str = 'End of practice!\n\nWhen ready to start the experiment, press space...' 
			else:
				block_str = f'Practice block {block_id} done!\n\nWhen ready to continue, press space...'				
		else:	
			if block_id == 4:
				block_str = 'End of experiment!\n\nThank you for participating!'
			else:
				block_str = f'Block {block_id} done!\n\nPlease take a break if you need one.\n\nWhen ready to continue, press space...'
		
		outro_txt = visual.TextStim(
					self.win,
					text=block_str,
					pos=(0, 0),
					units='norm',
					height = 0.07,
					wrapWidth=0.8)

		outro_txt.draw()
		self.win.flip()

	def display_question(self, q_txt, ans_opts):
		q_txt.draw()
		a_txt = []

		y_pos = -0.1
		num_opts = len(ans_opts)
		for a_idx, a_str in enumerate(ans_opts):
			if a_idx % 4 == 0:
				y_pos -= 0.07
			
			a = visual.TextStim(
						self.win,
						text=a_str,
						pos=((a_idx%4+1)/4-0.65, y_pos),
						units='norm',
						height=0.06,
						wrapWidth=0.8)
			a.draw()
			a_txt.append(a)

		self.win.flip()

		return a_txt

	def display_stimuli(self, image_path, practice=False):

		self.win.mouseVisible = False
		
		if practice:
			image_stim = visual.ImageStim(self.win,
								image=os.path.join(self.practice_dir_path,image_path),
								units='pix',
								size=(960, 640))
		else:
			image_stim = visual.ImageStim(self.win, image=os.path.join(self.image_dir_path,image_path))
		
		for frame_idx in range(self.fix_frames):
			self.fix.draw()
			self.win.flip()

		for frame_idx in range(self.image_frames):
			image_stim.draw()
			self.win.flip()

		for frame_idx in range(self.image_frames):
			self.noise_stim.draw()
			self.win.flip()
		
		self.win.mouseVisible = True

	def run_block(self, block_id, practice=False):

		print('Running block', block_id)

		if practice:
			block_df = self.practice_data_df[self.practice_data_df['Block_id']==block_id]
		else:
			block_df = self.exp_data_df[self.exp_data_df['Block_id'] == block_id]
		#print(block_df)

		image_paths = block_df['Path'].tolist()
		questions = block_df['Q'].tolist()
		gt = block_df['A'].tolist()
		q_idx = block_df['Q_id'].tolist()

		if practice:
			processed_images = []
		else:
			processed_images = [x['Path'] for x in self.answers if x['Block_id'] == block_id]

			if len(image_paths) == len(processed_images):
				return		

		image_idx = list(range(len(image_paths)))

		question = questions[0]

		q, instr, ans_opts = self.parse_question(question)
		print(question, q, instr, ans_opts)

		q_txt = visual.TextStim(
					self.win,
					text=q,
					pos=(0, 0),
					units='norm',
					height = 0.07,
					wrapWidth=1.5)

		self.display_block_intro(block_id, q_txt, ans_opts, practice=practice)
		event.waitKeys()

		for image_i, image_path, question, q_id in zip(image_idx, image_paths, questions, q_idx):
			if image_path in processed_images:
				continue

			q, instr, ans_opts = self.parse_question(question)

			self.display_stimuli(image_path, practice=practice)

			a_txt = self.display_question(q_txt, ans_opts)

			ans = self.get_feedback(q_id, q_txt, a_txt)

			if not practice:
				record = {'Path': image_path, 'Block_id': block_id, 'Subject_id': subject_id}
				
				for q_i in self.unique_q_idx:
					record[f'A{q_i}'] = '-' # set the default answer to dash

				record[f'A{q_id}'] = ans

				self.add_answer(record)

				self.cache_answers()

		if not practice:
			self.cache_answers()
		
		self.display_block_outro(block_id, practice=practice)

		event.waitKeys()

	def add_answer(self, record):
		self.answers.append(record)

	def load_answers(self):
		if os.path.exists(self.cache_path):
			with open(self.cache_path, 'rb') as fid:
				self.answers = pkl.load(fid)
		else:
			self.answers = []

	def cache_answers(self):
		with open(self.cache_path, 'wb') as fid:
			pkl.dump(self.answers, fid)

	def save_answers(self):
		ans_df = pd.DataFrame.from_dict(self.answers)
		ans_df.to_excel(self.save_path, index=False)

	def practice(self):
		block_ids = list(self.practice_data_df['Block_id'].unique())
		for block_id in block_ids:
			self.run_block(block_id, practice=True)

	def quit(self):

		print(f'{self.win.nDroppedFrames} frames dropped')
		self.win.close()
		core.quit()		

	def run(self):

		rerun = self.get_subject_info()

		if not rerun:
			self.display_exp_intro()

			self.practice()

		self.load_answers()

		block_ids = list(self.exp_data_df['Block_id'].unique())
		for block_id in block_ids:
			self.run_block(block_id)

		self.save_answers()

		self.quit()


if __name__ == '__main__':
	subject_id = int(sys.argv[1])

	print(f'Running experiment with subject {subject_id}')
	exp = Experiment(subject_id=subject_id)
	exp.run()

