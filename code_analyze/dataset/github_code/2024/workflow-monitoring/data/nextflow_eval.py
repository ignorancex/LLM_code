import csv
from decimal import Decimal
import logging
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import os
import portion as P
import re
import statistics
import sys

# usage: nextflow_eval.py <.nextflow.log file path> <global shared space prefix> <folder> [folder [...]]
# uses the nextflow log and the unique working directory of tasks to determine relevant processes inside the monitoring data to gather statistics on it
# different statistics / evaluations are available and can be commented in at the bottom of the file
# right now it would regenerate the figures used in the demonstration section of the paper given the same data

# constants / csv indices
BPF_TIME = 0
BPF_PID = 2
BPF_INODE = 7
BPF_OPERATION = 8
BPF_RESULT = 9
BPF_FILEHANDLE = 10
BPF_OFFSET = 11
BPF_SIZE = 12
BPF_FLAGS = 13
BPF_PATH = 14
CGROUPS_PARENT = 1
CGROUPS_CHILD = 2
CGROUPS_CGROUP = 3

PATTERN_READ = PATTERN_READONLY = 0b0001
PATTERN_WRITE = PATTERN_WRITEONLY = 0b0010
PATTERN_READWRITE = 0b0011
PATTERN_RANDOM = 0b0100
PATTERN_SEQUENTIAL = 0b1000

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.info('Script started.')

nflogpath = sys.argv[1]
globalworkdirprefix = sys.argv[2]
folderstocheck = [os.path.abspath(x) for x in sys.argv[3:] if os.path.isdir(os.path.abspath(x))]

def file_filter(file):
    #return True
    if file.path is None: # no files without pathes
        return False
    elif file.filesize == 0: # no files without size
        return False
    elif not file.path.startswith('/home/witzke/nf-rnaseq/outdir/work/'): # only files residing in the nextflow work directory
        return False
    else:
        ## only include files that are written to and read by tasks
        #pattern = 0
        #for accesses in file.accesses.values():
        #    pattern = pattern | accesses.pattern
        #if pattern & PATTERN_READWRITE != PATTERN_READWRITE:
        #    return False

        # take out a few nextflow specific files
        filename = file.path.split('/')[-1]
        if filename.startswith('.'): # no 'hidden' files (filters out all but one of the usual nextflow files)
            return False
        elif filename == 'versions.yml': # not the nextflow versions.yml
            return False
    return True

def accesses_filter(accesses):
    #return True
    if accesses.path is None:
        return False
    elif accesses.analysis_reads.unique_bytes < 10 and accesses.analysis_writes.unique_bytes < 10:
        return False
    #elif not accesses.path.startswith('/home/witzke/nf-rnaseq/outdir/work/'): # only files residing in the nextflow work directory
    #    return False
    else:
        # take out a few nextflow specific files
        filename = accesses.path.split('/')[-1]
        if filename.startswith('.'): # no 'hidden' files (filters out all but one of the usual nextflow files)
            return False
        elif filename == 'versions.yml': # not the nextflow versions.yml
            return False
    return True

class AccessAnalysis:
    min_offset = 0
    max_offset = 0
    unique_bytes = 0
    repetitive_bytes = 0
    total_bytes = 0
    seek_up = False
    seek_down = False
    first_access = 0.0
    last_access = 0.0

    def __init__(self):
        self.min_offset = 0
        self.max_offset = 0
        self.unique_bytes = 0
        self.repetitive_bytes = 0
        self.total_bytes = 0
        self.seek_up = False
        self.seek_down = False
        self.first_access = 0.0
        self.last_access = 0.0

# class that describes how a process accesses a file
class FileAccesses:
    pid = 0
    path = None

    # lists of the logged operations (ordered by timestamp); each entry is simply a list of the csv row
    opens = []
    closes = []
    reads = []
    writes = []
    deletes = []

    # access analysis
    pattern = 0
    analysis_reads = AccessAnalysis()
    analysis_writes = AccessAnalysis()
    first_log = Decimal('Infinity')
    last_log = Decimal(0)

    def __init__(self, pid) -> None:
        self.pid = pid
        self.path = None
        self.opens = list()
        self.closes = list()
        self.reads = list()
        self.writes = list()
        self.deletes = list()
        self.pattern = 0
        self.analysis_reads = AccessAnalysis()
        self.analysis_writes = AccessAnalysis()
        self.first_log = Decimal('Infinity')
        self.last_log = Decimal(0)

class File:
    inode = None
    path = None # last path the file was opened through
    total_read = 0
    total_read_size = 0
    total_writes = 0
    total_write_size = 0
    filesize = 0 # max offset read from or written to

    accesses = dict() # pids that touched the file mapped to a FileAccesses object

    def __init__(self) -> None:
        self.inode = None
        self.path = None
        self.total_reads = self.total_read_size = self.total_writes = self.total_write_size = self.filesize = 0
        self.accesses = dict()

class Task: #r"TaskHandler\[id: (\d+); name: (.*?); status: (.*?); exit: (\d+); error: (.*?); workDir: (.*?)\]"
    id = 0
    name = None
    status = None
    exit = None
    error = None
    workdir = None
    first_log = Decimal('Infinity')
    last_log = Decimal(0)
    pids = set() # list of pids belonging to the task
    files = set() # list of files touched by the task

    def __init__(self) -> None:
        self.id = 0
        self.name = ''
        self.status = ''
        self.exit = ''
        self.error = ''
        self.workdir = ''
        self.first_log = Decimal('Infinity')
        self.last_log = Decimal(0)
        self.pids = set()
        self.files = set()

logical_tasks = dict() # maps logical task name to set of task objects
tasks = set()
files = set() # all relevant files

folders = [] # all folders including relevant monitoring data

# find folders recursively that include the relevant data
while True:
    if len(folderstocheck) == 0:
        break

    curfolder = folderstocheck.pop()

    if not os.path.isdir(curfolder):
        continue

    cgroupspath = os.path.join(curfolder, 'cgroups.csv')
    bpfpath = os.path.join(curfolder, 'file-bpf.csv')

    # check if the necessary files exist
    if os.path.isfile(cgroupspath) and os.path.isfile(bpfpath):
        folders.append(curfolder)

    folderstocheck += [os.path.join(curfolder, sub) for sub in os.listdir(curfolder) if not sub.startswith('.')]

logging.info('Identifying tasks from Nextflow log.')

# parse nextflow log, getting task names and their work directories
with open(nflogpath, 'r') as nflog:
    for line in nflog:
        pattern = r"TaskHandler\[id: (\d+); name: (.*?); status: (.*?); exit: (\d+); error: (.*?); workDir: (.*?)\]"
        match = re.search(pattern, line)

        if match:
            t = Task()
            t.id = int(match.group(1))
            t.name = match.group(2)
            t.status = match.group(3)
            t.exit = match.group(4)
            t.error = match.group(5)
            t.workdir = match.group(6).removeprefix(globalworkdirprefix)
            tasks.add(t)

            # association to logical task
            pattern = r"(.*?) \((.*?)\)"
            match = re.search(pattern, t.name)
            logical_name = match.group(1) if match else t.name
            if logical_name in logical_tasks:
                logical_tasks[logical_name].add(t)
            else:
                logical_tasks[logical_name] = {t}

logging.info('%d tasks identified.', len(tasks))

# this loop parses all the data and puts it in the above structures and variables
for folder in folders:
    logging.info('Starting parsing on folder "%s". Parsing cgroups file.', folder)
    cgroupspath = os.path.join(folder, 'cgroups.csv')
    bpfpath = os.path.join(folder, 'file-bpf.csv')

    # a dictionary mapping a cgroupid to a set of PIDs
    cgroupdict = dict()

    # a dictionary mapping a pid to its cgroupid
    pid_cgroup_dict = dict()

    # a dictionary of all relevant task pids mapping to the task object
    pid_task_dict = dict()

    # dictionary mapping task relevant inodes to file objects
    inode_file_dict = dict()

    path_inode_dict = dict()

    with open(cgroupspath) as cgroupsfile:
        for row in csv.reader(cgroupsfile):
            cgroupid = int(row[CGROUPS_CGROUP].strip())
            pid = int(row[CGROUPS_CHILD].strip())
            pid_cgroup_dict[pid] = cgroupid

            if cgroupid in cgroupdict:
                cgroupdict[cgroupid].add(pid)
            else:
                newset = set()
                newset.add(pid)
                cgroupdict[cgroupid] = newset

    logging.info('Done parsing cgroups. Start associating tasks.')

    # loop through the BPF file twice:
    # 1. only checking open calls to find the readonly opens of command.sh file to connect pid and cgroup to task
    # 2. get all the file access information for files accessed by nextflow tasks

    with open(bpfpath) as bpffile:
        line = 0

        for row in csv.reader(bpffile):
            line += 1

            if row[0][0] == '#' or row[0][0] == 't':
                continue # row commented out or header

            row[BPF_PATH] = '_'.join(row[BPF_PATH:]) # in case the path contains ',' chars
            row[BPF_PATH] = row[BPF_PATH].split(':')[0] # ':' used in some cases for testing stuff in the path, only the first part is relevant
            row = [x.strip() for x in row]
            curpid = row[BPF_PID] = int(row[BPF_PID])

            # there are several files in a tasks directory that are exclusively written or read by the task owning the directory
            if row[BPF_OPERATION] != 'O':
                continue
            elif not (row[BPF_PATH].endswith('/.command.sh') and int(row[BPF_FLAGS], 16) & 0b11 == 0) and not (row[BPF_PATH].endswith('/.versions.yml') and int(row[BPF_FLAGS], 16) & 0b11 == 1):
                continue

            for task in tasks:
                if row[BPF_PATH].startswith(task.workdir):
                    logging.debug('File %s opened by PID %d belongs to task %s.', row[BPF_PATH], curpid, task.name)
                    if curpid not in pid_cgroup_dict:
                        logging.warning('cgroup for Main-PID %d for task "%s" not found. Skipping ...', curpid, task.name)
                        break
                    cgroupid = pid_cgroup_dict[curpid]
                    for pid in cgroupdict[pid_cgroup_dict[curpid]]: # all pids belonging to the task (also includes the main pid)
                        pid_task_dict[pid] = task
                        task.pids.add(pid)
                    logging.debug('Task has cgroupid %d which includes %d PIDs.', cgroupid, len(task.pids))
                    break

    logging.info('Tasks associated. Proceeding to IO access parsing.')

    with open(bpfpath) as bpffile:
        line = 0

        for row in csv.reader(bpffile):
            line += 1

            if row[0][0] == '#' or row[0][0] == 't':
                continue # row commented out or header

            row[BPF_PATH] = '_'.join(row[BPF_PATH:]) # in case the path contains ',' chars
            row[BPF_PATH] = row[BPF_PATH].split(':')[0] # ':' used in some cases for testing stuff in the path, only the first part is relevant
            row = [x.strip() for x in row]
            pid = row[BPF_PID] = int(row[BPF_PID])

            if pid not in pid_task_dict:
                continue # pid does not belong to a task -> not interesting / relevant

            task = pid_task_dict[pid]
            file = None
            time = row[BPF_TIME] = Decimal(row[BPF_TIME])
            inode = row[BPF_INODE] = int(row[BPF_INODE])
            row[BPF_FILEHANDLE] = int(row[BPF_FILEHANDLE])
            offset = row[BPF_OFFSET] = int(row[BPF_OFFSET])
            result = row[BPF_RESULT] = int(row[BPF_RESULT])
            size = row[BPF_SIZE] = int(row[BPF_SIZE])
            operation = row[BPF_OPERATION]

            if time < task.first_log:
                task.first_log = time
            if time > task.last_log:
                task.last_log = time

            if operation == 'U': # unlink (delete) does not have the inode set correctly, so take it from path->inode dict
                if row[BPF_PATH] in path_inode_dict:
                    inode = path_inode_dict[row[BPF_PATH]]
                else:
                    logging.warning('No inode found for "%s" (%s:%d), skipping file.', row[BPF_PATH], bpfpath, line)
                    continue

            if inode in inode_file_dict:
                file = inode_file_dict[inode]
            else:
                file = File()
                file.inode = inode
                inode_file_dict[inode] = file
                files.add(file)

            pid_task_dict[pid].files.add(file)
            accesses = file.accesses.get(pid, FileAccesses(pid))

            if time < accesses.first_log:
                accesses.first_log = time
            if time > accesses.last_log:
                accesses.last_log = time

            if operation == 'O':
                accesses.opens.append(row)
                accesses.path = row[BPF_PATH]
                if file.path is None or file.path == '':
                    file.path = accesses.path
                elif file.path != accesses.path:
                    #logging.debug('File with path %s opened with different path %s.', file.path, accesses.path)
                    pass
                path_inode_dict[file.path] = inode
            elif operation == 'C':
                accesses.closes.append(row)
            elif operation == 'U':
                accesses.deletes.append(row)
            elif operation == 'R':
                accesses.reads.append(row)
                if result > 0 and result <= size: # no read error
                    file.total_reads += 1
                    file.total_read_size += result
                    file.filesize = max(file.filesize, offset + result)
            elif operation == 'W':
                accesses.writes.append(row)
                if result > 0 and result <= size: # no write error
                    file.total_writes += 1
                    file.total_write_size += result
                    file.filesize = max(file.filesize, offset + result)

            file.accesses[pid] = accesses

    logging.info('Done parsing file-bpf.')

logging.info('Parsing finished, proceeding to data preparation.')

def analyse_accesses(accessarray):
    result = AccessAnalysis()
    if len(accessarray) > 0:
        accessed = P.empty()
        result.first_access = accessarray[0][BPF_TIME]
        result.last_access = accessarray[-1][BPF_TIME]
        last_ptr = accessarray[0][BPF_OFFSET]

        for access in accessarray:
            accessinterval = P.closedopen(access[BPF_OFFSET], access[BPF_OFFSET] + access[BPF_SIZE])
            result.total_bytes += access[BPF_SIZE]
            if accessinterval.lower < last_ptr:
                result.seek_down = True
            elif accessinterval.lower > last_ptr:
                result.seek_up = True
            last_ptr = accessinterval.upper
            repetitive_access = accessed & accessinterval
            result.repetitive_bytes += sum([x.upper - x.lower for x in repetitive_access])
            accessed = accessed | accessinterval

        result.unique_bytes = sum([x.upper - x.lower for x in accessed])
        if result.unique_bytes > 0:
            result.min_offset = accessed.lower
            result.max_offset = accessed.upper

    return result

# file access pattern analysis
for file in files:
    #if not filter_function(file): # problem if filter relies on pattern already
    #    continue
    for accesses in file.accesses.values():
        accesses.analysis_reads = analyse_accesses(accesses.reads)
        if accesses.analysis_reads.unique_bytes > 0:
            accesses.pattern = accesses.pattern | PATTERN_READ
            accesses.pattern = accesses.pattern | (PATTERN_RANDOM if (accesses.analysis_reads.seek_down or accesses.analysis_reads.seek_up) else PATTERN_SEQUENTIAL)
        accesses.analysis_writes = analyse_accesses(accesses.writes)
        if accesses.analysis_writes.unique_bytes > 0:
            accesses.pattern = accesses.pattern | PATTERN_WRITE
            accesses.pattern = accesses.pattern | (PATTERN_RANDOM if (accesses.analysis_writes.seek_down or accesses.analysis_writes.seek_up) else PATTERN_SEQUENTIAL)

        # old
        ##reads
        #ptr = None
        #cur_pattern = PATTERN_SEQUENTIAL
        #for r in accesses.reads:
        #    if r[BPF_RESULT] > 0 and r[BPF_RESULT] <= r[BPF_SIZE]: # no error
        #        accesses.pattern = accesses.pattern | PATTERN_READ
        #        cur_pattern = PATTERN_RANDOM if ptr is not None and ptr != r[BPF_OFFSET] else PATTERN_SEQUENTIAL
        #        ptr = r[BPF_OFFSET] + r[BPF_SIZE]
        #if ptr is not None:
        #    accesses.pattern = accesses.pattern | cur_pattern
        #
        ## writes
        #ptr = None
        #cur_pattern = PATTERN_SEQUENTIAL
        #for r in accesses.writes:
        #    if r[BPF_RESULT] > 0 and r[BPF_RESULT] <= r[BPF_SIZE]: # no error
        #        accesses.pattern = accesses.pattern | PATTERN_WRITE
        #        cur_pattern = PATTERN_RANDOM if ptr is not None and ptr != r[BPF_OFFSET] else PATTERN_SEQUENTIAL
        #        ptr = r[BPF_OFFSET] + r[BPF_SIZE]
        #if ptr is not None:
        #    accesses.pattern = accesses.pattern | cur_pattern

logging.info('Data preparation finished, proceeding to evaluation.')

# ----------------------------------------------------------------------------------------------------------------------
# evals

def show_or_save_figure(figname, extension = 'pdf'):
    plt.savefig('figures/' + figname + '.' + extension, bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    plt.close()

def list_logical_tasks():
    for logical_name, taskset in logical_tasks.items():
        print(len(taskset), 'physical tasks for logical task', logical_name)

def list_pids_per_task():
    for t in tasks:
        print('Task ', t.name, ':', sep='')
        for p in sorted(t.pids):
            print("\t", p, sep='')
    print()

def list_files_per_task():
    for t in tasks:
        print('Task ', t.name, ':', sep='')
        for f in t.files:
            if not file_filter(f):
                continue
            relevant_pids = t.pids.intersection(f.accesses.keys())
            pattern = 0
            for pid in relevant_pids:
                pattern = pattern | f.accesses[pid].pattern
            pattern_str = ''
            if pattern & PATTERN_READ != 0:
                pattern_str += ' | PATTERN_READ'
            if pattern & PATTERN_WRITE != 0:
                pattern_str += ' | PATTERN_WRITE'
            if pattern & PATTERN_SEQUENTIAL != 0:
                pattern_str += ' | PATTERN_SEQUENTIAL'
            if pattern & PATTERN_RANDOM != 0:
                pattern_str += ' | PATTERN_RANDOM'

            #print("\t", sep='')
            print("\t", f.path, ' (fs ', f.filesize, ' bytes, ', f.total_reads, ' reads ', f.total_read_size, ' bytes, ', f.total_writes, ' writes ', f.total_write_size, ' bytes) - ', pattern_str.removeprefix(' | '), sep='')
    print()

def list_files_per_logical_task():
    for logical_name, tasks in logical_tasks.items():
        print('Logical Task ', logical_name, ' (', len(tasks), ' physical tasks):', sep='')
        for t in tasks:
            for f in t.files:
                if not file_filter(f):
                    continue
                relevant_pids = t.pids.intersection(f.accesses.keys())
                pattern = 0
                for pid in relevant_pids:
                    pattern = pattern | f.accesses[pid].pattern
                pattern_str = ''
                if pattern & PATTERN_READ != 0:
                    pattern_str += ' | PATTERN_READ'
                if pattern & PATTERN_WRITE != 0:
                    pattern_str += ' | PATTERN_WRITE'
                if pattern & PATTERN_SEQUENTIAL != 0:
                    pattern_str += ' | PATTERN_SEQUENTIAL'
                if pattern & PATTERN_RANDOM != 0:
                    pattern_str += ' | PATTERN_RANDOM'

                #print("\t", sep='')
                print("\t", f.path, ' (fs ', f.filesize, ' bytes, ', f.total_reads, ' reads ', f.total_read_size, ' bytes, ', f.total_writes, ' writes ', f.total_write_size, ' bytes) - ', pattern_str.removeprefix(' | '), sep='')
    print()

def output_analysis(analysis, task_runtime):
    print('\t\tTotal bytes accessed:', analysis.total_bytes)
    print('\t\tUnique bytes accessed:', analysis.unique_bytes)
    print('\t\tRepititive bytes accessed:', analysis.repetitive_bytes)
    print('\t\tWhole access range: [', analysis.min_offset, ';', analysis.max_offset, ']', sep='')
    print('\t\tSeeks up / down:', analysis.seek_up, '/', analysis.seek_down)
    access_period = analysis.last_access - analysis.first_access
    print('\t\tBulkishness (less is more bulky): ', format(Decimal(100) * Decimal(access_period) / Decimal(task_runtime), '.2f'), '% (', format(access_period, '.2f'), '/', format(task_runtime, '0.2f'), ')', sep='')

def list_access_properties_per_logical_task(physical = False, relevant_only = True):
    for logical_name, tasks in logical_tasks.items():
        if not relevant_only and not physical:
            print('Logical Task ', logical_name, ' (', len(tasks), ' physical tasks):', sep='')
        task_relevant = not relevant_only
        for t in tasks:
            if not relevant_only and physical:
                print('Task ', t.name, ' (runtime ', task_runtime, '):', sep='')
            task_runtime = t.last_log - t.first_log
            for f in t.files:
                relevant_pids = t.pids.intersection(f.accesses.keys())
                for pid in relevant_pids:
                    accesses = f.accesses[pid]
                    if not accesses_filter(accesses):
                        continue
                    if not task_relevant:
                        task_relevant = True
                        if physical:
                            print('Task ', t.name, ' (runtime ', task_runtime, '):', sep='')
                        else:
                            print('Logical Task ', logical_name, ' (', len(tasks), ' physical tasks):', sep='')
                    if accesses.analysis_reads.unique_bytes > 0:
                        print('\tRead', accesses.path, 'from PID', pid)
                        print('\t\tAccesses:', len(accesses.reads))
                        output_analysis(accesses.analysis_reads, task_runtime)
                    if accesses.analysis_writes.unique_bytes > 0:
                        print('\tWritten', accesses.path, 'from PID', pid)
                        print('\t\tAccesses:', len(accesses.writes))
                        output_analysis(accesses.analysis_writes, task_runtime)
    print()

def internal_print_filesize_stats(filesizes):
    if len(filesizes) > 0:
        print("\t# Files:", len(filesizes))
        print("\tMinimum:", min(filesizes))
        print("\tMaximum:", max(filesizes))
        print("\tMedian:", statistics.median(filesizes))
        print("\tMean:", statistics.mean(filesizes))
        print("\tStdev:", statistics.stdev(filesizes))
    else:
        print("\tNo files accessed.")

def print_filesize_stats(remove_0):
    filesizes = sum([[f.filesize for f in t.files if not remove_0 or f.filesize > 0] for t in tasks.values()], [])
    print('Filesize statistics of all files read from and written to:')
    internal_print_filesize_stats(filesizes)
    print()

def print_filesize_stats_per_task(remove_0):
    for t in tasks:
        print('Filesize statistics of task ', t.name, ':', sep='')
        filesizes = sum([[f.filesize] for f in t.files if not remove_0 or f.filesize > 0], [])
        internal_print_filesize_stats(filesizes)
    print()

def output_filesuffix_statistics():
    endings = dict()

    for f in files:
        if not file_filter(f):
            continue
        last_path_part = f.path.split('/')[-1]
        ending = None
        if last_path_part.count('.') > 0:
            ending = last_path_part.split('.')[-1]
        if ending in endings:
            endings[ending] += 1
        else:
            endings[ending] = 1

    for ending, count in sorted(endings.items(), key=lambda item: item[1], reverse=True):
        print(count, '-', ending)

    print()

def histo_total_filesizes():
    filesizes = sum([[f.filesize for f in t.files if file_filter(f)] for t in tasks], [])
    plt.figure()
    plt.hist(filesizes, 100)
    show_or_save_figure('histo_filesizes_all')

def histo_filesizes_per_task():
    for t in tasks:
        filesizes = sum([[f.filesize] for f in t.files if file_filter(f)], [])
        if len(filesizes) > 0:
            plt.figure()
            plt.hist(filesizes, 100)
            show_or_save_figure('histo_filesizes_task_' + t.name)

def graph_file_accesses(filepath):
    graphcount = 0
    plt.figure()

    for f in files:
        plot = False
        for accesses in f.accesses.values():
            if accesses.path == filepath:
                plot = True
                break

        if plot:
            for pid, accesses in f.accesses.items():
                reads = [r for r in accesses.reads if r[BPF_RESULT] > 0 and r[BPF_RESULT] <= r[BPF_SIZE]]
                if len(reads) > 0:
                    plt.vlines([Decimal(r[BPF_TIME]) for r in reads], [r[BPF_OFFSET] for r in reads], [r[BPF_OFFSET] + r[BPF_RESULT] for r in reads], label='reads of ' + str(pid), color='C' + str(graphcount))
                    graphcount += 1
                writes = [r for r in accesses.writes if r[BPF_RESULT] > 0 and r[BPF_RESULT] <= r[BPF_SIZE]]
                if len(writes) > 0:
                    plt.vlines([Decimal(r[BPF_TIME]) for r in writes], [r[BPF_OFFSET] for r in writes], [r[BPF_OFFSET] + r[BPF_RESULT] for r in writes], label='writes of ' + str(pid), color='C' + str(graphcount))
                    graphcount += 1

    plt.legend()
    show_or_save_figure('file_accesses_' + filepath.replace('/', '_'))

def graph_accesses_from_pids_by_task_and_path(tasks, path, normalize_length = True, normalize_on_access = False):
    graphcount = 0
    plt.figure()

    for task in tasks:
        task_runtime = Decimal(task.last_log - task.first_log)

        for file in task.files:
            relevant_pids = task.pids.intersection(file.accesses.keys())
            for pid in relevant_pids:
                accesses = file.accesses[pid]
                if accesses.path != path:
                    continue
                start_correction = accesses.first_log if normalize_on_access else task.first_log
                length_correction = 1
                if normalize_length:
                    length_correction = Decimal(accesses.last_log - accesses.first_log) if normalize_on_access else task_runtime

                reads = [r for r in accesses.reads if r[BPF_RESULT] > 0 and r[BPF_RESULT] <= r[BPF_SIZE]]
                if len(reads) > 0:
                    plt.vlines([Decimal(100) * (Decimal(r[BPF_TIME]) - Decimal(start_correction)) / length_correction for r in reads], [r[BPF_OFFSET] for r in reads], [r[BPF_OFFSET] + r[BPF_RESULT] for r in reads], label='reads of ' + str(pid), color='C' + str(graphcount))
                    graphcount += 1
                writes = [r for r in accesses.writes if r[BPF_RESULT] > 0 and r[BPF_RESULT] <= r[BPF_SIZE]]
                if len(writes) > 0:
                    plt.vlines([Decimal(100) * (Decimal(r[BPF_TIME]) - Decimal(start_correction)) / length_correction for r in writes], [r[BPF_OFFSET] for r in writes], [r[BPF_OFFSET] + r[BPF_RESULT] for r in writes], label='writes of ' + str(pid), color='C' + str(graphcount))
                    graphcount += 1

    plt.legend()
    show_or_save_figure('accesses_by_pid_' + path.replace('/', '_'))

def graph_accesses_from_tasks_by_task_and_path(tasks, path, normalize_length = True, normalize_on_access = False):
    graphcount = 0
    plt.figure()

    for task in tasks:
        task_runtime = Decimal(task.last_log - task.first_log)

        for file in task.files:
            relevant_pids = task.pids.intersection(file.accesses.keys())
            graphed = False
            for pid in relevant_pids:
                accesses = file.accesses[pid]
                if accesses.path != path:
                    continue
                start_correction = accesses.first_log if normalize_on_access else task.first_log
                length_correction = 1
                if normalize_length:
                    length_correction = Decimal(accesses.last_log - accesses.first_log) if normalize_on_access else task_runtime
                taskname = task.name.split(':')[-1].split(' ')[0]
                offset_scale = 1024*1024 # MiB

                reads = [r for r in accesses.reads if r[BPF_RESULT] > 0 and r[BPF_RESULT] <= r[BPF_SIZE]]
                if len(reads) > 0:
                    plt.vlines([Decimal(100) * (Decimal(r[BPF_TIME]) - Decimal(start_correction)) / length_correction for r in reads], [r[BPF_OFFSET] / offset_scale for r in reads], [(r[BPF_OFFSET] + r[BPF_RESULT]) / offset_scale for r in reads], label='reads of task ' + taskname, color='C' + str(graphcount))
                    graphed = True
                writes = [r for r in accesses.writes if r[BPF_RESULT] > 0 and r[BPF_RESULT] <= r[BPF_SIZE]]
                if len(writes) > 0:
                    plt.vlines([Decimal(100) * (Decimal(r[BPF_TIME]) - Decimal(start_correction)) / length_correction for r in writes], [r[BPF_OFFSET] / offset_scale for r in writes], [(r[BPF_OFFSET] + r[BPF_RESULT]) / offset_scale for r in writes], label='writes of task ' + taskname, color='C' + str(graphcount+1))
                    graphed = True
            if graphed:
                graphcount += 2

    plt.ylabel('file offset in MiB')

    if normalize_length:
        plt.xlim(0, 101)
        if normalize_on_access:
            plt.xlabel('% of total access duration')
        else:
            plt.xlabel('% of total task duration')
    else:
        plt.xlabel('time after task start')

    plt.legend()
    show_or_save_figure('accesses_by_task_' + path.replace('/', '_'))

def graph_bulkishness_by_task(physical = False):
    COLOR_READ = 'C1'
    COLOR_WRITE = 'C2'
    categories = []
    values = []
    colors = []
    plt.figure()
    for logical_name, tasks in logical_tasks.items():
        for t in tasks:
            task_runtime = t.last_log - t.first_log
            for f in t.files:
                relevant_pids = t.pids.intersection(f.accesses.keys())
                for pid in relevant_pids:
                    accesses = f.accesses[pid]
                    if not accesses_filter(accesses):
                        continue
                    taskname = (t.name if physical else logical_name).split(':')[-1]
                    if not physical:
                        taskname = taskname + ' x ' + str(len(tasks))
                    if accesses.analysis_reads.unique_bytes > 0:
                        bulkishness = Decimal(100) * Decimal(accesses.analysis_reads.last_access - accesses.analysis_reads.first_access) / Decimal(task_runtime)
                        categories.append(taskname)
                        values.append(bulkishness)
                        colors.append(COLOR_READ)
                    if accesses.analysis_writes.unique_bytes > 0:
                        bulkishness = Decimal(100) * Decimal(accesses.analysis_writes.last_access - accesses.analysis_writes.first_access) / Decimal(task_runtime)
                        categories.append(taskname)
                        values.append(bulkishness)
                        colors.append(COLOR_WRITE)
    plt.scatter(categories, values, color=colors, s=20, alpha=0.6)
    plt.xticks(rotation=90)
    plt.ylim(0, 101)
    #plt.yscale('log')

    # Create custom legend
    write_dots = mlines.Line2D([], [], color=COLOR_WRITE, marker='o', linestyle='None', markersize=10, label='written file')
    read_dots = mlines.Line2D([], [], color=COLOR_READ, marker='o', linestyle='None', markersize=10, label='read file')
    plt.legend(handles=[write_dots, read_dots])
    if physical:
        plt.xlabel('task')
    else:
        plt.xlabel('LOGICAL_TASK x <number of physical instances>')
    plt.ylabel('fraction of runtime for data accesses in %')

    show_or_save_figure('bulkishness_by' + ('' if physical else '_logical') + '_task')

def graph_bulkishness_by_total_access_bytes():
    COLOR_READ = 'C1'
    COLOR_WRITE = 'C2'
    accesssizes = []
    bulkishnesses = []
    colors = []
    plt.figure()
    for t in tasks:
        task_runtime = t.last_log - t.first_log
        for f in t.files:
            relevant_pids = t.pids.intersection(f.accesses.keys())
            for pid in relevant_pids:
                accesses = f.accesses[pid]
                if not accesses_filter(accesses):
                    continue
                if accesses.analysis_reads.unique_bytes > 0:
                    bulkishness = Decimal(100) * Decimal(accesses.analysis_reads.last_access - accesses.analysis_reads.first_access) / Decimal(task_runtime)
                    accesssizes.append(accesses.analysis_reads.total_bytes)
                    bulkishnesses.append(bulkishness)
                    colors.append(COLOR_READ)
                if accesses.analysis_writes.unique_bytes > 0:
                    bulkishness = Decimal(100) * Decimal(accesses.analysis_writes.last_access - accesses.analysis_writes.first_access) / Decimal(task_runtime)
                    accesssizes.append(accesses.analysis_writes.total_bytes)
                    bulkishnesses.append(bulkishness)
                    colors.append(COLOR_WRITE)
    plt.scatter(accesssizes, bulkishnesses, color=colors)
    plt.xscale('log')
    #plt.yscale('log')
    show_or_save_figure('bulkishness_by_total_access_bytes')

def graph_bulkishness_histogram(bins = 20):
    bulkishnesses_read = []
    bulkishnesses_write = []
    plt.figure()
    for t in tasks:
        task_runtime = t.last_log - t.first_log
        for f in t.files:
            relevant_pids = t.pids.intersection(f.accesses.keys())
            for pid in relevant_pids:
                accesses = f.accesses[pid]
                if not accesses_filter(accesses):
                    continue
                if accesses.analysis_reads.unique_bytes > 0:
                    bulkishness = Decimal(100) * Decimal(accesses.analysis_reads.last_access - accesses.analysis_reads.first_access) / Decimal(task_runtime)
                    bulkishnesses_read.append(bulkishness)
                if accesses.analysis_writes.unique_bytes > 0:
                    bulkishness = Decimal(100) * Decimal(accesses.analysis_writes.last_access - accesses.analysis_writes.first_access) / Decimal(task_runtime)
                    bulkishnesses_write.append(bulkishness)
    plt.hist([bulkishnesses_read, bulkishnesses_write], bins=bins, alpha=0.5, label=['Reads', 'Writes'])
    show_or_save_figure('bulkishness_histogram')

# normal outputs
#list_logical_tasks()
#list_pids_per_task()
#list_files_per_task()
#list_files_per_logical_task()
#list_access_properties_per_logical_task(True)
#list_access_properties_per_logical_task()
#print_filesize_stats(True)
#print_filesize_stats_per_task(True)
#output_filesuffix_statistics()

# graphs
#histo_total_filesizes()
#histo_filesizes_per_task()
#graph_file_accesses('/soeren/work/0d/ceac11f41c3ffd22736ab14ebd6ab8/WT_REP1_primary_2.fastq.gz')
# ... for rnaseq
#graph_accesses_from_pids_by_task_and_path(tasks, '/home/witzke/nf-rnaseq/outdir/work/71/cc3b1de5d901297c04c0e72a70fed6/RAP1_UNINDUCED_REP2.markdup.sorted.bam', False, True)
graph_accesses_from_tasks_by_task_and_path(tasks, '/home/witzke/nf-rnaseq/outdir/work/49/61bd6655adfca9e6eb791ab25c8b7a/WT_REP1_2_val_2.fq.gz', True, False)
#graph_accesses_from_tasks_by_task_and_path(tasks, '/home/witzke/nf-rnaseq/outdir/work/71/cc3b1de5d901297c04c0e72a70fed6/RAP1_UNINDUCED_REP2.markdup.sorted.bam', True, False)
# ... for mag
#graph_accesses_from_tasks_by_task_and_path(tasks, '/home/witzke/nf-rnaseq/outdir/work/7b/7919bcca0a04602f50b3f16cbbdb87/SPAdes-test_minigut_sample2_scaffolds.fasta', False, True)
#graph_accesses_from_tasks_by_task_and_path(tasks, '/home/witzke/nf-rnaseq/outdir/work/b3/3a44209cdcc66adf4676cafae1d42f/test_minigut_sample2.gff', False, True)

graph_bulkishness_by_task(physical=False)
#graph_bulkishness_by_task(physical=True)
#graph_bulkishness_by_total_access_bytes()
#graph_bulkishness_histogram(20)

logging.info('Script finished.')
