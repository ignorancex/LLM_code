import json
from zac.ds.architecture import Architecture
from zac.zac import ZAC
from zac.simulator.simulator import Simulator
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_spec', metavar='S', type=str, help='experiment specficiation')
    args = parser.parse_args()
    with open(args.exp_spec, 'r') as f:
        exp_spec = json.load(f)
    benchmark_set = []
    for name in exp_spec["qasm_list"]:
        if os.path.isfile(name):
            # file exists
            benchmark_set.append(name)
        if os.path.isdir(name):
            # directory exists
            for filename in os.listdir(name):
                f = os.path.join(name, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    benchmark_set.append(f)
    list_zac_setting = exp_spec["zac_setting"]
    to_run_simulation = exp_spec["simulation"]

    dict_arch = dict()

    for benchmark in benchmark_set:
        print("==============================================")
        print("Compile circuit {}".format(benchmark))
        
        filename = benchmark.split('/')[-1]
        filename = filename.split('.')[0]
        
        for zac_setting in list_zac_setting:
            if zac_setting["arch_spec"] in dict_arch:
                (arch, spec) = dict_arch[zac_setting["arch_spec"]]
            else:
                with open(zac_setting["arch_spec"], 'r') as f:
                    spec = json.load(f)
                arch = Architecture(spec)
                arch.preprocessing() 
                dict_arch[zac_setting["arch_spec"]] = (arch, spec)

            zac_setting["name"] = filename

            zac_compiler = ZAC()
            zac_compiler.parse_setting(zac_setting)
            zac_compiler.set_architecture_spec_path(zac_setting["arch_spec"])

            zac_compiler.set_architecture(arch)
            zac_compiler.set_program(benchmark)
            # construct directory for result and time profiling
            directory = zac_compiler.dir+"code"
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = zac_compiler.dir+"time"
            if not os.path.exists(directory):
                os.makedirs(directory)
            code_dict = zac_compiler.solve(save_file=True)
            code_file = zac_compiler.code_filename
            with open(code_file, 'r') as f:
                result = json.load(f)
            if to_run_simulation:
                # set arch fidelity 
                # run simulation
                simulator = Simulator()
                simulator.set_arch_spec(spec)
                simulator.parse(zac_compiler.code_filename)
                fideilty_result = simulator.simulate()
                # continue
                # construct directory for fidelity result
                directory = zac_compiler.dir+"fidelity"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                tmp = zac_compiler.dir + f"fidelity/{zac_setting['name']}_fidelity.json"
                with open(tmp, 'w') as f:  
                    json.dump(fideilty_result, f, indent = 2)
                

            if exp_spec["animation"]:
                # construct directory for fidelity animation
                directory = zac_compiler.dir+"animation"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                tmp =  zac_compiler.dir + f"animation/{zac_setting['name']}.mp4"
                zac_compiler.animate(code_dict, output=tmp)